import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ai_traineree import DEVICE
from ai_traineree.agents.utils import EPS, compute_gae, normalize, revert_norm_returns
from ai_traineree.buffers import ReplayBuffer
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.policies import DirichletPolicy, MultivariateGaussianPolicySimple
from ai_traineree.types import AgentType
from ai_traineree.utils import to_tensor
from typing import Tuple


class PPOAgent(AgentType):
    """
    Proximal Policy Optimization (PPO) [1] is an online policy gradient method
    that could be considered as an implementation-wise simplified version of
    the Trust Region Policy Optimization (TRPO).


    [1] "Proximal Policy Optimization Algorithms" (2017) by J. Schulman, F. Wolski,
        P. Dhariwal, A. Radford, O. Klimov. https://arxiv.org/abs/1707.06347
    """

    name = "PPO"

    def __init__(self, state_size: int, action_size: int, hidden_layers=(200, 200), device=None, **kwargs):
        """
        Parameters:
            is_discrete: (default: False) Whether return discrete action.
            kl_div: (default: False) Whether to use KL divergence in loss.
            using_gae: (default: True) Whether to use General Advantage Estimator.
            gae_lambda: (default: 0.9) Value of \lambda in GAE.
            actor_lr: (default: 0.0003) Learning rate for the actor (policy).
            critic_lr: (default: 0.001) Learning rate for the critic (value function).
            actor_betas: (default: (0.9, 0.999) Adam's betas for actor optimizer.
            critic_betas: (default: (0.9, 0.999) Adam's betas for critic optimizer.
            gamma: (default: 0.99) Discount value.
            ppo_ratio_clip: (default: 0.25) Policy ratio clipping value.
            rollout_length: (default: 48) Number of actions to take before update.
            batch_size: (default: rollout_length) Number of samples used in learning.
            number_updates: (default: 1) How many times to learn from a rollout.
            entropy_weight: (default: 0.005) Weight of the entropy term in the loss.
            value_loss_weight: (default: 0.005) Weight of the entropy term in the loss.

        """
        self.device = device if device is not None else DEVICE

        self.state_size = state_size
        self.action_size = action_size
        self.iteration = 0

        self.is_discrete = bool(kwargs.get("is_discrete", False))
        self.kl_div = bool(kwargs.get("kl_div", False))
        self.kl_beta = 0.1
        self.using_gae = bool(kwargs.get("using_gae", True))
        self.gae_lambda = float(kwargs.get("gae_lambda", 0.9))

        self.actor_lr = float(kwargs.get('actor_lr', 3e-4))
        self.actor_betas: Tuple[float, float] = kwargs.get('actor_betas', (0.9, 0.999))
        self.critic_lr = float(kwargs.get('critic_lr', 1e-3))
        self.critic_betas: Tuple[float, float] = kwargs.get('critic_betas', (0.9, 0.999))
        self.gamma = float(kwargs.get("gamma", 0.99))
        self.ppo_ratio_clip = float(kwargs.get("ppo_ratio_clip", 0.25))

        self.executor_num = int(kwargs.get("executor_num", 1))  # TODO: Is this the right name?
        self.rollout_length = int(kwargs.get("rollout_length", 48))  # "Much less than the episode length"
        self.batch_size = int(kwargs.get("batch_size", self.rollout_length))
        self.number_updates = int(kwargs.get("number_updates", 1))
        self.entropy_weight = float(kwargs.get("entropy_weight", 0.5))
        self.value_loss_weight = float(kwargs.get("value_loss_weight", 1.0))

        self.local_memory_buffer = {}
        self.memory = ReplayBuffer(batch_size=self.rollout_length, buffer_size=self.rollout_length)

        self.action_scale: float = float(kwargs.get("action_scale", 1))
        self.action_min: float = float(kwargs.get("action_min", -1))
        self.action_max: float = float(kwargs.get("action_max", 1))
        self.max_grad_norm_actor: float = float(kwargs.get("max_grad_norm_actor", 100.0))
        self.max_grad_norm_critic: float = float(kwargs.get("max_grad_norm_critic", 100.0))

        self.hidden_layers = kwargs.get('hidden_layers', hidden_layers)
        # self.policy = DirichletPolicy()  # TODO: Apparently Beta dist is better than Normal in PPO. Leaving for validation.
        self.policy = MultivariateGaussianPolicySimple(action_size, self.batch_size, device=self.device)
        self.actor = ActorBody(state_size, self.policy.param_dim*action_size, hidden_layers=self.hidden_layers, device=self.device)
        self.critic = CriticBody(state_size, action_size, self.hidden_layers).to(self.device)

        self.actor_params = list(self.actor.parameters()) + list(self.policy.parameters())
        self.critic_params = list(self.critic.parameters())

        self.actor_opt = optim.Adam(self.actor_params, lr=self.actor_lr, betas=self.actor_betas)
        self.critic_opt = optim.Adam(self.critic_params, lr=self.critic_lr, betas=self.critic_betas)
        self.actor_loss = 0
        self.critic_loss = 0

    def __clear_memory(self):
        self.memory = ReplayBuffer(batch_size=self.rollout_length, buffer_size=self.rollout_length)

    def act(self, state, epsilon: float=0.):
        actions = []
        logprobs = []
        values = []
        with torch.no_grad():
            state = to_tensor(state).view(self.executor_num, self.state_size).float().to(self.device)
            for executor in range(self.executor_num):
                actor_est = self.actor.act(state[executor].unsqueeze(0))
                assert not torch.any(torch.isnan(actor_est))

                dist = self.policy(actor_est)
                action = dist.sample()
                value = self.critic.act(state[executor].unsqueeze(0), action)
                logprob = self.policy.log_prob(dist, action)
                values.append(value)
                logprobs.append(logprob)

                if self.is_discrete:  # *Technically* it's the max of Softmax but that's monotonic.
                    action = int(torch.argmax(action))
                else:
                    # TODO: This *makes sense* but seems that some environments work better without.
                    #       Should we leave min/scale/max to the policy learning?
                    # action = torch.clamp(action*self.action_scale, self.action_min, self.action_max).cpu()
                    action = action.numpy().flatten().tolist()
                actions.append(action)

            self.local_memory_buffer['value'] = torch.cat(values)
            self.local_memory_buffer['logprob'] = torch.cat(logprobs)
            assert len(actions) == self.executor_num
            return actions if self.executor_num > 1 else actions[0]

    def step(self, states, actions, rewards, next_state, done, **kwargs):
        self.iteration += 1

        self.memory.add(
            state=torch.tensor(states).reshape(self.executor_num, self.state_size).float(),
            action=torch.tensor(actions).reshape(self.executor_num, self.action_size).float(),
            reward=torch.tensor(rewards).reshape(self.executor_num, 1),
            done=torch.tensor(done).reshape(self.executor_num, 1),
            logprob=self.local_memory_buffer['logprob'].reshape(self.executor_num, 1),
            value=self.local_memory_buffer['value'].reshape(self.executor_num, 1),
        )

        if self.iteration % self.rollout_length == 0:
            self.train()
            self.__clear_memory()

    def train(self):
        """
        Main loop that initiates the training.
        """
        experiences = self.memory.sample()
        rewards = to_tensor(experiences['reward']).to(self.device)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device)
        states = to_tensor(experiences['state']).to(self.device)
        actions = to_tensor(experiences['action']).to(self.device)
        values = to_tensor(experiences['value']).to(self.device)
        logprobs = to_tensor(experiences['logprob']).to(self.device)
        assert rewards.shape == dones.shape == values.shape == logprobs.shape
        assert states.shape == (self.rollout_length, self.executor_num, self.state_size), f"Wrong state shape: {states.shape}"
        assert actions.shape == (self.rollout_length, self.executor_num, self.action_size), f"Wrong action shape: {actions.shape}"

        # Normalize values. Keep mean and std to update next_value estimate.
        values_mean, values_std = values.mean(dim=0), values.std(dim=0)
        values = (values - values_mean) / torch.clamp(values_std, EPS)

        with torch.no_grad():
            if self.using_gae:
                next_value = (self.critic.act(states[-1], actions[-1]) - values_mean) / torch.clamp(values_std, EPS)
                advantages = compute_gae(rewards, dones, values, next_value, self.gamma, self.gae_lambda)
                advantages = normalize(advantages)
                returns = advantages + values
                assert advantages.shape == returns.shape == values.shape
            else:
                returns = revert_norm_returns(rewards, dones, self.gamma)
                returns = returns.float()
                advantages = normalize(returns - values)
                assert advantages.shape == returns.shape == values.shape

        # Flatten all evaluation to pretend that they're independent samples
        states = states.view(-1, self.state_size)
        actions = actions.view(-1, self.action_size)
        logprobs = logprobs.view(-1, 1)
        returns = returns.view(-1, 1)
        dones = dones.view(-1, 1)
        advantages = advantages.view(-1, 1)

        all_indices = range(self.rollout_length * self.executor_num)
        for _ in range(self.number_updates):
            rand_ids = random.sample(all_indices, self.batch_size)
            self.learn((
                states[rand_ids].detach(),
                actions[rand_ids].detach(),
                logprobs[rand_ids].detach(),
                returns[rand_ids].detach(),
                advantages[rand_ids].detach()
            ))

    def learn(self, samples):
        states, actions, old_log_probs, returns, advantages = samples

        actor_est = self.actor(states.detach())
        dist = self.policy(actor_est)
        action_mu = dist.rsample()
        value = self.critic(states.detach(), action_mu.detach())

        if not self.using_gae:
            value = (value - value.mean(dim=0)) / torch.clamp(value.std(dim=0), 1e-8)
        assert value.shape == returns.shape

        entropy = dist.entropy()
        new_log_probs = self.policy.log_prob(dist, actions).view(-1, 1)
        assert new_log_probs.shape == old_log_probs.shape

        # advantages = advantages.unsqueeze(1)
        r_theta = (new_log_probs - old_log_probs).exp()
        r_theta_clip = torch.clamp(r_theta, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)
        assert r_theta.shape == r_theta_clip.shape

        if self.kl_div:
            # kl_div = F.kl_div(old_log_probs.exp(), new_log_probs.exp(), reduction='mean')  # Reverse KL, see [2]
            kl_div = torch.mean(new_log_probs.exp() * (new_log_probs - old_log_probs))  # Global mean
            policy_loss = -torch.mean(r_theta * advantages) + self.kl_beta * kl_div
        else:
            joint_theta_adv = torch.stack((r_theta * advantages, r_theta_clip * advantages))
            assert joint_theta_adv.shape[0] == 2
            policy_loss = -torch.amin(joint_theta_adv, dim=0).mean()
        entropy_loss = -self.entropy_weight * entropy.mean()

        # Update value and critic loss
        actor_loss = policy_loss + entropy_loss
        critic_loss = self.value_loss_weight * 0.5 * F.mse_loss(returns, value)

        # Update policy and actor loss
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm_actor)
        self.actor_opt.step()
        self.actor_loss = actor_loss.item()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm_critic)
        self.critic_opt.step()
        self.critic_loss = critic_loss.mean().item()

    def log_writer(self, writer, step):
        writer.add_scalar("loss/actor", self.actor_loss, step)
        writer.add_scalar("loss/critic", self.critic_loss, step)
        policy_params = {str(i): v for i, v in enumerate(itertools.chain.from_iterable(self.policy.parameters()))}
        writer.add_scalars("policy/param", policy_params, step)

        for idx, layer in enumerate(self.actor.layers):
            if hasattr(layer, "weight"):
                writer.add_histogram(f"actor/layer_weights_{idx}", layer.weight, step)
            if hasattr(layer, "bias") and layer.bias:
                writer.add_histogram(f"actor/layer_bias_{idx}", layer.bias, step)

        for idx, layer in enumerate(self.critic.layers):
            if hasattr(layer, "weight"):
                writer.add_histogram(f"critic/layer_weights_{idx}", layer.weight, step)
            if hasattr(layer, "bias") and layer.bias:
                writer.add_histogram(f"critic/layer_bias_{idx}", layer.bias, step)

    def save_state(self, path: str):
        agent_state = dict(policy=self.policy.state_dict(), actor=self.actor.state_dict(), critic=self.critic.state_dict())
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self.policy.load_state_dict(agent_state['policy'])
        self.actor.load_state_dict(agent_state['actor'])
        self.critic.load_state_dict(agent_state['critic'])
