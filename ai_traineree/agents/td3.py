from ai_traineree.utils import to_tensor
from ai_traineree import DEVICE
from ai_traineree.agents.utils import hard_update, soft_update
from ai_traineree.buffers import ReplayBuffer
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.networks.heads import DoubleCritic
from ai_traineree.noise import OUProcess
from ai_traineree.types import AgentType

import numpy as np
import random
import torch
from torch.optim import AdamW
from torch.nn.functional import mse_loss
from typing import Any, List, Sequence, Tuple


class TD3Agent(AgentType):
    """
    Twin Delayed Deep Deterministic (TD3) Policy Gradient.

    In short, it's a slightly modified/improved version of the DDPG. Compared to the DDPG in this package,
    which uses Guassian noise, this TD3 uses Ornstein–Uhlenbeck process as the noise.
    """

    name = "TD3"

    def __init__(
        self, state_size: int, action_size: int, hidden_layers: Sequence[int]=(128, 128),
        actor_lr: float=1e-3, critic_lr: float=1e-3,
        noise_scale: float=0.2, noise_sigma: float=0.1,
        device=None, **kwargs
    ):
        self.device = device if device is not None else DEVICE

        # Reason sequence initiation.
        self.action_size = action_size
        self.hidden_layers = kwargs.get('hidden_layers', hidden_layers)
        self.actor = ActorBody(state_size, action_size, hidden_layers=hidden_layers).to(self.device)
        self.critic = DoubleCritic(state_size, action_size, CriticBody, hidden_layers=hidden_layers).to(self.device)
        self.target_actor = ActorBody(state_size, action_size, hidden_layers=hidden_layers).to(self.device)
        self.target_critic = DoubleCritic(state_size, action_size, CriticBody, hidden_layers=hidden_layers).to(self.device)

        # Noise sequence initiation
        # self.noise = GaussianNoise(shape=(action_size,), mu=1e-8, sigma=noise_sigma, scale=noise_scale, device=device)
        self.noise = OUProcess(shape=action_size, scale=noise_scale, sigma=noise_sigma, device=self.device)

        # Target sequence initiation
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # Optimization sequence initiation.
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=critic_lr)
        self.max_grad_norm_actor: float = float(kwargs.get("max_grad_norm_actor", 10.0))
        self.max_grad_norm_critic: float = float(kwargs.get("max_grad_norm_critic", 10.0))
        self.action_min = kwargs.get('action_min', -1.)
        self.action_max = kwargs.get('action_max', 1.)
        self.action_scale = kwargs.get('action_scale', 1.)

        self.gamma: float = float(kwargs.get('gamma', 0.99))
        self.tau: float = float(kwargs.get('tau', 0.02))
        self.batch_size: int = int(kwargs.get('batch_size', 64))
        self.buffer_size: int = int(kwargs.get('buffer_size', int(1e5)))
        self.buffer = ReplayBuffer(self.batch_size, self.buffer_size)

        self.warm_up = int(kwargs.get('warm_up', 0))
        self.update_freq = int(kwargs.get('update_freq', 1))
        self.update_policy_freq = int(kwargs.get('update_policy_freq', 1))
        self.number_updates = int(kwargs.get('number_updates', 1))
        self.noise_reset_freq = int(kwargs.get('noise_reset_freq', 10000))

        # Breath, my child.
        self.reset_agent()
        self.iteration = 0
        self.actor_loss: float = 0.
        self.critic_loss: float = 0.

    def reset_agent(self) -> None:
        self.actor.reset_parameters()
        self.critic.reset_parameters()
        self.target_actor.reset_parameters()
        self.target_critic.reset_parameters()

    def act(self, state, epsilon: float=0.0, training_mode=True) -> List[float]:
        """
        Agent acting on observations.

        When the training_mode is True (default) a noise is added to each action. 
        """
        # Epsilon greedy
        if epsilon > 0 and random.random() < epsilon:
            rnd_actions = torch.rand(self.action_size)*(self.action_max - self.action_min) - self.action_min
            return rnd_actions.tolist()

        with torch.no_grad():
            state = to_tensor(state).float().to(self.device)
            action = self.actor(state)
            if training_mode:
                action += self.noise.sample()
            return (self.action_scale*torch.clamp(action, self.action_min, self.action_max)).tolist()

    def target_act(self, staten, noise: float=0.0):
        with torch.no_grad():
            staten = to_tensor(staten).float().to(self.device)
            action = self.target_actor(staten) + noise*self.noise.sample()
            return torch.clamp(action, self.action_min, self.action_max).cpu().numpy().astype(np.float32)

    def step(self, state, action, reward, next_state, done):
        self.iteration += 1
        self.buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        if (self.iteration % self.noise_reset_freq) == 0:
            self.noise.reset_states()

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) <= self.batch_size:
            return

        if not (self.iteration % self.update_freq) or not (self.iteration % self.update_policy_freq):
            for _ in range(self.number_updates):
                # Note: Inside this there's a delayed policy update.
                #       Every `update_policy_freq` it will learn `number_updates` times.
                self.learn(self.buffer.sample())

    def learn(self, experiences):
        """Update critics and actors"""
        rewards = to_tensor(experiences['reward']).float().to(self.device).unsqueeze(1)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device).unsqueeze(1)
        states = to_tensor(experiences['state']).float().to(self.device)
        actions = to_tensor(experiences['action']).to(self.device)
        next_states = to_tensor(experiences['next_state']).float().to(self.device)

        if (self.iteration % self.update_freq) == 0:
            self._update_value_function(states, actions, rewards, next_states, dones)

        if (self.iteration % self.update_policy_freq) == 0:
            self._update_policy(states)

            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)

    def _update_value_function(self, states, actions, rewards, next_states, dones):
        # critic loss
        next_actions = self.target_actor.act(next_states)
        Q_target_next = torch.min(*self.target_critic.act(next_states, next_actions))
        Q_target = rewards + (self.gamma * Q_target_next * (1 - dones))
        Q1_expected, Q2_expected = self.critic(states, actions)
        critic_loss = mse_loss(Q1_expected, Q_target) + mse_loss(Q2_expected, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm_critic)
        self.critic_optimizer.step()
        self.critic_loss = critic_loss.item()

    def _update_policy(self, states):
        # Compute actor loss
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions)[0].mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_actor)
        self.actor_optimizer.step()
        self.actor_loss = actor_loss.item()

    def describe_agent(self) -> Tuple[Any, Any, Any, Any]:
        """
        Returns network's weights in order:
        Actor, TargetActor, Critic, TargetCritic
        """
        return (self.actor.state_dict(), self.target_actor.state_dict(), self.critic.state_dict(), self.target_critic())

    def log_writer(self, writer, episode):
        writer.add_scalar("loss/actor", self.actor_loss, episode)
        writer.add_scalar("loss/critic", self.critic_loss, episode)

    def save_state(self, path: str):
        agent_state = dict(
            actor=self.actor.state_dict(), target_actor=self.target_actor.state_dict(),
            critic=self.critic.state_dict(), target_critic=self.target_critic.state_dict(),
        )
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self.actor.load_state_dict(agent_state['actor'])
        self.critic.load_state_dict(agent_state['critic'])
        self.target_actor.load_state_dict(agent_state['target_actor'])
        self.target_critic.load_state_dict(agent_state['target_critic'])
