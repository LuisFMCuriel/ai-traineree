from ai_traineree.types import AgentType
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai_traineree import DEVICE
from ai_traineree.agents.utils import soft_update
from ai_traineree.buffers import NStepBuffer, PERBuffer
from ai_traineree.networks.heads import RainbowNet
from ai_traineree.utils import to_tensor
from typing import Callable, Dict, List, Optional, Sequence, Union


class RainbowAgent(AgentType):
    """Rainbow agent as described in [1].

    Rainbow is a DQN agent with some improvments that were suggested before 2017.
    As mentioned by the authors it's not exhaustive improvment but all changes are in
    relatively separate areas so their connection makes sense. These improvements are:
    * Priority Experience Replay
    * Multi-step
    * Double Q net
    * Dueling nets
    * NoisyNet
    * CategoricalNet for Q estimate

    Consider this class as a particular version of the DQN agent.

    [1] "Rainbow: Combining Improvements in Deep Reinforcement Learning" by Hessel et al. (DeepMind team)
    https://arxiv.org/abs/1710.02298
    """

    name = "Rainbow"

    def __init__(
        self,
        input_shape: Union[Sequence[int], int],
        output_shape: Union[Sequence[int], int],
        state_transform: Optional[Callable]=None,
        reward_transform: Optional[Callable]=None,
        **kwargs
    ):
        """
        A wrapper over the DQN thus majority of the logic is in the DQNAgent.
        Special treatment is required because the Rainbow agent uses categorical nets
        which operate on probability distributions. Each action is taken as the estimate
        from such distributions.

        Parameters:
            input_shape (tuple of ints): Most likely that's your *state* shape.
            output_shape (tuple of ints): Most likely that's you *action* shape.
            pre_network_fn (function that takes input_shape and returns network):
                Used to preprocess state before it is used in the value- and advantage-function in the dueling nets.
            lr (default: 1e-3): Learning rate value.
            gamma (default: 0.99): Discount factor.
            tau (default: 0.002): Soft-copy factor.

        """
        self.device = self._register_param(kwargs, "device", DEVICE)
        self.input_shape: Sequence[int] = input_shape if not isinstance(input_shape, int) else (input_shape,)

        self.in_features: int = self.input_shape[0]
        self.output_shape: Sequence[int] = output_shape if not isinstance(output_shape, int) else (output_shape,)
        self.out_features: int = self.output_shape[0]

        self.lr = float(self._register_param(kwargs, 'lr', 3e-4))
        self.gamma = float(self._register_param(kwargs, 'gamma', 0.99))
        self.tau = float(self._register_param(kwargs, 'tau', 0.002))
        self.update_freq = int(self._register_param(kwargs, 'update_freq', 1))
        self.batch_size = int(self._register_param(kwargs, 'batch_size', 80))
        self.buffer_size = int(self._register_param(kwargs, 'buffer_size', 1e5))
        self.warm_up = int(self._register_param(kwargs, 'warm_up', 0))
        self.number_updates = int(self._register_param(kwargs, 'number_updates', 1))
        self.max_grad_norm = float(self._register_param(kwargs, 'max_grad_norm', 10))

        self.iteration: int = 0
        self.using_double_q = bool(self._register_param(kwargs, "using_double_q", True))

        self.state_transform = state_transform if state_transform is not None else lambda x: x
        self.reward_transform = reward_transform if reward_transform is not None else lambda x: x

        v_min = float(self._register_param(kwargs, "v_min", -10))
        v_max = float(self._register_param(kwargs, "v_max", 10))
        self.n_atoms = int(self._register_param(kwargs, "n_atoms", 21))
        self.z_atoms = torch.linspace(v_min, v_max, self.n_atoms, device=self.device)
        self.z_delta = self.z_atoms[1] - self.z_atoms[0]

        self.buffer = PERBuffer(batch_size=self.batch_size, buffer_size=self.buffer_size)
        self.__batch_indices = torch.arange(self.batch_size, device=self.device)

        self.n_steps = self._register_param(kwargs, "n_steps", 3)
        self.n_buffer = NStepBuffer(n_steps=self.n_steps, gamma=self.gamma)

        # Note that in case a pre_network is provided, e.g. a shared net that extracts pixels values,
        # it should be explicitly passed in kwargs
        self.net = RainbowNet(self.input_shape, self.output_shape, num_atoms=self.n_atoms, batch_size=self.batch_size, **kwargs)
        self.target_net = RainbowNet(self.input_shape, self.output_shape, num_atoms=self.n_atoms, batch_size=self.batch_size, **kwargs)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.dist_probs = None
        self._loss = float('inf')

    @property
    def loss(self):
        return {'loss': self._loss}

    @loss.setter
    def loss(self, value):
        if isinstance(value, dict):
            value = value['loss']
        self._loss = value

    def step(self, state, action, reward, next_state, done) -> None:
        """Letting the agent to take a step.

        On some steps the agent will initiate learning step. This is dependent on
        the `update_freq` value.

        Parameters:
            state: S(t)
            action: A(t)
            reward: R(t)
            nexxt_state: S(t+1)
            done: (bool) Whether the state is terminal. 

        """
        self.iteration += 1
        state = to_tensor(self.state_transform(state)).float().to("cpu")
        next_state = to_tensor(self.state_transform(next_state)).float().to("cpu")
        reward = self.reward_transform(reward)

        # Delay adding to buffer to account for n_steps (particularly the reward)
        self.n_buffer.add(state=state.numpy(), action=[int(action)], reward=[reward], done=[done], next_state=next_state.numpy())
        if not self.n_buffer.available:
            return

        self.buffer.add(**self.n_buffer.get().get_dict())

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) >= self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample())

            # Update networks only once - sync local & target
            soft_update(self.target_net, self.net, self.tau)

    def act(self, state, eps: float = 0.) -> int:
        """Returns actions for given state as per current policy.

        Parameters:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if np.random.random() < eps:
            return np.random.randint(self.out_features)

        state = to_tensor(self.state_transform(state)).float().unsqueeze(0).to(self.device)
        # state = to_tensor(self.state_transform(state)).float().to(self.device)
        self.dist_probs = self.net.act(state)
        q_values = (self.dist_probs * self.z_atoms).sum(-1)
        return int(q_values.argmax(-1))  # Action maximizes state-action value Q(s, a)

    def learn(self, experiences: Dict[str, List]) -> None:
        """
        Parameters:
            experiences: Contains all experiences for the agent. Typically sampled from the memory buffer.
                Five keys are expected, i.e. `state`, `action`, `reward`, `next_state`, `done`.
                Each key contains a array and all arrays have to have the same length.

        """
        rewards = to_tensor(experiences['reward']).float().to(self.device)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device)
        states = to_tensor(experiences['state']).float().to(self.device)
        next_states = to_tensor(experiences['next_state']).float().to(self.device)
        actions = to_tensor(experiences['action']).type(torch.long).to(self.device)
        assert rewards.shape == dones.shape == (self.batch_size, 1)
        assert states.shape == next_states.shape == (self.batch_size, self.in_features)
        assert actions.shape == (self.batch_size, 1)  # Discrete domain

        with torch.no_grad():
            prob_next = self.target_net.act(next_states)
            q_next = (prob_next * self.z_atoms).sum(-1) * self.z_delta
            if self.using_double_q:
                duel_prob_next = self.net.act(next_states)
                a_next = torch.argmax((duel_prob_next * self.z_atoms).sum(-1), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)

            prob_next = prob_next[self.__batch_indices, a_next, :]

        m = self.net.dist_projection(rewards, 1 - dones, self.gamma ** self.n_steps, prob_next)
        assert m.shape == (self.batch_size, self.n_atoms)

        log_prob = self.net(states, log_prob=True)
        assert log_prob.shape == (self.batch_size, self.out_features, self.n_atoms)
        log_prob = log_prob[self.__batch_indices, actions.squeeze(), :]
        assert log_prob.shape == m.shape == (self.batch_size, self.n_atoms)

        # Cross-entropy loss error and the loss is batch mean
        error = -torch.sum(m * log_prob, 1)
        assert error.shape == (self.batch_size,)
        loss = error.mean()
        assert loss >= 0

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._loss = float(loss.item())

        if hasattr(self.buffer, 'priority_update'):
            assert (~torch.isnan(error)).any()
            self.buffer.priority_update(experiences['index'], error.detach().cpu().numpy())

        # Update networks - sync local & target
        soft_update(self.target_net, self.net, self.tau)

    def describe_agent(self) -> Dict:
        """Returns agent's state dictionary.

        Returns:
            State dicrionary for internal networks.

        """
        return self.net.state_dict()

    def log_writer(self, writer, iteration, full_mode=False):
        writer.add_scalar("loss/agent", self._loss, iteration)

        if full_mode and self.dist_probs is not None:
            for action_idx in range(self.out_features):
                dist = self.dist_probs[0, action_idx]
                writer.add_scalar(f'dist/expected_{action_idx}', (dist*self.z_atoms).sum(), iteration)
                writer.add_histogram_raw(
                    f'dist/Q_{action_idx}', min=self.z_atoms[0], max=self.z_atoms[-1], num=len(self.z_atoms),
                    sum=dist.sum(), sum_squares=dist.pow(2).sum(), bucket_limits=self.z_atoms+self.z_delta,
                    bucket_counts=dist, global_step=iteration
                )

        # This method, `log_writer`, isn't executed on every iteration but just in case we delay plotting weights.
        # It simply might be quite costly. Thread wisely.
        if full_mode:
            for idx, layer in enumerate(self.net.value_net.layers):
                if hasattr(layer, "weight"):
                    writer.add_histogram(f"value_net/layer_weights_{idx}", layer.weight, iteration)
                if hasattr(layer, "bias") and layer.bias is not None:
                    writer.add_histogram(f"value_net/layer_bias_{idx}", layer.bias, iteration)
            for idx, layer in enumerate(self.net.advantage_net.layers):
                if hasattr(layer, "weight"):
                    writer.add_histogram(f"advantage_net/layer_{idx}", layer.weight, iteration)
                if hasattr(layer, "bias") and layer.bias is not None:
                    writer.add_histogram(f"advantage_net/layer_bias_{idx}", layer.bias, iteration)

    def save_state(self, path: str) -> None:
        """Saves agent's state into a file.

        Parameters:
            path: String path where to write the state.

        """
        agent_state = dict(
            net=self.net.state_dict(),
            target_net=self.target_net.state_dict(),
            config=self._config,
        )
        torch.save(agent_state, path)

    def load_state(self, path: str) -> None:
        """Loads state from a file under provided path.

        Parameters:
            path: String path indicating where the state is stored.

        """
        agent_state = torch.load(path)
        self._config = agent_state.get('config', {})
        self.__dict__.update(**self._config)

        self.net.load_state_dict(agent_state['net'])
        self.target_net.load_state_dict(agent_state['target_net'])

    def save_buffer(self, path: str) -> None:
        """Saves data from the buffer into a file under provided path.

        Parameters:
            path: String path where to write the buffer.

        """
        import json
        dump = self.buffer.dump_buffer(serialize=True)
        with open(path, 'w') as f:
            json.dump(dump, f)

    def load_buffer(self, path: str) -> None:
        """Loads data into the buffer from provided file path.

        Parameters:
            path: String path indicating where the buffer is stored.

        """
        import json
        with open(path, 'r') as f:
            buffer_dump = json.load(f)
        self.buffer.load_buffer(buffer_dump)
