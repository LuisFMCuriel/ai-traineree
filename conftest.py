import copy
import mock
import numpy as np
import pytest
import random

from typing import Any, List, Sequence, Tuple


class MockContinuousSpace:
    def __init__(self, *args):
        self.shape = args


class MockDiscreteSpace:
    def __init__(self, n):
        self.n = n

    def __call__(self):
        return self.n


@pytest.fixture
def fix_env_discrete():
    rnd_state = lambda: random.choices(range(10), k=5)  # noqa
    mock_env = mock.Mock()
    mock_env.reset.return_value = rnd_state()
    mock_env.step.return_value = (rnd_state(), 0, False, "")
    mock_env.observation_space = np.array((4, 2))
    mock_env.action_space = MockDiscreteSpace(2)
    return mock_env


@pytest.fixture
def fix_env():
    rnd_state = lambda: random.choices(range(10), k=5)  # noqa
    mock_env = mock.Mock()
    mock_env.reset.return_value = rnd_state()
    mock_env.step.return_value = (rnd_state(), 0, False, "")
    mock_env.observation_space = np.array((4, 2))
    mock_env.action_space = MockContinuousSpace(2, 4)
    return mock_env


def deterministic_interactions(agent, state_size=4, num_iters=50):
    state = [0]*state_size
    next_state = copy.copy(state)
    actions = []
    for i in range(num_iters):
        action = agent.act(state)
        actions.append(action)

        next_state[i % state_size] = (next_state[i % state_size] + 1) % 2
        reward = (i % 4 - 2) / 2.
        done = (i + 1) % 100 == 0

        agent.step(state, action, reward, next_state, done)
        state = copy.copy(next_state)
    return actions


def fake_step(step_shape: Sequence[int]) -> Tuple[List[Any], float, bool]:
    state = np.random.random(step_shape).tolist()
    reward = random.random()
    terminal = random.random() > 0.8
    return state, reward, terminal
