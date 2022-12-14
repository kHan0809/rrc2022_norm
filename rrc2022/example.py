"""Example policy for Real Robot Challenge 2022"""
import numpy as np
import os
import torch

from rrc_2022_datasets import PolicyBase

from . import policies
import torch
import torch.nn as nn


class TorchBasePolicy(PolicyBase):
    def __init__(
        self,
        torch_model,
        action_space,
        observation_space,
        episode_length,
    ):
        self.action_space = action_space
        self.device = "cpu"
        self.policy = torch_model
        self.policy.eval()

        # load torch script
        self.action_space = action_space
        self.obs_dim = observation_space.shape[0]
        self.action_max = self.action_space.high[0]
        self.init_state_count = 0


    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        observation = (observation - self.o_mean) @ self.o_cov
        if self.init_state_count == 0:
            self.stack_state = reset_state(observation)
        else:
            self.stack_state = stack_state(self.stack_state,observation,self.obs_dim)

        action = self.policy(self.stack_state.unsqueeze(0))
        action = action.detach().numpy()[0]
        action *= self.action_max
        return action


class TorchPushPolicyExpert(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        path = policies.get_path("push_expert")
        self.policy = PolicyNet(observation_space.shape[0]*3, action_space.shape[0])
        self.policy.load_state_dict(torch.load(str(path / "trifinger-cube-push-real-expert-v0_model.pt"), map_location=torch.device('cpu'))["pi"])
        self.o_mean = np.load(str(path / "o_mean.npy"))
        self.o_cov = np.load(str(path / "o_cov.npy"))
        self.o_mean = torch.tensor(self.o_mean, dtype=torch.float, device="cpu")
        self.o_cov = torch.tensor(self.o_cov, dtype=torch.float, device="cpu")

        super().__init__(self.policy, action_space, observation_space, episode_length)

class TorchPushPolicyMixed(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        path = policies.get_path("push_mixed")
        self.policy = PolicyNet(observation_space.shape[0]*3, action_space.shape[0])
        self.policy.load_state_dict(torch.load(str(path / "trifinger-cube-push-real-mixed-v0_model.pt"), map_location=torch.device('cpu'))["pi"])
        self.o_mean = np.load(str(path / "o_mean.npy"))
        self.o_cov = np.load(str(path / "o_cov.npy"))
        self.o_mean = torch.tensor(self.o_mean, dtype=torch.float, device="cpu")
        self.o_cov = torch.tensor(self.o_cov, dtype=torch.float, device="cpu")

        super().__init__(self.policy, action_space, observation_space, episode_length)


class TorchLiftPolicyExpert(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        path = policies.get_path("lift_expert")
        self.policy = PolicyNet(observation_space.shape[0]*3, action_space.shape[0])
        self.policy.load_state_dict(torch.load(str(path / "trifinger-cube-lift-real-expert-v0_model.pt"), map_location=torch.device('cpu'))["pi"])
        self.o_mean = np.load(str(path / "o_mean.npy"))
        self.o_cov = np.load(str(path / "o_cov.npy"))
        self.o_mean = torch.tensor(self.o_mean, dtype=torch.float, device="cpu")
        self.o_cov = torch.tensor(self.o_cov, dtype=torch.float, device="cpu")

        super().__init__(self.policy, action_space, observation_space, episode_length)

class TorchLiftPolicyMixed(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """
    def __init__(self, action_space, observation_space, episode_length):
        path = policies.get_path("lift_mixed")
        self.policy = PolicyNet(observation_space.shape[0]*3, action_space.shape[0])
        self.policy.load_state_dict(torch.load(str(path / "trifinger-cube-lift-real-mixed-v0_model1000000.pt"),map_location=torch.device('cpu'))["pi"])
        self.o_mean = np.load(str(path / "o_mean.npy"))
        self.o_cov = np.load(str(path / "o_cov.npy"))
        self.o_mean = torch.tensor(self.o_mean, dtype=torch.float, device="cpu")
        self.o_cov = torch.tensor(self.o_cov, dtype=torch.float, device="cpu")

        super().__init__(self.policy, action_space, observation_space, episode_length)

class PolicyNet(nn.Module):
    def __init__(self, o_dim,a_dim):
        super(PolicyNet, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.fc1 = nn.Linear(o_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, a_dim)
        self.relu1 = nn.GELU()
        self.relu2 = nn.GELU()
        self.tanh  = nn.Tanh()

    def forward(self, o):
        layer = self.relu1(self.fc1(o))
        layer = self.relu2(self.fc2(layer))
        action = self.tanh(self.fc3(layer))
        return action


def reset_state(obs):
    new_obs=torch.cat((obs,obs,obs))
    return new_obs
def stack_state(pre_stack_obs,obs,o_dim):
    new_obs = np.cat((pre_stack_obs[o_dim:],obs))
    return new_obs