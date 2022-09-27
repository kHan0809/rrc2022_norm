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
        self.action_max = self.action_space.high[0]


    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        observation = (observation - self.o_mean) / (self.o_std + 1e-3)
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        action = self.policy(observation.unsqueeze(0))
        action = action.detach().numpy()[0]
        action *= self.action_max
        return action


class TorchPushPolicyExpert(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        self.policy = PolicyNet(observation_space.shape[0], action_space.shape[0])
        self.policy.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+"/policies/trifinger-cube-push-real-mixed-v0_model.pt",map_location=torch.device('cpu'))["pi"])
        self.o_mean = np.load(os.path.dirname(os.path.abspath(__file__)) + "/policies/trifinger-cube-push-real-expert-v0/o_mean.npy")
        self.o_std  = np.load(os.path.dirname(os.path.abspath(__file__)) + "/policies/trifinger-cube-push-real-expert-v0/o_std.npy")
        super().__init__(self.policy, action_space, observation_space, episode_length)

class TorchPushPolicyMixed(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        self.policy = PolicyNet(observation_space.shape[0], action_space.shape[0])
        self.policy.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+"/policies/trifinger-cube-push-real-mixed-v0_model.pt",map_location=torch.device('cpu'))["pi"])
        self.o_mean = np.load(os.path.dirname(os.path.abspath(__file__)) + "/policies/trifinger-cube-push-real-mixed-v0/o_mean.npy")
        self.o_std  = np.load(os.path.dirname(os.path.abspath(__file__)) + "/policies/trifinger-cube-push-real-mixed-v0/o_std.npy")
        super().__init__(self.policy, action_space, observation_space, episode_length)


class TorchLiftPolicyExpert(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        self.policy = PolicyNet(observation_space.shape[0], action_space.shape[0])
        self.policy.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+"/policies/trifinger-cube-lift-real-expert-v0_model.pt",map_location=torch.device('cpu'))["pi"])
        self.o_mean = np.load(os.path.dirname(os.path.abspath(__file__)) + "/policies/o_mean.npy")
        self.o_std = np.load(os.path.dirname(os.path.abspath(__file__)) + "/policies/o_std.npy")
        super().__init__(self.policy, action_space, observation_space, episode_length)

class TorchLiftPolicyMixed(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        self.policy = PolicyNet(observation_space.shape[0], action_space.shape[0])
        self.policy.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+"/policies/trifinger-cube-lift-real-mixed-v0True-940.pt",map_location=torch.device('cpu'))["policy"])
        self.o_mean = np.load(os.path.dirname(os.path.abspath(__file__)) + "/policies/trifinger-cube-lift-real-mixed-v0/o_mean.npy")
        self.o_std = np.load(os.path.dirname(os.path.abspath(__file__)) + "/policies/trifinger-cube-lift-real-mixed-v0/o_std.npy")
        super().__init__(self.policy, action_space, observation_space, episode_length)

class PolicyNet(nn.Module):
    def __init__(self, o_dim,a_dim):
        super(PolicyNet, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.fc1 = nn.Linear(o_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, a_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.tanh  = nn.Tanh()

    def forward(self, o):
        layer = self.relu1(self.fc1(o))
        layer = self.relu2(self.fc2(layer))
        action = self.tanh(self.fc3(layer))
        return action