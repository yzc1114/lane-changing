from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, DDPG, A2C, TD3
from env import make_env, ObsType


class GridOccupancyCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(GridOccupancyCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def do_test():
    env = make_env(1, ObsType.OccupancyGrid)
    policy_kwargs = dict(
        features_extractor_class=GridOccupancyCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[256, dict(pi=[256, 64], vf=[256, 256])]
    )
    batch_size = 64
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=5e-4,
                n_steps=batch_size,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.8,
                verbose=2,
                tensorboard_log="./log/")
    model.learn(5000)


if __name__ == '__main__':
    do_test()