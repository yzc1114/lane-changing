import time

import highway_env
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from typing import Callable
from copy import deepcopy


kinematics_env_config = {
    "id": "highway-fast-v0",
    "import_module": "highway_env",
    "lanes_count": 3,
    "vehicles_count": 50,  # 环境车数量
    "duration": 50,  # 每个episode的step数
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        # "order": "shuffled"
    },
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
}

GrayscaleObservation_env_config = {
    "id": "highway-fast-v0",
    "import_module": "highway_env",
    "lanes_count": 3,
    "vehicles_count": 50,  # 环境车数量
    "duration": 50,  # 每个episode的step数
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "observation": {
        "type": "GrayscaleObservation",
        "vehicles_count": 5,
        "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
        "stack_size": 4,
        "observation_shape": (128, 64),
        "order": "shuffled"
    },
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
}
GrayscaleKinematicsObservation_env_config ={
    "id": "highway-fast-v0",
    "import_module": "highway_env",
    "lanes_count": 3,
    "vehicles_count": 50,  # 环境车数量
    "duration": 50,  # 每个episode的step数
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "observation": {
        "type": "GrayscaleKinematicsObservation",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
        "stack_size": 4,
        "observation_shape": (128, 64),
        "order": "shuffled"
    },
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
}


class ObsType:
    GrayscaleObservation = 0
    Kinematics = 1
    GrayscaleKinematicsObservation = 2


    obs_type_2_config = {
        Kinematics: kinematics_env_config,
        GrayscaleObservation:GrayscaleObservation_env_config,
        GrayscaleKinematicsObservation:GrayscaleKinematicsObservation_env_config
    }

def make_env(num=1, obs_type=ObsType.Kinematics, seed=None):
    if seed is None:
        seed = int(time.time())
    envs = []
    # Create the vectorized environment
    def make_env_fn(rank: int, seed: int = 0) -> Callable:
        def _init() -> gym.Env:
            _env = gym.make("highway-fast-v0")
            config = deepcopy(ObsType.obs_type_2_config[obs_type])
            _env.unwrapped.configure(config)
            _env.reset()
            _env = Monitor(_env)
            envs.append(_env)
            _env.seed(seed + rank)
            return _env

        set_random_seed(seed)
        return _init
    if num == 1:
        return make_env_fn(1, seed)()
    env = SubprocVecEnv([make_env_fn(i, i+seed) for i in range(num)])
    return env


if __name__ == '__main__':
    env = make_env(1)
    print(env.config)