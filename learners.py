import gym
import highway_env
from stable_baselines3 import DQN, PPO, DDPG, TD3, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Tuple
from env import ObsType


class LearnerType:
    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"
    DDPG = "DDPG"
    TD3 = "TD3"

def make_learner_fn(learner_type: LearnerType, obs_type):
    learner_map = {
        ObsType.Kinematics: {
            LearnerType.DQN: LearnerFactory.DQN_Kinematics,
            LearnerType.PPO: LearnerFactory.PPO_Kinematics,
            LearnerType.A2C: LearnerFactory.A2C_Kinematics,
            LearnerType.DDPG: LearnerFactory.DDPG_Kinematics,
            LearnerType.TD3: LearnerFactory.TD3_Kinematics,
        }
    }
    return learner_map[obs_type][learner_type]

class LearnerFactory(object):
    @classmethod
    def DQN_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "DQN_Kinematics"
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log="./log/")
        return learner_name, model

    @classmethod
    def PPO_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "PPO_Kinematics"
        model = PPO('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256, dict(vf=[256], pi=[16])]),
                    learning_rate=5e-4,
                    batch_size=32,
                    gamma=0.8,
                    verbose=1,
                    tensorboard_log="./log/")
        return learner_name, model

    @classmethod
    def DDPG_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "DDPG_Kinematics"
        model = DDPG('MlpPolicy', env,
                    policy_kwargs=dict(qf=[256, 256], pi=[128, 128]),
                    learning_rate=5e-4,
                    batch_size=32,
                    buffer_size=15000,
                    train_freq=(100, 'step'),
                    gradient_steps=-1,
                    gamma=0.8,
                    verbose=1,
                    tensorboard_log="./log/")
        return learner_name, model

    @classmethod
    def A2C_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "A2C_Kinematics"
        model = A2C('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256, dict(vf=[256], pi=[16])]),
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=1,
                    tensorboard_log="./log/")
        return learner_name, model

    @classmethod
    def TD3_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "TD3_Kinematics"
        model = TD3('MlpPolicy', env,
                    policy_kwargs=dict(qf=[256, 256], pi=[128, 128]),
                    learning_rate=5e-4,
                    batch_size=32,
                    buffer_size=15000,
                    train_freq=(100, 'step'),
                    gradient_steps=-1,
                    gamma=0.8,
                    verbose=1,
                    tensorboard_log="./log/")
        return learner_name, model
