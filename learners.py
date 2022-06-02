import gym

import env
import highway_env
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from typing import Tuple
from env import ObsType


class LearnerType:
    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"
    DQN_CNN = "DQN_CNN"

def make_learner_fn(learner_type: LearnerType, obs_type):
    learner_map = {
        ObsType.Kinematics: {
            LearnerType.DQN: LearnerFactory.DQN_Kinematics,
            LearnerType.PPO: LearnerFactory.PPO_Kinematics,
            LearnerType.A2C: LearnerFactory.A2C_Kinematics
            
        },
        ObsType.GrayscaleObservation:{
            LearnerType.DQN_CNN: LearnerFactory.DQN_GrayscaleObservation
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
    def DQN_GrayscaleObservation(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "DQN_GrayscaleObservation"
        model = DQN('CnnPolicy', env,
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log="./log/")
        return learner_name, model
    
    

    @classmethod
    def PPO_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        # assert isinstance(env, SubprocVecEnv)
        # n_envs = env.num_envs
        n_envs = 1
        learner_name = "PPO_Kinematics"
        batch_size = 64
        model = PPO('MlpPolicy', env,
                    learning_rate=5e-4,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12 // n_envs,
                    batch_size=batch_size,
                    n_epochs=10,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log="./log/")
        return learner_name, model

    @classmethod
    def A2C_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "A2C_Kinematics"
        model = A2C('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    learning_rate=5e-4,
                    gamma=0.8,
                    n_steps=6,
                    verbose=2,
                    tensorboard_log="./log/")
        return learner_name, model

