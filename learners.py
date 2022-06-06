import gym

import env
import highway_env
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from typing import Tuple
from env import ObsType
from custom_model import GridOccupancyCNN
from models import *


class LearnerType:
    PPO = "PPO"
    A2C = "A2C"
    EGO = "EGO"
    DQN = "DQN"
    DQN_CNN = "DQN_CNN"

def make_learner_fn(learner_type: LearnerType, obs_type):
    learner_map = {
        ObsType.Kinematics: {
            LearnerType.DQN: LearnerFactory.DQN_Kinematics,
            LearnerType.PPO: LearnerFactory.PPO_Kinematics,
            LearnerType.A2C: LearnerFactory.A2C_Kinematics,
            LearnerType.EGO: LearnerFactory.EGO_Kinematics
            
        },
        ObsType.Grayscale:{
            LearnerType.DQN_CNN: LearnerFactory.DQN_GrayscaleObservation
        },
        ObsType.OccupancyGrid:{
            LearnerType.DQN: LearnerFactory.DQN_GridOccupancy,
            LearnerType.PPO: LearnerFactory.PPO_GridOccupancy,
            LearnerType.A2C: LearnerFactory.A2C_GridOccupancy,
        },
        ObsType.TimeToCollision:{
            LearnerType.DQN: LearnerFactory.DQN_TTC,
            LearnerType.PPO: LearnerFactory.PPO_TTC,
            LearnerType.A2C: LearnerFactory.A2C_TTC,
            LearnerType.EGO: LearnerFactory.EGO_TTC,

        }
    }
    return learner_map[obs_type][learner_type]

class LearnerFactory(object):
    @classmethod
    def DQN_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "DQN_Kinematics"
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[512, 256, 256, 256]),
                    learning_rate=5e-4,
                    buffer_size=int(1e6),
                    learning_starts=2000,
                    batch_size=64,
                    exploration_fraction=0.1,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log="./log/kinematics/")
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
                exploration_fraction=0.1,
                verbose=1,
                tensorboard_log="./log/")
        return learner_name, model
    


    @classmethod
    def PPO_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        # assert isinstance(env, SubprocVecEnv)
        # n_envs = env.num_envs
        learner_name = "PPO_Kinematics"
        batch_size = 64
        model = PPO('MlpPolicy', env,
                    learning_rate=5e-4,
                    policy_kwargs=dict(net_arch=[512, 256, dict(pi=[256, 64], vf=[256, 256])]),
                    n_steps=batch_size * 12,
                    batch_size=batch_size,
                    n_epochs=10,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log="./log/kinematics/")
        return learner_name, model

    @classmethod
    def A2C_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "A2C_Kinematics"
        model = A2C('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[512, 256, dict(pi=[256, 64], vf=[256, 256])]),
                    learning_rate=5e-4,
                    gamma=0.8,
                    n_steps=32,
                    verbose=2,
                    tensorboard_log="./log/kinematics/")
        return learner_name, model

    @classmethod
    def PPO_GridOccupancy(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "PPO_GridOccupancy"
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
                    n_steps=batch_size * 12,
                    batch_size=batch_size,
                    n_epochs=10,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log="./log/gridoccupancy/")
        return learner_name, model

    @classmethod
    def DQN_GridOccupancy(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "DQN_GridOccupancy"
        policy_kwargs = dict(
            features_extractor_class=GridOccupancyCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[512, 256, 256, 256]
        )
        model = DQN("MlpPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=5e-4,
                    buffer_size=int(1e6),
                    learning_starts=2000,
                    batch_size=64,
                    exploration_fraction=0.1,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log="./log/gridoccupancy/")
        return learner_name, model

    @classmethod
    def A2C_GridOccupancy(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "A2C_Kinematics"
        policy_kwargs = dict(
            features_extractor_class=GridOccupancyCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[256, dict(pi=[256, 64], vf=[256, 256])]
        )
        model = A2C('MlpPolicy', env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=5e-4,
                    gamma=0.8,
                    n_steps=32,
                    verbose=2,
                    tensorboard_log="./log/gridoccupancy/")
        return learner_name, model

    @classmethod
    def DQN_TTC(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "DQN_TTC"
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[512, 256, 256, 256]),
                    learning_rate=5e-4,
                    buffer_size=int(1e6),
                    learning_starts=2000,
                    batch_size=64,
                    exploration_fraction=0.1,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log="./log/TTC/")
        return learner_name, model

    @classmethod
    def PPO_TTC(cls, env) -> Tuple[str, BaseAlgorithm]:
        # assert isinstance(env, SubprocVecEnv)
        # n_envs = env.num_envs
        learner_name = "PPO_TTC"
        batch_size = 64
        model = PPO('MlpPolicy', env,
                    learning_rate=5e-4,
                    policy_kwargs=dict(net_arch=[512, 256, dict(pi=[256, 64], vf=[256, 256])]),
                    n_steps=batch_size * 12,
                    batch_size=batch_size,
                    n_epochs=10,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log="./log/TTC/")
        return learner_name, model

    @classmethod
    def A2C_TTC(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "A2C_TTC"
        model = A2C('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[512, 256, dict(pi=[256, 64], vf=[256, 256])]),
                    learning_rate=5e-4,
                    gamma=0.8,
                    n_steps=6,
                    verbose=2,
                    tensorboard_log="./log/TTC/")
        return learner_name, model
    
    @classmethod
    def EGO_Kinematics(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "EGO_Kinematics"
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(features_extractor_class=EgoAttentionNetwork_feature_extractor,
                    net_arch = [128,64,32]),
                    learning_rate=5e-4,
                    buffer_size=int(1e6),
                    learning_starts=2000,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log="./log/")
        return learner_name, model

    @classmethod
    def EGO_TTC(cls, env) -> Tuple[str, BaseAlgorithm]:
        learner_name = "EGO_TTC"
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(features_extractor_class=EgoAttentionNetwork_feature_extractor,net_arch=[16]),
                    learning_rate=5e-4,
                    buffer_size=int(1e6),
                    learning_starts=2000,
                    batch_size=64,
                    exploration_fraction=0.1,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log="./log/TTC/")
        return learner_name, model

