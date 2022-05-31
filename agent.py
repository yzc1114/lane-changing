import os
import sys
import argparse
import time

import gym
from itertools import count

import learners
from env import make_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from learners import make_learner_fn
from tqdm.notebook import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None

class Agent(object):
    @staticmethod
    def parse_args():
        """ Parse arguments from command line input
        """
        parser = argparse.ArgumentParser(description='Training parameters')
        parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])  # mode = 'train' or 'test'
        parser.add_argument('--obs_type', type=int, default=1, choices=[1], help="observation type, 1: kinematics")
        parser.add_argument('--learner_type', type=str, default='DDPG',
                            help="Algorithm to train from {PPO, A2C, DQN, DDPG, TD3}")
        parser.add_argument('--parallels', type=int, default=1)
        parser.add_argument('--nb_steps', type=int, default=int(10000*50), help="Number of training steps")
        parser.add_argument('--eval_interval_steps', type=int, default=500, help="Eval and checkpoint interval steps")
        parser.add_argument('--init_weight_path', type=str, default=None, help="initial weight path.")
        parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
        parser.set_defaults(render=True)
        return parser.parse_args()

    @staticmethod
    def init_learner(args, env):
        learner_type = args.learner_type
        obs_type = args.obs_type
        learner_name, learner = make_learner_fn(learner_type, obs_type)(env)
        if args.init_weight_path is not None:
            learner.load(args.init_weight_path)
        return learner_name, learner

    def train(self):
        args = self.parse_args()
        obs_type = args.obs_type

        train_env = make_env(args.parallels, obs_type)
        learner_name, learner = self.init_learner(args, train_env)
        now = int(time.time())
        prefix = learner_name + "_" + str(now)
        # 编写算法训练过程
        weights_directory = './weights/'
        nb_steps = args.nb_steps

        checkpoint_callback = CheckpointCallback(save_freq=args.eval_interval_steps, save_path=weights_directory,
                                                 name_prefix=prefix)
        # Separate evaluation env
        eval_env = make_env(1, obs_type)
        eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(weights_directory, prefix + '_best'),
                                     log_path='./log/', eval_freq=args.eval_interval_steps,
                                     deterministic=True, render=True)

        tqdm_callback = TqdmCallback()
        callback = CallbackList([checkpoint_callback, eval_callback, tqdm_callback])
        print(f"start train, init weight: {args.init_weight_path}")
        print(f"nb_steps: {nb_steps}, learner name: {prefix}, obs_type: {obs_type}")

        learner.learn(nb_steps, callback=callback, log_interval=100)

    def test(self):
        args = self.parse_args()
        obs_type = args.obs_type
        eval_env = make_env(1, obs_type)
        learner_name, learner = self.init_learner(args, eval_env)
        # 测试10个episode的结果
        steps = []
        total_rewards = []
        no_collisions = []
        for i in range(10):
            obs = eval_env.reset()
            done = False
            rewards = []
            step = 0
            while not done:
                step += 1
                action, _states = learner.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                rewards.append(reward)
                eval_env.render()
            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            steps.append(step)


if __name__ == "__main__":
    agent = Agent()
    agent.train()
