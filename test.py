import random

import gym
import highway_env
import numpy as np
from env import make_env

def do_test_env():
    e = make_env(1, 1)
    for i in range(10):
        e.reset()
        done = False
        steps = 0
        random.seed(0)
        total_reward = 0
        while not done:
            action = np.random.randint(5)
            obs, reward, done, _ = e.step(4)
            print(f"reward: {reward}")
            total_reward += reward
            # print(obs.shape)
            e.render()
            steps += 1
            print(steps)
        print(total_reward)


if __name__ == '__main__':
    do_test_env()
