import gym
import highway_env
import numpy as np
from env import make_env

def do_test_env():
    e = make_env(1, 1)
    e.reset()
    done = False
    steps = 0
    while not done:
        action = np.random.randint(5)
        obs, reward, done, _ = e.step(action)
        # print(obs.shape)
        # e.render()
        steps += 1
        print(steps)


if __name__ == '__main__':
    do_test_env()
