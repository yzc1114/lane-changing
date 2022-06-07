import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import FormatStrFormatter
import matplotlib.ticker as plt_ticker
import numpy as np
from env import ObsType, obs_type_2_string
import pandas as pd
from typing import List


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.family'] = 'Arial'

learners = ["PPO", "DQN", "A2C", "Attention"]
obs_types: List[ObsType] = [ObsType.Kinematics, ObsType.TimeToCollision, ObsType.OccupancyGrid]

# learner_2_color = {
#     "PPO": "blue",
#     "DQN": "red",
#     "A2C": "green",
#     "Attention": "purple"
# }

# obs_type_2_color = {
#     ObsType.Kinematics: "blue",
#     ObsType.TimeToCollision: "red",
#     ObsType.OccupancyGrid: "green"
# }

def inside_ticks(ax, x=True, y=True):
    if y:
        ax.tick_params(axis="y", which='major', direction="in", length=4)
        ax.tick_params(axis="y", which='minor', direction="in", length=2)
    if x:
        ax.tick_params(axis="x", which='major', direction="in", length=4)
        ax.tick_params(axis="x", which='minor', direction="in", length=2)


def save_fig(fig, filepath, mode="pdf"):
    fig.savefig(filepath, dpi=400, format=mode, bbox_inches='tight')

def plot_compare_learner(ax, learner2values, y_label):
    x_data = np.arange(100) # training step is 100_000
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    inside_ticks(ax)
    y_major_loc = plt_ticker.MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_loc)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
    x_major_loc = plt_ticker.MultipleLocator(base=25)
    ax.xaxis.set_major_locator(x_major_loc)
    for learner in learners:
        if learner not in learner2values:
            continue
        y_data = learner2values[learner]
        ax.plot(x_data, y_data,
                marker=None,
                linestyle='solid',
                linewidth='2',
                # color=learner_2_color[learner],
                label=learner)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Training Step (1k)')
    ax.legend(bbox_to_anchor=(1, 0), loc='lower right')


def plot_compare_obs(ax, obs2values, y_label):
    x_data = np.arange(100)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    inside_ticks(ax)
    y_major_loc = plt_ticker.MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_loc)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
    x_major_loc = plt_ticker.MultipleLocator(base=25)
    ax.xaxis.set_major_locator(x_major_loc)
    for obs_type in obs_types:
        obs_type_str = obs_type_2_string(obs_type)
        if obs_type not in obs2values:
            continue
        y_data = obs2values[obs_type]
        ax.plot(x_data, y_data,
                marker=None,
                linestyle='solid',
                linewidth='2',
                # color=obs_type_2_color[obs_type],
                label=obs_type_str)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Training Step (1k)')
    ax.legend(bbox_to_anchor=(1, 0), loc='lower right')


def load_learner_values_data(obs_type: ObsType):
    obs_type = obs_type_2_string(obs_type)
    reward_template = f"%s_{obs_type}_reward.csv"
    ep_length_template = f"%s_{obs_type}_ep_length.csv"

    def path(filename):
        return os.path.join("data", filename)

    def load_pd_to_y_data(filepath):
        df = pd.read_csv(filepath)
        return df["Value"].values.tolist()

    learner_2_reward_values = dict()
    learner_2_ep_length_values = dict()

    for learner in learners:
        reward_filepath = path(reward_template % learner)
        ep_length_filepath = path(ep_length_template % learner)
        if not os.path.exists(ep_length_filepath) or not os.path.exists(reward_filepath):
            continue
        learner_2_reward_values[learner] = load_pd_to_y_data(reward_filepath)
        learner_2_ep_length_values[learner] = load_pd_to_y_data(ep_length_filepath)
    return learner_2_reward_values, learner_2_ep_length_values


def plot():
    def do_one(name, plot_func):
        fig, ax = plt.subplots()
        plot_func(ax)
        # save_fig(fig, os.path.join("results", f"{name}.pdf"), mode="pdf")
        save_fig(fig, os.path.join("results", f"{name}.png"), mode="png")

    obs_type_2_learner_data = dict()
    for obs_type in obs_types:
        learner_2_rewards, learner_2_ep_lengths = load_learner_values_data(obs_type)
        obs_type_2_learner_data[obs_type] = (learner_2_rewards, learner_2_ep_lengths)
    for obs_type in obs_types:
        learner_2_rewards, learner_2_ep_lengths = obs_type_2_learner_data[obs_type]
        do_one(f"compare_learner_{obs_type_2_string(obs_type)}_reward",
               lambda ax: plot_compare_learner(ax, learner_2_rewards, "Mean Reward"))
        do_one(f"compare_learner_{obs_type_2_string(obs_type)}_ep_length",
               lambda ax: plot_compare_learner(ax, learner_2_ep_lengths, "Mean Episode Length"))

    for learner in learners:
        obs_type_2_rewards = dict()
        obs_type_2_ep_lengths = dict()
        for obs_type in obs_types:
            learner_2_rewards, learner_2_ep_lengths = obs_type_2_learner_data[obs_type]
            if learner not in learner_2_rewards:
                continue
            obs_type_2_rewards[obs_type] = learner_2_rewards[learner]
            obs_type_2_ep_lengths[obs_type] = learner_2_ep_lengths[learner]

        do_one(f"compare_obs_{learner}_reward", lambda ax: plot_compare_obs(ax, obs_type_2_rewards, learner + " Mean Reward"))
        do_one(f"compare_obs_{learner}_ep_length", lambda ax: plot_compare_obs(ax, obs_type_2_ep_lengths, learner + " Mean Episode Length"))


if __name__ == '__main__':
    plot()
