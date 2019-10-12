import torch
from torch import optim

import numpy as np

import matplotlib.pyplot as plt
import gym

# from .actor_critic import ValueFunction, QValueFunction, Actor
from .actor_critic import ValueFunction, QValueFunction, Actor


def smooth(x, N):
    """

    :param x:
    :param N:
    :return:
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# TODO: implement a def plot_results function ...
def plot_results(results):
    """

    :param results:
    :return:
    """
    # plt.plot(smooth(results, 20))
    # plt.title('Episode durations per episode')
    # plt.legend(['Policy gradient'])
    plt.plot(results)
    plt.show()


def init_model(model_type, env, learn_rate, device, n_hidden=(128, 256)):
    """

    :param model_type:
    :param env:
    :param learn_rate:
    :param device:
    :param n_hidden:
    :return:
    """
    if model_type == "TD":

        n_state_features = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # actor = Actor(n_state_features, n_actions, n_hidden)
        actor = Actor(n_state_features, n_actions, n_hidden, device).to(device)
        # critic = Critic(n_state_features, n_hidden)
        critic = ValueFunction(n_state_features, n_hidden, device).to(device)

        opt_actor = optim.Adam(actor.parameters(), learn_rate)
        opt_critic = optim.Adam(critic.parameters(), learn_rate)

        return (actor, critic), (opt_actor, opt_critic)

    else:
        print("No matching procedure")
