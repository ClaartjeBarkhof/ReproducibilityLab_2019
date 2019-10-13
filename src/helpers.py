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
    """  #

    # TODO: maybe misleading for some plots...
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# TODO: implement a def plot_results function ...
def plot_results(episode_durations, reward_across_episodes):
    """

    :param episode_durations:
    :param reward_across_episodes:
    :return:
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    axes[0].plot(smooth(episode_durations, 20))
    axes[1].plot(reward_across_episodes)
    fig.tight_layout()

    # plt.title('Episode durations per episode')
    # plt.legend(['Policy gradient'])
    plt.show()


def init_model(model_type, env, learn_rates, device, n_hidden=(128, 256)):
    """

    :param model_type:
    :param env:
    :param learn_rate:
    :param device:
    :param n_hidden:
    :return:
    """
    lr_actor, lr_critic = learn_rates
    print("Observation and action space: ")
    print(env.observation_space)
    print(env.action_space)
    n_state_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if model_type == "TD":
        actor = Actor(n_state_features, n_actions, n_hidden, device).to(device)
        critic = ValueFunction(n_state_features, n_hidden, device).to(device)

        opt_actor = optim.Adam(actor.parameters(), lr_actor)
        opt_critic = optim.Adam(critic.parameters(), lr_critic)

        return (actor, critic), (opt_actor, opt_critic)

    else:
        print("No matching procedure")


def save_models(models, model_names):
    """

    :param models:
    :param model_names:
    :return:
    """
    path = "models/{}.pth"
    actor, critic = models
    torch.save(actor, path.format(model_names[0]))
    torch.save(critic, path.format(model_names[1]))


def visualize_performance(environment_name, model_path, device):
    env = gym.make(environment_name)

    actor = torch.load(model_path)

    for i in np.arange(30):
        s = env.reset()
        while True:
            env.render()
            pi_s_a = actor(s)
            a = pi_s_a.sample().item()
            s_, reward, done, _ = env.step(a)

            s = s_

            if done:
                break
