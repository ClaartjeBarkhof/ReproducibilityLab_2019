import random
import gym

import torch

import numpy as np

from src import train, helpers


def start():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environments = ["CartPole-v0", "MountainCar-v0"]
    model_types = ["TD", "Q", "Advantage"]

    # TODO: Some more elaborated loop where we organize our experiments

    env = gym.make(environments[0])
    num_episodes = 400
    gamma = 0.99
    learn_rate = 0.001

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    n_step = 3

    models, optimizer = helpers.init_model(model_types[0], env, learn_rate, device, n_hidden=(128, 256))
    performance, cumulative_reward = train.train_actor_critic(env, models, optimizer, num_episodes, gamma, n_step)

    helpers.plot_results(performance)


if __name__ == "__main__":
    start()
