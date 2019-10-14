import random
import gym

import torch

import numpy as np

from src import train, helpers


def start_training(device):
    environments = ["CartPole-v0", "MountainCar-v0", "LunarLander-v2", "Taxi-v2", "Blackjack-v0"]
    # names for saving the models afterwards
    model_names = [("actor_cartpole", "v_cartpole"), ("actor_mountainCar", "v_mountainCar"),
                   ("actor_lunarlander", "v_lunarlander"), ("actor_taxi", "v_taxi")]

    learning_rates = [(7e-4, 7e-4), (7e-4, 7e-4), (7e-4, 7e-4), (7e-4, 7e-4), (7e-4, 7e-4)]

    model_types = ["TD", "Q", "Advantage"]

    # TODO: Some more elaborated loop where we organize our experiments

    # later in loop
    no_experiment = 0

    env = gym.make(environments[no_experiment]).unwrapped
    model_names = model_names[no_experiment]
    learn_rates = learning_rates[no_experiment]

    num_episodes = 600
    gamma = 0.99

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    n_step = 3

    models, optimizer = helpers.init_model(model_types[0], env, learn_rates, device, n_hidden=(64, 64))
    performance, cumulative_reward = train.train_actor_critic(env, models, optimizer, num_episodes, gamma, n_step,
                                                              model_names)

    # performance, cumulative_reward = train.train_one_step(env, models, optimizer, num_episodes, gamma, n_step, model_names)

    helpers.plot_results(performance, cumulative_reward)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_training(device)

    # model_path = "models/actor_mountainCar.pth"
    model_path = "models/actor_cartpole.pth"
    # model_path = "models/actor_lunarlander.pth"
    # model_path = "models/actor_taxi.pth"
    # environment_name = "MountainCar-v0"
    environment_name = "CartPole-v0"
    # environment_name = "LunarLander-v2"
    # environment_name = "Taxi-v2"
    helpers.visualize_performance(environment_name, model_path, device)
