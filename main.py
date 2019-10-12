import random
import gym

import torch

import numpy as np

from src import train, helpers


def start_training(device):
    environments = ["CartPole-v0", "MountainCar-v0"]
    # names for saving the models afterwards
    model_names = [("actor_cartpole", "v_cartpole"), ("actor_mountainCar", "v_mountainCar")]

    learning_rates = [(0.001, 0.001), (0.00002, 0.001)]

    model_types = ["TD", "Q", "Advantage"]

    # TODO: Some more elaborated loop where we organize our experiments

    # later in loop
    no_experiment = 0

    env = gym.make(environments[no_experiment])
    model_names = model_names[no_experiment]
    learn_rates = learning_rates[no_experiment]

    num_episodes = 400
    gamma = 0.99

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    n_step = 5

    # Decided to model actorlearning_rates and critic in two seperate classes
    models, optimizer = helpers.init_model(model_types[0], env, learn_rates, device, n_hidden=(128, 256))
    performance, cumulative_reward = train.train_actor_critic(env, models, optimizer, num_episodes, gamma, n_step,
                                                              model_names)

    helpers.plot_results(performance, cumulative_reward)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_training(device)

    # environment_name = "MountainCar-v0"
    # model_path = "models/actor_mountainCar.pth"

    model_path = "models/actor_cartpole.pth"
    environment_name = "CartPole-v0"
    helpers.visualize_performance(environment_name, model_path, device)
