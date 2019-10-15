import random
import gym

import torch

import numpy as np

from src import train, helpers


def start_training(device):
    # TODO: We want to loop over model_type_no (reinforce, Advantage, Q) and no_experiment (environments, model names..) and different n-steps
    # However, reinforce is always Monte carlo
    model_type_no = 1
    no_experiment = 0

    n_steps = [1, 3, 5]

    environments = ["CartPole-v0", "MountainCar-v0", "LunarLander-v2", "Taxi-v2"]
    # names for saving the models afterwards
    model_names = [("actor_cartpole", "v_cartpole"), ("actor_mountainCar", "v_mountainCar"),
                   ("actor_lunarlander", "v_lunarlander"), ("actor_taxi", "v_taxi")]

    learning_rates = [(7e-4, 7e-4), (7e-4, 7e-4), (7e-4, 7e-4), (7e-4, 7e-4), (7e-4, 7e-4)]

    model_types = ["Advantage", "Q", "Reinforce"]

    env = gym.make(environments[no_experiment])
    model_names = model_names[no_experiment]
    learn_rates = learning_rates[no_experiment]

    model_to_use = model_types[model_type_no]

    print("Model we are using: ", model_to_use)

    if model_to_use in ["Advantage", "Q"]:
        n_step = n_steps[2]  # here 0-4

    else:
        n_step = "Monte Carlo"

    num_episodes = 1000
    gamma = 0.99

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    models, optimizer = helpers.init_model(model_to_use, env, learn_rates, device, n_hidden=(64, 64))
    episode_durations, cumulative_reward, actor_losses, running_average = train.train_actor_critic(env, models, optimizer, num_episodes, gamma,
                                                                            n_step, model_names,
                                                                            model_to_use)

    helpers.plot_results(episode_durations, running_average, cumulative_reward, actor_losses, n_step, environments[no_experiment],
                         model_to_use)


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
