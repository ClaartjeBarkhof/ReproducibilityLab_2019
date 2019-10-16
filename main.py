import random
import gym

import torch

import numpy as np

from src import train, helpers


def start_training(environment, model_type, n_step):
    env = gym.make(environment)

    # SETTINGS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learn_rates = (7e-4, 7e-4)
    num_episodes = 100
    gamma = 0.99
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    model_names = ("actor_"+environment, 'critic_'+environment)

    # INIT
    models, optimizer = helpers.init_model(model_type, env, learn_rates, device, n_hidden=(64, 64))
    # TRAIN
    episode_durations, cumulative_reward, actor_losses, running_average = train.train_actor_critic(env, models, optimizer, num_episodes, gamma,
                                                                            n_step, model_names,
                                                                            model_type)
    # PLOT
    helpers.plot_results(episode_durations, running_average, cumulative_reward, actor_losses, n_step, environment,
                         model_type)


def run_experiments():
    # names for saving the models afterwards
    # model_names = [("actor_cartpole", "v_cartpole"), ("actor_mountainCar", "v_mountainCar"),
               # ("actor_lunarlander", "v_lunarlander"), ("actor_taxi", "v_taxi")]

    environments = ["CartPole-v0", "MountainCar-v0", "LunarLander-v2", "Taxi-v2"]
    for environment in environments:
        model_types = ["Advantage", "Q", "Reinforce"]
        for model_type in model_types:
            # n_steps = [1, 2, 4, 8, 16]
            # for n_step in n_steps:
            n_step = 2
            if model_type == "Reinforce":
                n_step = "Monte Carlo"
            print('**********************************************************************')
            print("Environment:", environment, 'Model type:', model_type, 'N_step:', n_step)
            print('**********************************************************************')
            start_training(environment, model_type, n_step)
            if model_type == "Reinforce":
                print("Reinforce is automatically all N steps, so no n-step variations needed.")
                break
        
if __name__ == "__main__":

    
    run_experiments()

    # start_training(device)
    # model_path = "models/actor_mountainCar.pth"
    # model_path = "models/actor_cartpole.pth"
    # model_path = "models/actor_lunarlander.pth"
    # model_path = "models/actor_taxi.pth"
    # environment_name = "MountainCar-v0"
    # environment_name = "CartPole-v0"
    # environment_name = "LunarLander-v2"
    # environment_name = "Taxi-v2"
    # helpers.visualize_performance(environment_name, model_path, device)
