# Imports
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import gym
import pandas as pd

# The class that contains the actor/critic networks
from actor_critic import ActorCritic

# GLOBALS
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Adapted from https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
def train():
    env = gym.make(args.environment)

    n_state_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    actor_critic = ActorCritic(n_state_features, n_actions, args.n_hidden)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(args.n_episodes):
        log_probs = []
        values = []
        rewards = []

        # NEW EPISDE, SO RESET ENVIRONMENT
        state = env.reset()
        for steps in range(args.n_steps):
            # FORWARD
            state_value, policy_tensor = actor_critic.forward(state)
            
            # DETACH
            state_value = state_value.detach().numpy()[0,0]
            policy = policy_tensor.detach().numpy() 

            # SELECT ACTION ACCORDING TO POLICY
            action = np.random.choice(n_actions, p=np.squeeze(policy)) # how does this break ties?
            log_prob = torch.log(policy_tensor.squeeze(0)[action])
            entropy = -np.sum(np.mean(policy) * np.log(policy))
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(state_value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            
            # IF DONE OR MAX STEPS REACHED
            if done or steps == args.n_steps-1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % args.print_every == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                break
        
        # COMPUTE Q VALUES
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + args.gamma * Qval
            Qvals[t] = Qval
  
        # UPDATE ACTOR & CRITIC NETWORKS
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        # ADVANTAGE TERM
        if args.actor_critic_type == 'Advantage':
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
        # TODO: Q-actor critic (pseudocode in google docs)
        elif args.actor_critic_type == 'Q':
            raise NotImplementedError

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    # TODO: SAVE MODEL
    # torch.save()

    return all_rewards, all_lengths, average_lengths

def plot(all_rewards, all_lengths, average_lengths):
    # PLOT THE RESULTS
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    
    fig, axs = plt.subplots(1, 2, figsize=(20,7))

    title = "".join([str(key)+':'+str(value)+' | ' for key, value in vars(args).items()])
    fig.suptitle(title, fontsize=10)

    axs[0].plot(all_rewards, label='all_rewards')
    axs[0].plot(smoothed_rewards, label='smoothed_rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')

    axs[1].plot(all_lengths, label='all_lengths')
    axs[1].plot(average_lengths, label='average_lengths')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Episode length')
    
    # TODO: SAVE PLOT
    # plt.savefig()
    plt.show()

def print_flags(args):
  # Check whether actor_critic_type is set correctly
  if args.actor_critic_type not in ['Q', 'Advantage']:
    print("Specify valid actor_critic_type: Q or Advantage")
    quit()
  
  # Check whether environment is set correctly:
  if args.environment not in ['MountainCar-v0', 'CartPole-v0']:
    print("Specify valid actor_critic_type: MountainCar-v0 or CartPole-v0")
    quit()

  print('-----------FLAGS-----------')
  for key, value in vars(args).items():
    print(key + ' : ' + str(value))
  print('-----------FLAGS-----------')

def main():
    print_flags(args)

    # MAKE OUTPUT DIRECTORY
    os.makedirs(args.output_dir, exist_ok=True)

    # TRAIN
    all_rewards, all_lengths, average_lengths = train()
    
    # PLOT
    plot(all_rewards, all_lengths, average_lengths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Options: Q & Advantage
    parser.add_argument('--actor_critic_type', required=True, type=str, 
                        default='Q', help='Specify valid actor_critic_type: Q or Advantage')
    parser.add_argument('--environment', required=True, type=str, 
                        default='CartPole-v0', help='Specify valid actor_critic_type: MountainCar-v0 or CartPole-v0')
    parser.add_argument('--n_episodes', type=int, default=200,
                        help='number of episodes')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='learning rate')
    parser.add_argument('--n_hidden', type=int, default=256,
                        help='number of episodes')
    parser.add_argument('--n_steps', type=int, default=300,
                        help='number of steps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--print_every', type=int, default=10,
                        help='print the progress every N episodes')
    parser.add_argument('--output_dir', type=str, default='Results',
                        help='directory to save results')
    args = parser.parse_args()

    main()