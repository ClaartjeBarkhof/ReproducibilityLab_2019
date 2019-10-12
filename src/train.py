import torch
import numpy as np
from tqdm import tqdm

from .helpers import save_models


def train_actor_critic(env, models, optimizer, num_episodes, gamma, n_step=1,
                       model_names=("actor_cartpole", "v_cartpole")):
    """

    :param env:
    :param models:
    :param optimizer:
    :param num_episodes:
    :param gamma:
    :param n_step:
    :return:
    """
    episode_durations = []
    cumulative_reward = []

    actor, critic = models
    opt_actor, opt_critic = optimizer

    # loop for each episode
    for episode in tqdm(range(num_episodes)):
        states, actor_values, rewards, log_probs, done_list = run_episode(env, actor, critic)
        cumulative_reward.append(np.sum(rewards))
        episode_durations.append(len(rewards))

        if n_step == 1:
            discounted_returns = get_cumulative_discounted_rewards(rewards, done_list, gamma)
            actor_loss, critic_loss = calculate_loss(discounted_returns, log_probs, actor_values)
        else:
            n_step_returns = compute_n_step_returns(rewards, actor_values, gamma, n_step)
            actor_loss, critic_loss = calculate_loss(n_step_returns, log_probs, actor_values)

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        opt_actor.step()
        opt_critic.step()

    save_models((actor, critic), model_names)

    return episode_durations, cumulative_reward


def run_episode(env, actor, critic):
    """

    :param env:
    :param actor:
    :param critic:
    :return:
    """
    states, actor_values, rewards, log_probs, done_list = [], [], [], [], []
    s = env.reset()
    while True:
        a, log_prob = select_action(actor, s)
        v_s = critic(s)
        s_, reward, done, _ = env.step(a)

        states.append(s_)
        actor_values.append(v_s)
        rewards.append(torch.FloatTensor([reward]))
        log_probs.append(log_prob)
        done_list.append(torch.FloatTensor([1 - done]))

        s = s_

        if done:
            break

    return states, torch.stack(actor_values), rewards, torch.stack(log_probs), done_list


def select_action(model, state):
    """

    :param model:
    :param state:
    :return:
    """
    pi_s_a = model(state)
    a = pi_s_a.sample()
    log_prob = pi_s_a.log_prob(a).unsqueeze(0)

    return a.item(), log_prob


def get_cumulative_discounted_rewards(rewards, done_list, gamma):
    """
    This calculates basically the full reinforce return

    :param rewards:
    :param done_list:
    :param gamma:
    :return:
    """
    cumulative_reward, _r = [], 0
    for i in np.arange(len(rewards))[::-1]:
        _r = done_list[i] * _r * gamma + rewards[i]
        cumulative_reward.insert(0, _r)
    return torch.stack(cumulative_reward)


def compute_n_step_returns(rewards, values, gamma, n_step):
    """
    Should return n_step target ((γ^0) Rt+1 +  (γ^1)Rt+2 ...  (γ^t+n-1)Rt+n + (γ^t+n)V(St+n)) - V(S_2)
    :param rewards:
    :param values:
    :param gamma:
    :param n_step:
    :return:
    """

    T = len(rewards)
    returns = []
    for i in np.arange(len(rewards)):
        returns.append(torch.sum(
            torch.stack(rewards[i: np.min((i + n_step, T))]) * torch.FloatTensor(
                np.power(gamma, np.arange(len(rewards[i: np.min((i + n_step, T))]))))[:, None]))

    returns = torch.stack(returns)[:, None]

    returns[:T - (n_step - 1)] += torch.FloatTensor(np.power(gamma, np.ones(T - (n_step - 1)) * n_step))[:,
                                  None] * values[n_step - 1:]

    return returns


def calculate_loss(discounted_returns, log_prob, v_s):
    """

    :param discounted_returns:
    :param log_prob:
    :param v_s:
    :return:
    """
    advantage = discounted_returns - v_s

    loss_actor = -(log_prob * advantage.detach()).mean()
    loss_critic = advantage.pow(2).mean()

    return loss_actor, loss_critic
