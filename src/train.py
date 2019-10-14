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
    :param model_names:
    :return:
    """
    episode_durations = []
    cumulative_reward = []

    actor, critic = models
    opt_actor, opt_critic = optimizer

    # loop for each episode
    for episode in range(num_episodes):
        states, critic_values, rewards, log_probs, pi_entropy, done_list = run_episode(env, actor, critic)
        cumulative_reward.append(np.sum(rewards))
        episode_durations.append(len(rewards))

        # TODO: probably here a good place distinguish different critics and returns
        # Monte Carlo: -1, otherwise n-step 1 ... N
        if n_step == -1:
            discounted_returns = get_cumulative_discounted_rewards(rewards, done_list, gamma)
            discounted_returns = (discounted_returns - discounted_returns.mean()) / discounted_returns.std()
            actor_loss, critic_loss = calculate_loss(discounted_returns, log_probs, critic_values, pi_entropy)
        else:
            n_step_returns = compute_n_step_returns(rewards, critic_values, gamma, n_step)
            n_step_returns = (n_step_returns - n_step_returns.mean()) / n_step_returns.std()
            actor_loss, critic_loss = calculate_loss(n_step_returns, log_probs, critic_values, pi_entropy)

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        opt_actor.step()
        opt_critic.step()

        if episode % 10 == 0:
            print(
                "Episode: {}:  current reward: {}, average reward: {},   current steps: {},   loss: {} ".format(episode,
                                                                                                                np.sum(
                                                                                                                    rewards),
                                                                                                                np.mean(
                                                                                                                    cumulative_reward),
                                                                                                                len(
                                                                                                                    rewards),
                                                                                                                actor_loss + critic_loss))

        save_models((actor, critic), model_names)

    return episode_durations, cumulative_reward


def run_episode(env, actor, critic):
    """

    :param env:
    :param actor:
    :param critic:
    :return:
    """
    states, critic_values, rewards, log_probs, pi_entropies, done_list = [], [], [], [], [], []
    s = env.reset()
    while True:
        a, log_prob, pi_s_a = select_action(actor, s)

        v_s = critic(s)
        s_, reward, done, _ = env.step(a)

        # For better exploration ?!
        pi_entropy = pi_s_a.entropy()

        states.append(s)
        critic_values.append(v_s)
        rewards.append(torch.FloatTensor([reward]))
        log_probs.append(log_prob)
        pi_entropies.append(pi_entropy)
        done_list.append(torch.FloatTensor([1 - done]))

        s = s_

        if done:
            v_s = critic(s)
            critic_values.append(v_s)
            break

    return states, torch.stack(critic_values), rewards, torch.stack(log_probs), torch.stack(pi_entropies), done_list


def select_action(model, state):
    """

    :param model:
    :param state:
    :return:
    """
    pi_s_a = model(state)
    a = pi_s_a.sample()
    log_prob = pi_s_a.log_prob(a).unsqueeze(0)

    return a.item(), log_prob, pi_s_a


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

    if n_step == 1:
        returns += torch.FloatTensor(np.power(gamma, np.ones(T)))[:, None] * values[n_step:]

    else:
        returns[:T - (n_step - 1)] += torch.FloatTensor(np.power(gamma, np.ones(T - (n_step - 1)) * n_step))[:,
                                      None] * values[n_step:]

        returns[T - (n_step - 1):] += torch.FloatTensor(np.power(gamma, np.ones(n_step - 1) * n_step))[:,
                                      None] * values[T]

    return returns


def calculate_loss(discounted_returns, log_prob, v_s, pi_entropy, batch=True):
    """

    :param discounted_returns:.unwrapped
    :param log_prob:
    :param v_s:
    :return:
    """

    advantage = discounted_returns - v_s[:-1]

    if batch:
        loss_actor = (-log_prob * advantage.detach() - pi_entropy * 0.01).mean()
        loss_critic = advantage.pow(2).mean()

        return loss_actor, loss_critic * 0.5

    else:
        loss_actor = -(log_prob * advantage.detach()).mean() - pi_entropy * 0.01
        loss_critic = advantage.pow(2).mean()

        return loss_actor, loss_critic * 0.5


## EXPERIMENT WITH ONE STEP AGAIN
def train_one_step(env, models, optimizer, num_episodes, gamma, n_step=1, model_names=("actor_cartpole", "v_cartpole")):
    episode_durations = []
    cumulative_reward = []

    actor, critic = models
    opt_actor, opt_critic = optimizer

    # loop for each episode
    for episode in tqdm(range(num_episodes)):

        T = np.inf
        I = 1
        step, t = 0, 0
        s = env.reset()

        states = [s]  # S_0, S_1, ...
        rewards = []  # R_1, R_2, ...

        while True:

            if t < T:
                with torch.no_grad():
                    a, log_prob, pi_s_a = select_action(actor, s)

                # take action
                s_, r, done, _ = env.step(a)

                rewards.append(r)
                states.append(s_)

                if done:
                    T = t + 1

            tau = t - n_step + 1

            if tau >= 0:

                # compute G
                G = compute_G_n_step(rewards, gamma, tau, n_step, t, T)

                if tau + n_step < T:
                    with torch.no_grad():
                        v_s_new = critic(s_)
                        G += (gamma ** n_step) * v_s_new

                # update model for state s_tau
                state_tau = states[tau]
                a, log_prob_tau, pi_s_a_tau = select_action(actor, state_tau)
                v_s_tau = critic(s)

                # For better exploration ?!
                pi_entropy = pi_s_a_tau.entropy().mean()

                actor_loss, critic_loss = calculate_loss(G, log_prob_tau, v_s_tau, pi_entropy, False)

                # backprop
                opt_actor.zero_grad()
                opt_critic.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                opt_actor.step()
                opt_critic.step()

            # update episode and records
            s = s_
            t += 1
            I = gamma * I

            if tau == T - 1:
                cumulative_reward.append(np.sum(rewards))
                break

        if episode % 10 == 0:
            # print("E{0}- Steps:{1} Loss:{2}".format(episode, t, loss))
            pass

        episode_durations.append(t)

    save_models((actor, critic), model_names)

    return episode_durations, cumulative_reward


def compute_G_n_step(rewards, gamma, tau, n_step, t, T):
    # print("Alex loss: ", np.sum(gamma ** (i - tau) * rewards[i] for i in range(tau, min(tau + n_step, T))))
    # print("me loss: ", np.sum(rewards[tau:t + 1] * np.power(gamma, range(len(rewards[tau:t + 1])))))
    # return np.sum(rewards[tau:t + 1] * np.power(gamma, range(len(rewards[tau:t + 1]))))
    return np.sum(gamma ** (i - tau) * rewards[i] for i in range(tau, min(tau + n_step, T)))
