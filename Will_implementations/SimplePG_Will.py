# @author William Berdanier
# This is an implementation of Simple Policy Gradient closely following the one in the introduction section.

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

def mlp(sizes, activation = nn.Tanh, output_activation = nn.Identity):
    # Build the feed-forward neural net
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes):
            act = activation
        else
            act = output_activation
        layers.append(nn.Linear(sizes[j], sizes[j+1]))
        layers.append(act())
    return nn.Sequential(*layers)

def train(env_name = "CartPole-v0", hidden_sizes = [32], lr = 1e-2,
          epochs = 50, batch_size = 5000, render = False):
    # Make the environment, check spaces, get observation / action dimensions.
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "Environment must be a continuous state space."
    assert isinstance(env.action_space, Discrete), \
        "Action space must be discrete."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Make the core of the policy network.
    logits_net = mlp(sizes = [obs_dim] + hidden_sizes + [n_acts])

    # Compute action distribution.
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits = logits)

    # Make the action selection function.
    # Outputs int actions, sampled from policy.
    def get_action(obs):
        return get_policy(obs).sample().item()

    # Make loss function whose gradient, for the right data, is policy gradient.
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # Make the optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        # Lists for logging.
        batch_obs = []     # observations
        batch_acts = []    # actions
        batch_weights = [] # R(tau) weighting in policy gradient
        batch_rets = []    # episode returns
        batch_lens = []    # episode lengths

        # Reset episode-specific variables.
        obs = env.reset()  # first obs comes from starting dist
        done = False       # signal from environment that ep is over
        ep_rews = []       # list for rewards accrued

        # Render first episode of each epoch.
        finished_rendering_this_epoch = False

        # Collect experience by acting in the env with policy.
        while True:
            # Rendering.
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # Save observations.
            batch_obs.append(obs.copy())

            # Take action
            act = get_action(torch.as_tensor(obs, dtype = torch.float32))
            obs, rew, done, _ = env.step(act)

            # Save action, reward.
            batch_acts.append(act)
            ep_rews.append(rew)
            if done:
                # If over, record data from episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # Weight for each logP(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # Reset ep-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # Won't render again this epoch
                finished_rendering_this_epoch = True

                # End the exp loop if we have enough
                if len(batch_obs) > batch_size:
                    break

        # Take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs = torch.as_tensor(batch_obs, dtype = torch.float32),
                                  act = torch.as_tensor(batch_acts, dtype = torch.int32),
                                  weights = torch.as_tensor(batch_weights, dtype = torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # Training loop.
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)