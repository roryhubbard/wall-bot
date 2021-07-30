from collections import deque
import random
import numpy as np
import pickle
import torch


def annealed_epsilon(initial_exploration, final_exploration,
                     final_exploration_frame, frames_count):
    """
    Return linearly annealed exploration value
    """
    return initial_exploration - (initial_exploration - final_exploration) \
        * min(frames_count / final_exploration_frame, 1)


def get_greedy_action(Q, state):
    with torch.no_grad():
        return Q(state).max(1)[1].view(1)


def get_epsilon_greedy_action(Q, state, epsilon, n_actions):
    """
    Select a random action with probabily of epsilon or select the greedy
    action according to Q

    Input:
    - Q: network that maps states to actions
    - state: current state as given by the environment
    - epsilon: probability of exploring
    - n_actions: action space cardinality

    Output:
    - action
    """
    return torch.tensor([random.choice(tuple(range(n_actions)))],
                        dtype=torch.int64) \
        if random.uniform(0.0, 1.0) < epsilon \
        else get_greedy_action(Q, state)


def save_stuff(Q, episode_rewards):
    torch.save(Q, 'trained_Q.pth')
    with open("episode_rewards.txt", "wb") as f:
        pickle.dump(episode_rewards, f)
