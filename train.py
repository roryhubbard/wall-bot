import math
from copy import deepcopy
from collections import deque
from robot_model import WallBot
from environment import Environment
import torch
import torch.nn as nn
import torch.optim as optim
from q_network import QNetwork
from utils import annealed_epsilon, get_epsilon_greedy_action, save_stuff
from network_update import sgd_update


def deep_qlearning(env, nepisodes, discount_factor, N, C, mini_batch_size,
                   replay_start_size, sgd_update_frequency, initial_exploration,
                   final_exploration, final_exploration_episode, lr, alpha, m):
    n_actions = 3
    Q = QNetwork(n_actions)
    Q_target = deepcopy(Q)
    Q_target.eval()

    optimizer = optim.RMSprop(Q.parameters(), lr=lr, alpha=alpha)
    criterion = nn.MSELoss()

    D = deque(maxlen=N)  # replay memory

    last_Q_target_update = 0
    last_sgd_update = 0
    episode_rewards = []

    for i in range(nepisodes):
        state = torch.tensor(env.reset() * m)

        episode_reward = 0
        done = False

        while not done:
            epsilon = annealed_epsilon(
                initial_exploration, final_exploration,
                final_exploration_episode, i)

            action = get_epsilon_greedy_action(
                Q, state.unsqueeze(0), epsilon, n_actions)

            new_state, reward, done = env.step(action.item())
            reward = torch.tensor([reward])

            episode_reward += reward.item()
            if done:
                next_state = None
                episode_rewards.append(episode_reward)
            else:
                next_state = torch.tensor(state[1:].tolist() + new_state)

            D.append((state, action, reward, next_state))

            state = next_state

            if len(D) < replay_start_size:
                continue

            last_sgd_update += 1
            if last_sgd_update < sgd_update_frequency:
                continue
            last_sgd_update = 0

            sgd_update(Q, Q_target, D, mini_batch_size,
                       discount_factor, optimizer, criterion)

            last_Q_target_update += 1

            if last_Q_target_update % C == 0:
                Q_target = deepcopy(Q)
                Q_target.eval()


        if i % 1000 == 0:
            save_stuff(Q, episode_rewards)
            print(f'episodes completed = {i}')

    return Q, episode_rewards

def main():
    dt = .01
    goal_x = .4
    goal_theta = math.pi / 4

    robot = WallBot(2.25, .2, 1., dt)

    env = Environment(robot, goal_x, goal_theta)

    nepisodes = 500000  # train for a total of 50 million frames
    discount_factor = 0.99
    N = 100000  # replay memory size (paper uses 1000000)
    C = 1000  # number of steps before updating Q target network
    mini_batch_size = 32
    replay_start_size = 10000  # minimum size of replay memory before learning starts
    sgd_update_frequency = 4  # number of action selections in between consecutive SGD updates
    initial_exploration = 1.  # initial epsilon valu3
    final_exploration = 0.1  # final epsilon value
    final_exploration_episode = 100000  # number of frames over which the epsilon is annealed to its final value
    lr = 0.01  # learning rate used by RMSprop
    alpha = 0.99  # alpha value used by RMSprop
    m = 2  # number of consecutive frames to stack for input to Q network

    Q, episode_rewards = deep_qlearning(
        env, nepisodes, discount_factor, N, C, mini_batch_size, replay_start_size,
        sgd_update_frequency, initial_exploration, final_exploration,
        final_exploration_episode, lr, alpha, m)

    save_stuff(Q, episode_rewards)


if __name__ == "__main__":
    main()
