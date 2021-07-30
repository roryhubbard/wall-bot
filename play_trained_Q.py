import gym
import numpy as np
import torch
from utils import get_greedy_action
import math
import matplotlib.pyplot as plt
from robot_model import WallBot
from environment import Environment
from animation import AxData, Animator


def main(animate):
    Q = torch.load('trained_Q.pth')
    Q.eval()

    m = 2

    dt = .01
    goal_x = .4
    goal_theta = math.pi / 4

    robot = WallBot(2.25, .2, 1., dt)

    env = Environment(robot, goal_x, goal_theta)

    bot_states = [env.robot.observe()]
    gripper_coords = [env.robot.get_gripper_coordinates()]

    done = False
    state = torch.tensor(env.reset() * m)
    while not done:
        action = get_greedy_action(
            Q, state.unsqueeze(0))
        new_state, reward, done = env.step(action.item())
        state = torch.tensor(state[1:].tolist() + new_state)

        bot_states.append(env.robot.observe())
        gripper_coords.append(env.robot.get_gripper_coordinates())

    bot_x = list(map(lambda x: x[0], bot_states))
    bot_y = list(map(lambda x: x[1], bot_states))
    bot_theta = list(map(lambda x: x[2] * 180 / math.pi, bot_states))

    gripper_x = list(map(lambda x: list(zip(*x))[0], gripper_coords))
    gripper_y = list(map(lambda x: list(zip(*x))[1], gripper_coords))

    fig, ax = plt.subplots()
    # ax.set_aspect('equal')

    if animate:
        ax_data = [
            AxData(gripper_x, gripper_y, 'position', plot_history=False),
        ]

        animator = Animator(1/2, ax_data, dt, fig, ax)
        animator.run()

    else:
        ax.plot(bot_x, bot_y, label='position')
        # ax.plot(bot_theta, label='theta')

        ax.set_title('Position')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')

        plt.show()
        plt.close()


if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = [16, 10]
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['figure.edgecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['axes.titlecolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams["figure.autolayout"] = True
    # plt.rcParams['legend.facecolor'] = 'white'

    animate = True

    main(animate)
