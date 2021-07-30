import math
import matplotlib.pyplot as plt
from robot_model import WallBot
from animation import AxData, Animator


def main(animate):
    dt = .005
    goal_x = 1
    goal_theta = -math.pi / 4

    robot = WallBot(2.25, .2, 1., dt)

    bot_states = [robot.observe()]
    gripper_coords = [robot.get_gripper_coordinates()]

    robot.switch_grip()

    last_two_swings = False

    final_hold_count = 0
    i = 0
    while 1:
        i+=1
        robot.update()
        bot_states.append(robot.observe())
        gripper_coords.append(robot.get_gripper_coordinates())

        if robot.goal_state_reached:
            final_hold_count += 1
            if final_hold_count >= 10:
                break

        if last_two_swings:
            robot.final_swing_control(goal_x, goal_theta)

        elif robot.rotation_switched:
            robot.switch_grip()
            if abs(goal_x - robot.get_gripper_coordinates()[0][0]) < robot.l / 2:
                last_two_swings = True

    bot_x = list(map(lambda x: x[0], bot_states))
    bot_y = list(map(lambda x: x[1], bot_states))
    bot_theta = list(map(lambda x: x[2] * 180 / math.pi, bot_states))

    gripper_x = list(map(lambda x: list(zip(*x))[0], gripper_coords))
    gripper_y = list(map(lambda x: list(zip(*x))[1], gripper_coords))

    fig, ax = plt.subplots()
    ax.plot([goal_x, goal_x], [0, -.5], '--')
    ax.set_aspect('equal')

    if animate:
        ax_data = [
            AxData(gripper_x, gripper_y, 'position', plot_history=False),
        ]

        animator = Animator(1/2, ax_data, dt, fig, ax, show_legend=False)
        animator.set_save_path('/home/raven.ravenind.net/r103943/Desktop/a')
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
