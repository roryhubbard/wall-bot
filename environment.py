import math


class Environment:

    def __init__(self, robot, goal_x, goal_theta):
        self.robot = robot
        self.goal_x = goal_x
        self.goal_theta = goal_theta
        self.step_counter = 0

    def step(self, u):
        done = False

        if u == 1:
            self.robot.switch_grip()

        elif u == 2:
            self.robot.hold()

        self.robot.update()

        x_error = self.goal_x - self.robot.observe()[0]
        # theta_error = self.goal_theta - math.atan(math.tan(self.robot.observe()[2]))
        # state = [x_error, theta_error]
        state = [x_error]

        # cost = 100 * x_error**2 + theta_error**2 / 10
        reward = -x_error**2

        self.step_counter += 1
        if self.step_counter >= 100:
            done = True

        return state, reward, done

    def reset(self):
        self.robot.reset()
        x_error = self.goal_x - self.robot.observe()[0]
        # theta_error = self.goal_theta - self.robot.observe()[2]
        # state = [x_error, theta_error]
        state = [x_error]
        self.step_counter = 0
        return state
