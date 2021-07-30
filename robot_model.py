import math
from enum import Enum
import numpy as np


G = 9.81


class GripState(Enum):
    BOTH = 0
    LEFT = 1
    RIGHT = 2
    NEITHER = 3


class WallBot:

    def __init__(self, mass, length, friction, dt):
        self.dt = dt
        self.m = mass
        self.l = length
        self.fr = friction
        self.reset()

    def reset(self):
        self.theta = math.pi / 2
        self.theta_dot = 0.
        self.x = self.l / 2 * math.sin(self.theta)
        self.x_dot = 0.
        self.y = -self.l / 2 * math.cos(self.theta)
        self.y_dot = 0.
        self.grip_state = GripState.BOTH
        self.rotation_switched = False
        self.last_swing = False
        self.goal_state_reached = False

    def switch_grip(self):
        self.grip_state = GripState.LEFT \
            if self.grip_state is GripState.RIGHT \
            else GripState.RIGHT
        self.theta += math.pi
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        self.theta_dot = 0.
        self.x_dot = 0.
        self.y_dot = 0.

    def hold(self):
        self.grip_state = GripState.BOTH
        self.theta_dot = 0.
        self.x_dot = 0.
        self.y_dot = 0.

    def release(self):
        self.grip_state = GripState.NEITHER

    def final_swing_control(self, goal_x, goal_theta):
        if self.last_swing:
            theta_error = abs(goal_theta - math.atan(math.tan(self.theta)))
            if theta_error < 5 * math.pi / 180:
                self.hold()
                self.goal_state_reached = True
            return

        goal_grip = goal_x - self.l / 2 * math.sin(goal_theta)
        if self.get_gripper_coordinates()[1][0] >= goal_grip:
            self.switch_grip()
            self.last_swing = True

    def update(self):
        if self.grip_state is GripState.BOTH:
            return

        elif self.grip_state is GripState.NEITHER:
            self.x += self.x_dot * self.dt
            self.y_dot += G * self.dt
            self.y += self.y_dot * self.dt

        else:
            prev_rotation_dir = np.sign(self.theta_dot)

            self.theta_dot += (-G / (self.l / 2) * math.sin(self.theta)
                - self.fr / self.m * self.theta_dot) * self.dt

            theta_updated = self.theta + self.theta_dot * self.dt
            self.x += self.l / 2 * (math.sin(theta_updated) -
                                    math.sin(self.theta))
            self.y -= self.l / 2 * (math.cos(theta_updated) -
                                    math.cos(self.theta))
            self.theta = math.atan2(math.sin(theta_updated),
                                    math.cos(theta_updated))

            if prev_rotation_dir != 0 \
                    and prev_rotation_dir != np.sign(self.theta_dot):
                self.rotation_switched = True
            else:
                self.rotation_switched = False

    def observe(self):
        return (self.x, self.y, self.theta)

    def get_gripper_coordinates(self):
        return [
            (self.x - self.l / 2 * math.sin(self.theta),
             self.y + self.l / 2 * math.cos(self.theta)),
            (self.x + self.l / 2 * math.sin(self.theta),
             self.y - self.l / 2 * math.cos(self.theta)),
        ]
