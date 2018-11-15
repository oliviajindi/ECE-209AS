# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:51:15 2018

@author: oliviajin
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time
import warnings
warnings.filterwarnings("ignore")

class Node:
    """
    Define Node type for RRT.
    x,y: Node coordinate.
    Parent: Previous node index in node list.
    """
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.parent = None

class Robot:
    def __init__(self, target_pos, obstacleList, environment_length=3000, environment_width=3000, wheel_radius=20,
                 wheel_distance=85, robot_diameter=115, delta_t=0.2, init_state=None, rpm=130, max_epoch=500):

        # ------Initialize the environment and the robot.------
        self.le = environment_length
        self.w = environment_width
        self.r = wheel_radius
        self.d = wheel_distance
        self.dt = delta_t
        self.Max_square = environment_length ** 2 + environment_width ** 2
        self.target = target_pos
        self.obstacleList = obstacleList
        self.diameter = robot_diameter

        # Setting the max motor speed as 130 RPM at 6V
        self.rpm = rpm
        self.max_motor_speed = rpm * 2 * np.pi / 60
        # limit the turning speed to make it smooth
        self.max_turn_speed = rpm * 2 * np.pi / 60 / 2

        # Initialize the state and input.
        # If robot doesn't know its initial position, then initialize it to the 0 state.
        # state: [x, y, theta, v]
        if init_state is None:
            self.s = np.array([0, 0, 0, 0])
        else:
            self.s = init_state

        # ------Set parameters for RRT algorithm.------
        self.step_distance = 150
        self.minx = -self.le/2
        self.maxx = self.le/2
        self.miny = -self.w/2
        self.maxy = self.w/2
        self.max_epoch = max_epoch
        self.start = Node(x=init_state[0], y=init_state[1])
        self.target_node = Node(x=self.target[0], y=self.target[1])
        self.nodes = []
        fig = plt.figure(figsize=(9, 6))
        self.ax = fig.add_subplot(111, aspect='equal')
        self.rerun = True


    # Real Action
    def act(self, state, servo_input, dt):
        """
        Given current state and input speed for the wheels, calculate the next state of robot.
        :param state: Current state.
        :param servo_input: Input for 2 wheels.
        :param dt: Time step.
        :return: Next step of the robot.
        """
        control = servo_input

        vl = control[0] * self.r
        vr = control[1] * self.r
        vk = (vr + vl) / 2
        wk = (vr - vl) / self.d
        move_distance = vk * dt
        d_theta = wk * dt

        theta_k = state[2]
        dx = move_distance * math.cos(theta_k + d_theta / 2)
        dy = move_distance * math.sin(theta_k + d_theta / 2)

        next_state = state + np.array((dx, dy, d_theta, 0))
        next_state[2] = np.mod(next_state[2], 2 * np.pi)
        next_state[2] = self.angle_convert(next_state[2])
        next_state[3] = vk

        return next_state

    # ---------------------------2a---------------------------
    # Find the points in V that are closest to the target point
    def closest_points(self, V, target):
        V_closest = []
        min_dis = self.Max_square
        for point in V:
            dis_square = (point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2
            if dis_square < min_dis:
                min_dis = dis_square
                V_closest.clear()
                V_closest.append(point)
            elif dis_square == min_dis:
                V_closest.append(point)

        return V_closest

    # ---------------------------2b---------------------------
    # Generate a smooth achievable trajectory from the inital state
    # towards the target lasting 1 second
    def angle_convert(self, theta):
        """
        Convert the angle to [-pi,pi)

        :param theta:
        :return: converted angle.
        """
        theta_new = theta % (2 * np.pi)
        if theta_new >= np.pi:
            theta_new -= 2 * np.pi
        return theta_new

    def turn(self, cur_state, target_state):
        """
        Given target state, calculate the speed for the robot to reach the target.
        :param cur_state: Current state.
        :param target_state: Target state.
        :return: Speed for the two wheels.
        """
        theta = math.atan((target_state[1] - cur_state[1]) / (target_state[0] - cur_state[0]))
        if target_state[0] < cur_state[0]:
            if theta >= 0:
                theta = theta - np.pi
            else:
                theta = theta + np.pi
        angle = self.angle_convert(cur_state[2] - theta)
        if -0.02 <= angle <= 0.02:
            # if not too far from the target, use a evenly distributed speed
            even_speed = math.sqrt(
                (cur_state[0] - target_state[0]) ** 2 + (cur_state[1] - target_state[1]) ** 2) / self.dt
            if self.max_motor_speed * self.r > even_speed:
                motor_speed = [even_speed / self.r, even_speed / self.r]
            else:
                motor_speed = [self.max_motor_speed, self.max_motor_speed]
        elif -np.pi < angle < -0.02:
            motor_speed = [0, self.max_turn_speed / np.pi * np.abs(angle)]
        elif 0.02 < angle < np.pi:
            motor_speed = [self.max_turn_speed / np.pi * np.abs(angle), 0]
        else:
            motor_speed = [self.max_turn_speed, 0]

        return motor_speed

    def gen_traj(self, target_state, time, ax):
        """
        Given target position, plot the trajectory of the robot.
        :param target_state: Target position.
        :param time: Max move time.
        :param ax: Plot ax parameter.
        :return: Plot the trajectory of the robot.
        """
        step_num = time / self.dt
        collision = 0
        # print(self.rerun)
        for i in range(int(step_num)):
            dx = self.s[0] - target_state[0]
            dy = self.s[1] - target_state[1]
            if not (-1 <= dx <= 1 and -1 <= dy <= 1):
                motor_speed = self.turn(self.s, target_state)
                self.s = self.act(self.s, motor_speed, self.dt)
                if collision == 0:
                    collision = self.collision_check(self.s[:2])
                    if collision == 1:
                        print("This trajectory has collision, the first collision state: ", self.s)
                        self.rerun = True
                        break
                # plt.plot(self.s[0], self.s[1], 'o', markersize='50')
                circle = plt.Circle((self.s[0], self.s[1]), self.diameter/2, edgecolor='c', facecolor='w')
                ax.arrow(self.s[0], self.s[1], np.cos(self.s[2]), np.sin(self.s[2]), head_width=0.5, head_length=15,
                         fc='k', ec='k')
                ax.add_artist(circle)
            # print("motor_speed: ", motor_speed)
            #                print("state: ",self.s)
            else:
                self.rerun = False
                # if collision == 0:
                #     self.rerun = False
                # print(" This trajectory is collision free. ")
                break

    # ---------------------------2c---------------------------
    # Create a visualization of this map with initial and goal states indicated (with obstacles)
    # blue dot: initial position
    # red cross: target position
    def DrawMap(self, initial_state):
        """
        Draw simulation environments with obstacles.

        :param initial_state: Initial state of the robot.
        :return: No return. Plot obstacles.
        """
        plt.cla()
        plt.ion()
        for (ox, oy, wx, wy) in self.obstacleList:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy),  # (x,y)
                    wx,  # width
                    wy,  # height
                )
            )
        plt.plot(initial_state[0], initial_state[1], "bo")
        plt.plot(self.target[0], self.target[1], "xr")
        plt.axis([self.minx, self.maxx, self.miny, self.maxy])
        plt.grid(True)

    # ---------------------------2d---------------------------
    def collision_check(self, current_state):
        """
        Determine whether this trajectory is collision free.
        :param current_state: Current state of the robot.
        :return: If there is a collision return 1, else return 0.
        """
        robot_center = current_state
        collision = 0
        for (ox, oy, wx, wy) in self.obstacleList:
            rectangle_center = [ox + wx / 2, oy + wy / 2]
            v = [abs(robot_center[0] - rectangle_center[0]), abs(robot_center[1] - rectangle_center[1])]
            h = [wx / 2, wy / 2]
            u = [v[0] - h[0], v[1] - h[1]]
            if u[0] < 0:
                u[0] = 0
            if u[1] < 0:
                u[1] = 0
            if u[0] ** 2 + u[1] ** 2 - (self.diameter / 2) ** 2 < 0:
                collision = 1
                break

        return collision

    def path_collision_check(self, current_state):
        """
        Determine whether this trajectory is collision free.
        :param current_state: Current state of the robot.
        :return: If there is a collision return 1, else return 0.
        """
        robot_center = current_state
        collision = 0
        for (ox, oy, wx, wy) in self.obstacleList:
            rectangle_center = [ox + wx / 2, oy + wy / 2]
            v = [abs(robot_center[0] - rectangle_center[0]), abs(robot_center[1] - rectangle_center[1])]
            h = [wx / 2, wy / 2]
            u = [v[0] - h[0], v[1] - h[1]]
            if u[0] < 0:
                u[0] = 0
            if u[1] < 0:
                u[1] = 0
            if u[0] ** 2 + u[1] ** 2 - ((self.diameter+50) / 2) ** 2 < 0:
                collision = 1
                break

        return collision

    # ---------------------------2e---------------------------
    def RRT(self):
        """
        RRT Algorithm.
        :return: Plot every step in RRT. Stop RRT Algorithm when the robot reaches the target.
        """
        self.nodes = []
        self.nodes.append(self.start)
        start_time = time.clock()
        for i in range(self.max_epoch):
            q_rand = self.random_point()
            nearest_id = self.find_nearest_point(q_rand)
            q_nearest = self.nodes[nearest_id]

            theta = math.atan2(q_rand[1]-q_nearest.y, q_rand[0]-q_nearest.x)
            q_new = Node()
            dx = self.step_distance * math.cos(theta)
            dy = self.step_distance * math.sin(theta)
            q_new.x = q_nearest.x + dx
            q_new.y = q_nearest.y + dy
            q_new.parent = nearest_id

            collision_flag = self.path_collision_check([q_new.x, q_new.y])
            if collision_flag:
                continue

            self.nodes.append(q_new)
            if q_new.parent is not None:
                plt.plot([q_new.x, self.nodes[q_new.parent].x], [q_new.y, self.nodes[q_new.parent].y], "-g")
                plt.pause(0.05)

            # Check target
            if self.reach_target(q_new) is True:
                print("Reach Target Position.")
                break
        end_time = time.clock()
        run_time = end_time - start_time
        print("Running time for RRT Algorithm to reach the target is", run_time)
        path = [[self.target_node.x, self.target_node.y]]
        node_index = len(self.nodes) - 1
        while self.nodes[node_index].parent is not None:
            node = self.nodes[node_index]
            path.append([node.x, node.y])
            node_index = node.parent
        path.append([self.start.x, self.start.y])

        return path

    def random_point(self):
        """
        Create a random point in C-space for RRT.
        :return: Random point.
        """
        rand_x = random.uniform(self.minx, self.maxx)
        rand_y = random.uniform(self.miny, self.maxy)
        return [rand_x, rand_y]

    def find_nearest_point(self, rand_node):
        """
        Find the nearest node to the random node in node list.
        :param rand_node: Random node created in previous step.
        :return: The id of the nearest node.
        """
        node_list = self.nodes
        dis_list = [(node.x - rand_node[0])**2+(node.y-rand_node[1])**2 for node in node_list]
        min_index = dis_list.index(min(dis_list))
        return min_index

    def reach_target(self, check_node):
        dx = check_node.x - self.target_node.x
        dy = check_node.y - self.target_node.y
        distance = math.sqrt(dx**2 + dy**2)
        if distance <= self.step_distance:
            return True
        return False

    def show_path(self, robot_path):
        """
        Plot the choose path in RRT Algorithm in red lines.
        :param robot_path: Path.
        :return: Plot.
        """
        plt.plot([x for (x, y) in robot_path], [y for (x, y) in robot_path], '-r')

    def robot_to_target(self, paths):
        """
        Given the path from RRT. Let the robot move towards the target along the path.
        :param paths: Path calculated by the RRT.
        :return: Plot the trajectory of the robot.
        """
        paths = paths[::-1]
        for temp_target in paths[1:]:
            self.gen_traj(temp_target, 15, self.ax)


if __name__ == '__main__':
    target = [800, 500]
    obstacleList = [
        (30, 150, 300, 300),
        (-600, 80, 300, 300),
        (-590, -900, 540, 380),
        (400, 100, 350, 200),
        (-600, -450, 160, 150)
    ]
    initial_state = [0, 0, np.pi / 2, 0]
    robot = Robot(init_state=initial_state, target_pos=target, obstacleList=obstacleList)
    robot.DrawMap(initial_state=initial_state)
    move_path = robot.RRT()
    robot.show_path(move_path)
    robot.robot_to_target(move_path)

    plt.ioff()
    plt.show()





