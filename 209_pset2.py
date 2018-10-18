#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:01:17 2018

@author: Di Jin, Qilin Gu
"""

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')

# Question 1(a) ################################################################
""" 
Question 1(a)
Create a space to record the location of the robot and the direction 
"""
Width = 6
Length = 6
stateSpace = np.zeros((Width, Length, 12))

# Question 1(b) ################################################################
"""
Question 1(b)
For actions there are 7 actions which includes the forward or backward move.
and the rotation after the move.
An action [m,r] m represents the movement and the r represents the rotation
    m = 0 no move
    m = 1 forward
    m = -1 backward
    r = 0 no rotation
    r = 1 move left
    r = -1 move right
"""
actionSpace = [(0, 0), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]


'''
*** Note ***. 
For calculate convenience, we use matrix coordinate system 
which means go upward is equal to -x direction, go downward is equal to +x direction, 
go left is equal to -y direction and go right is equal to +y direction. 
It is no influence for calculation, but the input heading needs some transformation.
'''
# Question 1(c) Edited.############################################################
class Environment:
    def __init__(self, width, length, pe):
        self.w = width
        self.l = length
        self.pe = pe
        self.target = (3, 4)
        self.up_heading = [11, 0, 1]
        self.down_heading = [5, 6, 7]
        self.right_heading = [2, 3, 4]
        self.left_heading = [8, 9, 10]
        self.action_space = [(0, 0), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        self.S = []
        for i in range(width):
            for j in range(length):
                for k in range(12):
                    self.S.append((i, j, k))
                    
    '''
    For Each steps, the heading follows below equation:
        next_heading = current_heading + error_rotation + rotation
    If next_heading - rotation = current_heading + error_rotation: turn left/right  prob = pe
    if next_heading - rotation = current_heading + error_rotation: no turn  prob = 1- 2pe
    '''
    def rotate_prob(self, dir1, dir2, rotation):
        steps = dir2 - dir1
        if steps > 6:
            steps = steps - 12
        elif steps < -6:
            steps = steps + 12
        if steps == 0 + rotation:
            pr = 1 - 2 * self.pe
        elif steps == -1 + rotation or steps == 1 + rotation:
            pr = self.pe
        else:
            pr = 0
        return pr
    
    '''
    When heading after the error rotation is  in [11,0,1] and movement = 1, 
    the robots will go upward. Otherwise the probability for go upward is 0.
    Hence the go upward probability is equal to the probability of the heading in [11,0,1] after the error rotate.
    We can call rotate_prob function above to calculate that.
    So do the other move directions.
    '''
    def move_prob(self, movement, rotation, dir1, dir2, heading_set1, heading_set2):
        dir_before_rotate = (dir2 - rotation) % 12
        if movement == 1 and dir_before_rotate in heading_set1:
            pm = self.rotate_prob(dir1, dir2, rotation)
        elif movement == -1 and dir_before_rotate in heading_set2:
            pm = self.rotate_prob(dir1, dir2, rotation)
        else:
            pm = 0
        return pm
    
    '''
    Use above move_prob and rotate_prob to calculate psa.
    '''
    def psa(self, current_state, next_state, action):
        prob = 0
        x = current_state[0]
        y = current_state[1]
        heading = current_state[2]
        next_x = next_state[0]
        next_y = next_state[1]
        next_heading = next_state[2]
        movement = action[0]
        rotation = action[1]

        up_heading = [11, 0, 1]
        down_heading = [5, 6, 7]
        right_heading = [2, 3, 4]
        left_heading = [8, 9, 10]

        # Stay still
        if next_x == x and next_y == y:
            # # The robot will stay for 2 cases:
            # # 1) movement = 0
            # # 2) the robot will move off the grid after the action

            # # Case 1: Movement = 0
            if movement == 0:
                if rotation == 0 and heading == next_heading:
                    prob = 1
                else:
                    prob = 0
            # # Case 2: Movement != 0 and the robot will cross the border after the movement
            else:
                # ## up bound
                if x == 0 and y in range(1, self.l - 1, 1):
                    prob = self.move_prob(movement, rotation, heading, next_heading, up_heading, down_heading)
                # ## down bound
                elif x == self.w - 1 and y in range(1, self.l - 1, 1):
                    prob = self.move_prob(movement, rotation, heading, next_heading, down_heading, up_heading)
                # ## left bound
                elif y == 0 and x in range(1, self.w - 1, 1):
                    prob = self.move_prob(movement, rotation, heading, next_heading, left_heading, right_heading)
                # ## right bound
                elif y == self.l - 1 and x in range(1, self.w - 1, 1):
                    prob = self.move_prob(movement, rotation, heading, next_heading, right_heading, left_heading)
                # ## upper left corner
                elif x == 0 and y == 0:
                    # ### go upward or go left
                    prob = self.move_prob(movement, rotation, heading, next_heading, up_heading, down_heading) \
                           + self.move_prob(movement, rotation, heading, next_heading, left_heading, right_heading)
                # ## upper right corner
                elif x == 0 and y == self.l - 1:
                    # ### go upward or go right
                    prob = self.move_prob(movement, rotation, heading, next_heading, up_heading, down_heading) \
                           + self.move_prob(movement, rotation, heading, next_heading, right_heading, left_heading)
                # ## lower left corner
                elif x == self.w - 1 and y == 0:
                    # ### go downward or go left
                    prob = self.move_prob(movement, rotation, heading, next_heading, down_heading, up_heading) \
                           + self.move_prob(movement, rotation, heading, next_heading, left_heading, right_heading)
                # ## lower right corner
                elif x == self.w - 1 and y == self.l - 1:
                    # ### go downward or go right
                    prob = self.move_prob(movement, rotation, heading, next_heading, down_heading, up_heading) \
                           + self.move_prob(movement, rotation, heading, next_heading, right_heading, left_heading)
                else:
                    prob = 0
        # go upward
        elif next_x == x - 1 and next_y == y:
            prob = self.move_prob(movement, rotation, heading, next_heading, up_heading, down_heading)
        # go downward
        elif next_x == x + 1 and next_y == y:
            prob = self.move_prob(movement, rotation, heading, next_heading, down_heading, up_heading)
        # go left
        elif next_x == x and next_y == y - 1:
            prob = self.move_prob(movement, rotation, heading, next_heading, left_heading, right_heading)
        # go right
        elif next_x == x and next_y == y + 1:
            prob = self.move_prob(movement, rotation, heading, next_heading, right_heading, left_heading)

        return prob
    
    '''
    Define a function to return the next state given the action.
    '''
    def move(self, heading, movement, x, y):
        up_heading = [11, 0, 1]
        down_heading = [5, 6, 7]
        right_heading = [2, 3, 4]
        left_heading = [8, 9, 10]

        # stay still
        if movement == 0:
            return x, y
        # go upward
        elif (heading in up_heading and movement == 1) or (heading in down_heading and movement == -1):
            if x in range(1, self.w):
                x = x - 1
            else:
                pass
        # go downward
        elif (heading in down_heading and movement == 1) or (heading in up_heading and movement == -1):
            if x in range(0, self.w - 1):
                x = x + 1
            else:
                pass
        # go left
        elif (heading in left_heading and movement == 1) or (heading in right_heading and movement == -1):
            if y in range(1, self.l):
                y = y - 1
            else:
                pass
        # go right
        elif (heading in right_heading and movement == 1) or (heading in left_heading and movement == -1):
            if y in range(0, self.l - 1):
                y = y + 1
            else:
                pass

        return x, y

    # Question 1(d) ################################################################
    '''
    For each state, there will be 3 next states because of the error rotation.
    The probability for each heading is the same with error probality for each eroor rotation.
    
    For question 1d we want to return the next state according to the probability, 
    and further we want to use the function to record all the next states and probability.
    We use a parameter called operation to return different kind of result.
    
    If operation = 'move', the function will return a state according to the probability.
    If operation = 'dic', the function will return a dictionary includes all the states and their probability.
    '''
    def next_state(self, current_state, action, operation):
        x = current_state[0]
        y = current_state[1]
        heading = current_state[2]
        movement = action[0]
        rotation = action[1]

        next_heading1 = heading
        next_heading2 = (heading + 1) % 12
        next_heading3 = (heading - 1) % 12

        headings = [next_heading1, next_heading2, next_heading3]

        next_states = []
        for h in headings:
            next_x, next_y = self.move(h, movement, x, y)
            next_heading = (h + rotation) % 12
            next_states.append((next_x, next_y, next_heading))

        state_prob = []
        next_s_dic = {}
        for state in next_states:
            prob = self.psa(current_state, state, action)
            state_prob.append(prob)
            next_s_dic[state] = prob

        if operation == 'move':
            p = random.random()
            if p <= state_prob[0]:
                return next_states[0]
            elif state_prob[0] < p <= state_prob[0] + state_prob[1]:
                return next_states[1]
            else:
                return next_states[2]
        elif operation == 'dic':
            return next_s_dic

    # Question 2 ##############################################################
    '''
    Define reward function.
    '''
    def reward_cal(self, state):
        if state[0] == 0 or state[0] == self.l - 1:
            rw = -100
        elif state[1] == 0 or state[1] == self.w - 1:
            rw = -100
        elif (state[0] == 2 or state[0] == 4) and (state[1] == 2 or state[1] == 3 or state[1] == 4):
            rw = -10
        elif state[0] == 3 and state[1] == 4:
            rw = 1
        else:
            rw = 0
        return rw

    # ###### prepare for 5(b) #################################################
    '''
    Define reward function for 5(b).
    '''
    def reward_cal_2(self, state):
        if state[0] == 0 or state[0] == self.l - 1:
            rw = -100
        elif state[1] == 0 or state[1] == self.w - 1:
            rw = -100
        elif (state[0] == 2 or state[0] == 4) and (state[1] == 2 or state[1] == 3 or state[1] == 4):
            rw = -10
        elif state[0] == 3 and state[1] == 4 and state[2] in self.left_heading:
            rw = 1
        else:
            rw = 0
        return rw

    # Question 3(a) Edited.####################################################
    '''
    Set the intial policy.
    '''
    def policy_cal(self, current_state):
        movement, rotation = 0, 0
        x = current_state[0]
        y = current_state[1]
        heading = current_state[2]
        x_delta = self.target[0] - x
        y_delta = self.target[1] - y

        # If the robot reaches the target, stop acting.
        if x_delta == 0 and y_delta == 0:
            return 0, 0

        # Compute the movement and the rotation.
        # If the target is in front of you, move forward. (movement = 1)
        # If the target is behind the robot, move backward. (movement = 1)
        if heading in self.up_heading:
            movement = 1 if (x_delta <= 0) else -1
            if y_delta == 0:
                rotation = 0
            else:
                rotation = 1 if y_delta > 0 else -1
        elif heading in self.down_heading:
            movement = 1 if (x_delta >= 0) else -1
            if y_delta == 0:
                rotation = 0
            else:
                rotation = 1 if y_delta < 0 else -1
        elif heading in self.left_heading:
            movement = 1 if (y_delta <= 0) else -1
            if x_delta == 0:
                rotation = 0
            else:
                rotation = 1 if x_delta < 0 else -1
        elif heading in self.right_heading:
            movement = 1 if (y_delta >= 0) else -1
            if x_delta == 0:
                rotation = 0
            else:
                rotation = 1 if x_delta > 0 else -1

        return movement, rotation

    # Question 3(b) ###########################################################
    '''
    Plot the trajectory.
    '''
    def plot_trajectory(self, state, policy=None):
        print('Trajectory')
        if policy is None:
            policy = {}
            for s in self.S:
                policy[s] = self.policy_cal(s)

        # Generate grid
        state = (state[0], state[1], (state[2] + 3) % 12)
        act = self.policy_cal(state)
        s0 = copy.deepcopy(state)
        s0_loc = (s0[1] + 0.5, s0[0] + 0.5)

        fig, ax = plt.subplots()
        red1 = plt.Rectangle((0, 0), 1, 6, color='red')
        red2 = plt.Rectangle((1, 0), 5, 1, color='red')
        red3 = plt.Rectangle((5, 1), 1, 5, color='red')
        red4 = plt.Rectangle((1, 5), 4, 1, color='red')
        ax.add_patch(red1)
        ax.add_patch(red2)
        ax.add_patch(red3)
        ax.add_patch(red4)
        yellow1 = plt.Rectangle((2, 4), 3, 1, color='yellow')
        yellow2 = plt.Rectangle((2, 2), 3, 1, color='yellow')
        ax.add_patch(yellow1)
        ax.add_patch(yellow2)
        green1 = plt.Rectangle((4, 3), 1, 1, color='green')
        ax.add_patch(green1)
        # plt.pcolor(Reward, cmap='hot')
        # plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('x')
        ax.xaxis.tick_top()
        ax.set_xlim(0, 6)
        ax.set_ylim(6, 0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='255')
        ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='255')
        plt.plot(s0_loc[0], s0_loc[1], 'o', markersize='10', color='r', )
        print('Start State', (s0[0], s0[1], (s0[2] - 3) % 12))
        paths = []
        action_set = []
        while act != (0, 0):
            s = self.next_state(s0, act, 'move')
            s1 = (s[0], s[1], s[2])
            s1_loc = (s[1] + 0.5, s[0] + 0.5)
            act = policy[s1]
            out_s1 = (s1[0], s1[1], (s1[2] - 3) % 12)
            # print(out_s1)
            paths.append(out_s1)
            action_set.append(act)
            # path = Path([s0_loc, s1_loc], [Path.MOVETO, Path.LINETO])
            # patch = patches.PathPatch(path)
            ax.arrow(s0_loc[0], s0_loc[1], s1_loc[0] - s0_loc[0], s1_loc[1] - s0_loc[1], head_width=0.2,
                     head_length=0.2, fc='b', ec='b')
            # ax.add_patch(patch)

            s0 = copy.deepcopy(s1)
            s0_loc = (s0[1] + 0.5, s0[0] + 0.5)
        for path in paths:
            print('→', path)
        plt.show()

    # Question 3(d) ################################################################
    '''
    Define policy evaluation function.
    '''
    def policy_evaluation(self, discount_factor, accuracy, policy=None):
        value = {}
        for s in self.S:
            value[s] = 0
        if policy is None:
            policy = {}
            for s in self.S:
                policy[s] = self.policy_cal(s)

        delta = 1
        while delta > accuracy:
            delta = 0
            for s in self.S:
                v = 0
                action = policy[s]
                reward = self.reward_cal(s)
                next_s = self.next_state(s, action, 'dic')
                for ss in next_s:
                    psa_s = next_s[ss]
                    # reward = self.reward_cal(ss)
                    v = v + psa_s * (reward + discount_factor * value[ss])
                delta = max(delta, abs(v - value[s]))
                value[s] = v
        return value

    # Question 3(f)################################################################
    def one_step_lookahead(self, value, discount):
        one_step_policy = {}
        for s in self.S:
            v = np.zeros(7)
            reward = self.reward_cal(s)
            for i in range(7):
                action = self.action_space[i]
                next_s = self.next_state(s, action, 'dic')
                for ss in next_s:
                    psa_s = next_s[ss]
                    v[i] = v[i] + psa_s * (reward + discount * value[ss])
            best_index = np.argmax(v)
            one_step_policy[s] = self.action_space[best_index]
        return one_step_policy
    
    # Question 3(g)################################################################
    '''
    Define policy iteration function.
    '''
    def policy_iter(self, policy, discount_factor, accuracy):
        print('Policy iteration starts.')
        policy_unstable = 1
        while policy_unstable:
            value = self.policy_evaluation(discount_factor, accuracy, policy)
            # policy_unstable = 0
            policy_new = self.one_step_lookahead(value, discount_factor)
            if policy_new != policy:
                policy = policy_new   
            else:
                policy_unstable = 0
                print('Policy iteration ends.')
        return policy_new, value

    # Question 4(a) ################################################################
    '''
    Define value iteration function.
    '''
    def value_iteration(self, discount_factor, accuracy, reward_name=None):
        print('Value iteration starts.')
        value = {}
        policy = {}
        for s in self.S:
            policy[s] = self.policy_cal(s)
            value[s] = 0

        delta = 1
        while delta > accuracy:
            delta = 0
            for s in self.S:
                if reward_name is None:
                    reward = self.reward_cal(s)
                elif reward_name == 'new':
                    reward = self.reward_cal_2(s)
                action_value = np.zeros(len(self.action_space))
                for i in range(len(self.action_space)):
                    action = self.action_space[i]
                    next_s = self.next_state(s, action, 'dic')
                    for ss in next_s:
                        # psa_s = self.psa(s, ss, action)
                        psa_s = next_s[ss]
                        action_value[i] += psa_s * (reward + discount_factor * value[ss])
                max_v = np.max(action_value)
                max_a = self.action_space[np.argmax(action_value)]
                delta = max(delta, abs(max_v - value[s]))
                value[s] = max_v
                policy[s] = max_a
        print('Value iteration ends.')
        # print('Running time for value iteration:', running_time)
        return value, policy


if __name__ == '__main__':
    obj = Environment(Width, Length, 0)

    # Question 3(c) ###########################################################
    init_state = (1, 4, 6)
    print('###### Question 3(c) Trajectory for initial policy ######')
    obj.plot_trajectory(init_state)
    v = obj.policy_evaluation(0.9, 0.000001)
    state_trans = (init_state[0], init_state[1], (init_state[2]+3)%12)
    
    # Question 3(e) ###########################################################
    print('###### Question 3(e) Trajectory value for initial state ######')
    print('The value for intial state',init_state,'is', v[state_trans])
    print()
    
    # Question 3(g) ###########################################################
    policy_init = {}
    for i in range(Width):
        for j in range(Length):
            for k in range(12):
                policy_init[(i, j, k)] = obj.policy_cal((i, j, k))
    start_time = time.time()
    policy_pi, value_pi = obj.policy_iter(policy_init, 0.9, 0.000001)
    end_time = time.time()
    running_time = end_time - start_time
    
    # Question 3(h) ###########################################################
    print('###### Question 3(h) Trajectory after policy iteration ######')
    obj.plot_trajectory(init_state, policy_pi)
    print('The value for state', init_state, 'is', value_pi[state_trans])
    print()
    
    # Question 3(h) ###########################################################
    print('###### Question 3(i) Running time for policy iteration ######')
    print('Running time:', running_time)
    print()

    # Question 4 ###########################################################
    print('###### Question 4(a) Trajectory value after value iteration ######')
    start_time = time.time()
    value_vi, policy_vi = obj.value_iteration(0.9, 0.00001)
    end_time = time.time()
    running_time = end_time - start_time
    print('The value for state', init_state, 'is', value_vi[state_trans])
    print()
    
    print('###### Question 4(b) Trajectory after value iteration ######')
    obj.plot_trajectory(init_state, policy_vi)
    print()
    
    print('###### Question 4(c) Running time for value iteration ######')
    print('Running time:', running_time)
    print()

    # Question 5(a) ###########################################################
    print('###### Question 5(a) Trajectory with pe = 25% ######')
    obj2 = Environment(Width, Length, 0.25)    
    value_vi_2, policy_vi_2 = obj2.value_iteration(0.9, 0.00001)    
    obj2.plot_trajectory(init_state, policy_vi_2)
    print('The value for intial state',init_state,'after value iteration is',value_vi_2[state_trans])
    print()

    # Question 5(b) ###########################################################
    print('###### Question 5(b) Trajectory with new reward function(pe = 0%) ######')
    value_vi_new, policy_vi_new = obj.value_iteration(0.9, 0.00001, 'new')
    print('The value for intial state',init_state,'after value iteration is',value_vi_new[state_trans])
    obj.plot_trajectory(init_state, policy_vi_new)

    print('###### Question 5(b) Trajectory with new reward function(pe = 25%) ######')
    value_vi_new_2, policy_vi_new_2 = obj2.value_iteration(0.9, 0.00001, 'new')
    print('The value for intial state',init_state,'after value iteration is',value_vi_new_2[state_trans])
    obj2.plot_trajectory(init_state, policy_vi_new_2)
