import numpy as np
from math import sin, cos
import sympy as sp
from numpy.random import randn
import matplotlib.pyplot as plt


class Robot:
    def __init__(self, environment_length=750, environment_width=500, wheel_radius=20, wheel_distance=85,
                 delta_t=0.2, initial_state=None, rpm=130,
                 sigma_range=0.03, sigma_bearing=np.radians(0.1), input_error_rate=0.05):

        # Initialize the environment and the robot.
        self.le = environment_length
        self.w = environment_width
        self.r = wheel_radius
        self.d = wheel_distance
        self.dt = delta_t

        # Setting the max motor speed as 130 RPM at 6V
        self.rpm = rpm
        self.max_motor_speed = rpm * 2 * np.pi / 60
        self.range_std = sigma_range
        self.bearing_std = sigma_bearing

        # Setting the max motor speed as 130 RPM at 6V
        self.rpm = rpm
        self.max_motor_speed = rpm * 2 * np.pi / 60
        self.speed_error_range = input_error_rate
        self.speed_std = self.max_motor_speed * self.speed_error_range
        self.input_error = np.random.normal(0, self.speed_error_range**2)

        # Initialize the state and input.
        # If robot doesn't know its initial position, then initialize it to the 0 state.
        if initial_state is None:
            self.s = np.array([0, 0, 0, 0, 0])
        else:
            self.s = initial_state


        # if initial_input is None:
            # self.u = np.zeros((1, 2))

        # Setting symbol function for updating.
        # Time updating Function
        x, y, theta, wl, wr, ww1, ww2, v, r, w, dt, bias = sp.symbols('x, y, theta, wl, wr, '
                                                                      'ww1, ww2, v, r, w, dt, bias')
        self.f = sp.Matrix([[x + (1 / 2 * (wl + ww1 + wr + ww2) * r) * sp.cos(theta) * dt],
                            [y + (1 / 2 * (wl + ww1 + wr + ww2) * r) * sp.sin(theta) * dt],
                            [theta + ((wl - wr) * r) / w * dt],
                            [v],
                            [bias]])

        # Observation Updating Function

        # Time update function: x_{t+1} = f(x_t,u_t,w_t)
        # F: time update process matrix
        # G: input process matrix
        # Q: Noise covariance matrix
        # Sigma: Covariance matrix for state
        self.symbol_state = sp.Matrix([x, y, theta, v, bias])
        self.symbol_input = sp.Matrix([ww1, ww2])
        self.F = self.f.jacobian(self.symbol_state)
        self.W = self.f.jacobian(self.symbol_input)
        self.Q = np.eye(2)
        self.R = np.diag([1200*sigma_range ** 2, 1200*sigma_range ** 2, sigma_bearing ** 2])
        self.sigma = np.eye(5)

    # Real Action
    def act(self, state, servo_input, d_step):
        control = servo_input + self.input_error

        vl = control[0] * self.r
        vr = control[1] * self.r
        vk = (vl + vr) / 2
        wk = (vl - vr) / self.d
        move_distance = vk * d_step
        d_theta = wk * d_step

        theta_k = state[2]
        dx = move_distance * cos(theta_k+d_theta/2)
        dy = move_distance * sin(theta_k+d_theta/2)

        next_state = state + np.array((dx, dy, d_theta, 0, 0))
        next_state[2] = np.mod(next_state[2], 2 * np.pi)
        next_state[3] = vk

        return next_state

    # Time Update Functions.
    def time_update(self, servo_input, step):
        """
        Time Update.
        Update the mean state x_{t+1} and state covariance sigma.

        :param servo_input: angular velocity values for each wheel
        :return: No return. Update self.state
        """

        # Update the mean state.
        self.s = self.act(self.s, servo_input, self.dt * step)

        # Update the covariance.
        x, y, theta, wl, wr, ww1, ww2, v, r, w, dt, bias = sp.symbols('x, y, theta, wl, wr, ww1, ww2, v, r, w, dt, bias')
        subs_value = {x: self.s[0], y: self.s[1], theta: self.s[2], v: self.s[3], bias: self.s[4], wl: servo_input[0],
                      wr: servo_input[1], ww1: self.input_error, ww2: self.input_error, dt: self.dt, r: self.r, w: self.d}

        # Calculate F_t and W_t
        Ft = np.array(self.F.evalf(subs=subs_value))
        Wt = np.array(self.W.evalf(subs=subs_value))
        self.Q = np.array([[self.input_error**2, 0], [0, self.input_error**2]])

        self.sigma = np.dot(Ft, self.sigma).dot(Ft.T) + np.dot(Wt, self.Q).dot(Wt.T)

    # Observe Functions.
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

    def distance_cal(self, x, y, theta, boundary):
        """
        help calculate hx

        :param x: loc x
        :param y: loc y
        :param theta: heading of the robot.
        :param boundary: different boundary for different calculation cases.
        :return:
        """
        theta = self.angle_convert(theta)

        if 0 <= y + (boundary[0] - x) * np.tan(theta) <= self.w:
            return (boundary[0] - x) / np.cos(theta)
        else:
            return (boundary[1] - y) / np.sin(theta)

    def observe_without_noise(self, state):
        """
        Takes state variables [x, y, angle] and returns the measurement [d_front, d_right, theta] with 0 noise

        :param state: Current state.
        :return: measurement without noise.
        """
        theta = self.angle_convert(state[2])
        x = state[0]
        y = state[1]

        if -np.pi <= theta < -np.pi / 2:
            if theta != -np.pi:
                boundary = [0, 0]
                d_front = self.distance_cal(x, y, theta, boundary)
                d_right = self.distance_cal(x, y, theta - np.pi / 2, boundary)
            else:
                d_front = x
                d_right = self.w - y
        elif 0 > theta >= -np.pi / 2:
            if theta != -np.pi / 2:
                boundary = [self.le, 0]
                d_front = self.distance_cal(x, y, theta, boundary)
                d_right = self.distance_cal(x, y, theta - np.pi / 2, boundary)
            else:
                d_front = y
                d_right = x
        elif 0 <= theta < np.pi / 2:
            if theta != 0:
                boundary = [self.le, self.w]
                d_front = self.distance_cal(x, y, theta, boundary)
                d_right = self.distance_cal(x, y, theta - np.pi / 2, boundary)
            else:
                d_front = self.le - x
                d_right = y
        else:
            if theta != np.pi / 2:
                boundary = [0, self.w]
                d_front = self.distance_cal(x, y, theta, boundary)
                d_right = self.distance_cal(x, y, theta - np.pi / 2, boundary)
            else:
                d_front = self.w - y
                d_right = self.le - x
        return np.array([d_front, d_right, theta])

    def noise_reading(self):
        """
        Set measurement noise.

        :return: sensor noise.
        """
        sensor_noise = [0, 0, 0]
        sensor_noise[0] += randn() * self.range_std * 1200
        sensor_noise[1] += randn() * self.range_std * 1200
        sensor_noise[2] += randn() * self.bearing_std

        return np.array(sensor_noise)

    def observe(self, state):
        """
        observation with noise.

        :return: observation result.
        """
        result_without_noise = self.observe_without_noise(state)
        noise = self.noise_reading()
        observation_result = result_without_noise + noise

        return observation_result

    # Observation Update Functions.
    def partial_diff(self, x, y, theta, boundary):
        """
        Help to calculate the H Matrix. Return Hi.

        :param x: loc x
        :param y: loc y
        :param theta: heading of the robot.
        :param boundary: different boundary for different calculation cases.
        :return: Hi for input cases.
        """
        if 0 <= y + (boundary[0] - x) * np.tan(theta) <= self.w:
            return np.array([-1 / np.cos(theta), 0, (boundary[0] - x) * np.sin(theta) / np.cos(theta) ** 2])
        else:
            return np.array([0, -1 / np.sin(theta), -(boundary[1] - y) * np.cos(theta) / np.sin(theta) ** 2])

    def H_d(self, state, direction):
        """
        Calculate the Hi matrix to for H matrix.

        :param state: current state
        :param direction: measure direction(front or right)
        :return: Hi matrix
        """
        """
        takes state variables [x, y, angle] and returns the H [dH/dx, dH/dy, dH/dtheta]
        """
        if direction == 1:
            theta = self.angle_convert(state[2])
        else:
            theta = self.angle_convert(state[2] - np.pi / 2)

        x = state[0]
        y = state[1]

        if -np.pi <= theta < -np.pi/2:
            if theta != -np.pi:
                boundary = [0, 0]
                return self.partial_diff(x, y, theta, boundary)
            else:
                return np.array([1, 0, 0])
        elif 0 > theta >= -np.pi / 2:
            if theta != -np.pi / 2:
                boundary = [self.le, 0]
                return self.partial_diff(x, y, theta, boundary)
            else:
                return np.array([0, 1, 0])
        elif 0 <= theta < np.pi / 2:
            if theta != 0:
                boundary = [self.le, self.w]
                return self.partial_diff(x, y, theta, boundary)
            else:
                return np.array([-1, 0, 0])
        else:
            if theta != np.pi / 2:
                boundary = [0, self.w]
                return self.partial_diff(x, y, theta, boundary)
            else:
                return np.array([0, -1, 0])

    def H(self, state):
        """
        Calculate the H matrix used in observation update.

        :param state: current state
        :return: H matrix for calculating.
        """
        H_d1 = self.H_d(state, 1)
        H_d2 = self.H_d(state, 2)
        theta = np.array([0, 0, 1])

        return np.array([H_d1, H_d2, theta])

    def observation_update(self, observe_result):
        """
        Observation Update.

        :return: Updated state.
        """
        Ht = np.array(self.H(self.s[:3]))
        y0 = self.observe_without_noise(self.s)
        yt = observe_result

        res = yt - y0
        M1 = np.dot(self.sigma[:3, :3], Ht.transpose())
        M2 = np.matrix(np.dot(Ht, self.sigma[:3, :3]).dot(Ht.transpose())+self.R, dtype=float)
        M3 = np.linalg.inv(M2)
        K = np.dot(M1,M3)
        self.s[:3] = self.s[:3] + K.dot(res)
        self.sigma[:3, :3] = self.sigma[:3, :3] - np.dot(K, Ht).dot(self.sigma[:3, :3])


# Define simulation function.
def simulation(simulation_time, start_state, robot_control, steps, well_know):
    """
    Evaluation the EKF Model.

    :param simulation_time: Set the simulation time you want.
    :param start_state: Start state.
    :param robot_control: Instruction for robot.
    :param steps: Update step.
    :param well_know: If the robot know its initial position
    :return: True path and path estimated by EKF.
    """
    if well_know is True:
        robot = Robot(initial_state=start_state)
    else:
        robot = Robot()
    s = start_state
    dt = robot.dt
    x_trajectory = []
    y_trajectory = []
    observation = []
    x_ekf_estimate = []
    y_ekf_estimate = []

    for i in range(int(simulation_time/dt)):
        control = robot_control[i]

        # STEP 1. Move.
        s = robot.act(s, control, dt)
        x_trajectory.append(s[0])
        y_trajectory.append(s[1])

        # Check if the robot torch the wall.
        if not (0 <= s[0] <= 750 and 0 <= s[1] <= 500):
            print("Out of bound!")
            break

        if i % steps == 0:
            # STEP 1. Time Update
            robot.time_update(control, steps)

            # STEP 2. Observation.
            observe_result = robot.observe(s)
            observation.append(observe_result)

            # STEP 3. Observation Update.
            robot.observation_update(observe_result)
            x_ekf_estimate.append(robot.s[0])
            y_ekf_estimate.append(robot.s[1])

    return x_trajectory, y_trajectory, x_ekf_estimate, y_ekf_estimate


# Define plot function.
def plot_trajectory(x_path, y_path, x_estimator, y_estimator):
    """
    Plot the trajectory.

    :param x_path: x coordinate for true path.
    :param y_path: y coordinate for true path.
    :param x_estimator: x coordinate for EKF estimator.
    :param y_estimator: y coordinate for EKF estimator.
    :return: Plot.
    """
    plt.figure()
    plt.grid(True)
    plt.xlim((0, 750))
    plt.ylim((0, 500))

    plt.plot(x_path, y_path, marker='*', markersize=2)
    plt.plot(x_estimator, y_estimator, marker='d', markersize=4)
    plt.legend(['True trajectory', 'Estimated Trajectory'])
    plt.title('Extended Kalman Filter')
    plt.show()


if __name__ == '__main__':
    # Setting initial robot state and inputs.
    run_time = 40
    init_state = np.array([80, 100, np.pi/3, 0, 0])

    # Set the instruction for robot.
    instruction = np.zeros((1000, 2))
    for i in range(1000):
        if i <= 50:
            instruction[i] = np.array([1, 1.4])
        elif i > 50:
            instruction[i] = np.array([1.2, 1.1])

    # Evaluation
    x1, y1, x2, y2 = simulation(run_time, init_state, instruction, steps=5, well_know=False)
    plot_trajectory(x1, y1, x2, y2)
