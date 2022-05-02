import os
import time
import numpy as np
import matplotlib.pyplot as plt

from visualize import *

np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)

"""
--------------------------- EKF ---------------------------------
Markers : (x, y, z) in camera frame
Control Inputs : (x, y, z) in global frame
"""


class History:
    def __init__(self, X_pre, P_pre, measurements):
        self.predictions = {}
        self.measurements = {}

        self.predictions["State"] = X_pre
        self.predictions["P"] = P_pre
        for i, data in enumerate(measurements):
            self.measurements["Marker {}".format(i)] = data

    def show_history(self):
        print("Predictions:")
        print(self.predictions)
        print("Measurements:")
        print(self.measurements)


class HEC_SLAM_EKF:
    def __init__(self, sd, num_markers, landmarks):
        self.k = num_markers
        self.landmark_ids = landmarks

        # * Measurement and Control Covariance Matrices
        self.control_covariance = np.diag(list(sd["control"].values()))  # 6x6
        self.measurement_covariance = np.diag(list(sd["measurement"].values()))  # 3x3

        # * Robot Pose Covariances
        self.pose = np.zeros((6, 1))  # x, y, z, phi, theta, psi
        self.pose_covariance = np.diag(list(sd["pose"].values()))  # 6x6

        self.X = None  # State (6 + 3 * k)x1
        self.P = None  # Covariance (6 + 3 * k)x(6 + 3 * k)
        self.X_predicted = None  # Predicted State (6 + 3 * k)x1
        self.P_predicted = None  # Predicted Covariance (6 + 3 * k)x(6 + 3 * k)

    def rotate_point(self, orientation, point):
        phi, theta, psi = orientation
        s1 = np.sin(phi)
        c1 = np.cos(phi)
        s2 = np.sin(theta)
        c2 = np.cos(theta)
        s3 = np.sin(psi)
        c3 = np.sin(psi)
        R = np.zeros((3, 3))
        R[0, 0] = c1 * c2
        R[0, 1] = c1 * s2 * s3 - c3 * s1
        R[0, 2] = s1 * s3 + c1 * c3 * s2
        R[1, 0] = c2 * s1
        R[1, 1] = c1 * c3 + s1 * s2 * s3
        R[1, 2] = c3 * s1 * s2 - c1 * s3
        R[2, 0] = -s2
        R[2, 1] = c2 * s3
        R[2, 2] = c2 * c3
        return R @ point

    def initialize_markers(self, init_measurement, init_pose):
        M = 3 * self.k

        # * Initialize Landmark using measurement and pose
        self.landmark = np.array([init_measurement[i] for i in self.landmark_ids]).reshape(-1, 1)  # (x, y, z) * num_markers x 1
        self.landmark_covariance = np.zeros((M, M))

        orientation = init_pose[3:6]
        x_y_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.landmark = ((init_pose[0:3] + x_y_z @ (self.rotate_point(orientation, self.landmark.reshape(3, -1)))).T).flatten()

        self.pose = np.array(init_pose).reshape(-1, 1)  # (x, y, z, phi, theta, psi) x 1

        self.X = np.vstack((self.pose, self.landmark))
        self.P = np.block([[self.pose_covariance, np.zeros((6, M))], [np.zeros((M, 6)), self.landmark_covariance]])

    def get_kalman_gain(self, P_pre, H):
        P = H @ P_pre @ H.T
        meas = np.linalg.inv(P + self.measurement_covariance)
        K = P_pre @ H.T @ meas
        return K

    def predict(self, new_control_input):
        M = 3 * self.k
        self.X_predicted = np.zeros_like(self.X)
        self.P_predicted = np.zeros_like(self.P)
        F = np.hstack((np.eye(6), np.zeros((6, M))))

        # * Predicted State
        '''
        x_prime, y_prime, z_prime = new_control_input[0:3]
        phi_prime, theta_prime, psi_prime = new_control_input[3:6]
        g = np.array([x_prime, y_prime, z_prime, phi_prime, theta_prime, psi_prime]).reshape(-1, 1)  # Non-linear function
        '''
        x_y_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        g_xyz = x_y_z @ (self.rotate_point((self.X_predicted[3:6]), new_control_input[0:3]))
        g = np.hstack((g_xyz, new_control_input[3:6]))
        self.X_predicted = self.X + F.T @ g

        # * Predicted Covariance
        J = np.identity(6)
        J = F.T @ J @ F
        self.P_predicted = J @ self.P @ J.T + F.T @ self.control_covariance @ F

        return self.X_predicted, self.P_predicted

    def update(self, new_control_input, new_measurement):
        measurements = {i: new_measurement[i] for i in self.landmark_ids}
        positions = np.array(list(measurements.values()))

        for i, (marker, position) in enumerate(measurements.items()):
            print("Processing Marker {}".format(marker))

            phi, theta, psi = self.X_predicted[3:6]
            x, y, z = new_control_input[0:3] # orientation ignored as not needed in Jacobian
            measure_pre = self.X_predicted[(6 + 3 * i) : (9 + 3 * i)].reshape(-1, 1)

            orientation = self.X_predicted[3:6]
            x_y_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            position = (self.X_predicted[0:3] + x_y_z @ (self.rotate_point(orientation, position))).flatten()
            measure_tru = np.array(position).reshape(-1, 1)
            measure_error = measure_tru - measure_pre
            # print(measure_pre, measure_tru)
            # print(measure_error)

            s1 = np.sin(phi)
            c1 = np.cos(phi)
            s2 = np.sin(theta)
            c2 = np.cos(theta)
            s3 = np.sin(psi)
            c3 = np.sin(psi)

            # * Jacobian
            # H_p = np.hstack((np.eye(3), np.zeros((3, 3))))  # 3 x 6
            # H_l = np.eye(3)
            x_y_z = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])
            H_p_xyz = np.eye(3)
            H_p_o = np.zeros((3, 3))
            H_p_o[0, 0] = -s1*c2*x + (-s1*s2*s3-c3*c1)*y + (c1*s3-s1*c3*s2)*z
            H_p_o[0, 1] = -c1*s2*x + (c1*c2*s3)*y + c1*c3*c2*z
            H_p_o[0, 2] = c2*y + c2*z
            H_p_o[1, 0] = c2*c1*x + (-s1*c3+c1*s2*s3)*y + (c3*c1*s2+s1*s3)*z
            H_p_o[1, 1] = -s2*s1*x + s1*c2*s3*y + c3*s1*c2*z
            H_p_o[1, 2] = (-c1*s3+s1*s2*c3)*y + (-s3*s1*s2-c1*c3)*z
            H_p_o[2, 0] = 0
            H_p_o[2, 1] = -c2*x - s2*s3*y -s2*c3*z
            H_p_o[2, 2] = c2*c3*y - c2*s3*z
            H_p_o = x_y_z @ H_p_o
            H_p = np.hstack((H_p_xyz, H_p_o))
            H_l = x_y_z
            H_low = np.hstack((H_p, H_l))  # 3 x 9

            # * Mapping matrix
            F = np.zeros((H_low.shape[1], 6 + 3 * self.k))
            F[:6, :6] = np.identity(6)
            F[6:, (3 * i + 6) : (3 * i + 9)] = np.identity(3)
            H = H_low @ F

            # * Update state and covariances
            K = self.get_kalman_gain(self.P_predicted, H)
            self.X_predicted = K @ measure_error + self.X_predicted
            self.P_predicted = (np.identity(6 + positions.size) - K @ H) @ self.P_predicted

            # self.X = self.X_predicted
            # self.P = self.P_predicted
        return self.X_predicted, self.P_predicted
