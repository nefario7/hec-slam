import os
import time
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)

"""
--------------------------- EKF ---------------------------------
Markers : (x, y, z) in camera frame
Control Inputs : (x, y, z) in global frame

"""


class HEC_SLAM_EKF:
    def __init__(self, sd, num_markers):
        self.k = num_markers

        # Measurement and Control Covariance Matrices
        self.control_covariance = np.diag(sd["control"].values())  # 6x6
        self.measurement_covariance = np.diag(sd["measurement"].values())  # 3x3

        self.pose = np.zeros((6, 1))  # x, y, z, phi, theta, psi
        self.pose_covariance = np.diag(sd["pose"].values())  # 6x6

    def initialize_markers(self, measurement, pose):
        """Initialize the Landmark positions and covariances

        Args:
            measurement (_type_): Measurement of the markers
            pose (_type_): Iinitial pose of the robot
        """
        M = 3 * self.k

        # TODO Initialize Landmark using measurement and pose
        self.landmark = np.zeros((M, 1))  # (x, y, z) * num_markers x 1
        self.landmark_covariance = np.zeros((M, M))

        self.X = np.vstack((self.pose, self.landmark))
        self.P = np.block([[self.pose_covariance, np.zeros((6, M))], [np.zeros((M, 6)), self.landmark_covariance]])

    def get_kalman_gain(self, P_pre, H):
        """_summary_

        Args:
            P_pre (_type_): _description_
            H (_type_): _description_

        Returns:
            K (_type_): Kalman Gain
        """
        meas = np.linalg.inv(H @ P_pre @ H.T + self.marker_covariance)
        K = P_pre @ H.T @ meas
        return K

    def predict(self, new_control_input):
        M = 3 * self.k
        self.X_predicted = np.zeros_like(self.X)
        self.P_predicted = np.zeros_like(self.P)

        mapping = np.hstack((np.eye(6), np.zeros(6, M)))

        # ? Predicted State - g(u, (x, y, theta)(t-1))
        g = np.array([d * np.cos(theta), d * np.sin(theta), alpha])  # Non-linear function

        X_pre = X + F.T @ g
        X_pre[2] = warp2pi(X_pre[2])

        # ? Jacobian - G
        J = np.zeros((3, 3))
        J[0, 2] = -d * np.sin(theta)
        J[1, 2] = d * np.cos(theta)
        G = np.identity(3 + 2 * k) + F.T @ J @ F

        P_predicted = G @ P @ G.T + mapping.T @ self.control_covariance @ mapping

    def update(self, new_measurement):
        beta = new_measurement[::3]
        gamma = new_measurement[1::3]
        r = new_measurement[2::3]

        for i in range(self.k):
            print("Processing Marker {}".format(i))

            # X_pre with x, y, z, phi, theta, psi of robot
            # TODO Write the mesurement error
            # TODO Calculate updated state and covariance
        return P
