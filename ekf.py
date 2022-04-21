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
    def __init__(self, sd, num_markers):
        self.k = num_markers

        # Measurement and Control Covariance Matrices
        self.control_covariance = np.diag(sd["control"].values())  # 6x6
        self.measurement_covariance = np.diag(sd["measurement"].values())  # 3x3

        self.pose = np.zeros((6, 1))  # x, y, z, phi, theta, psi
        self.pose_covariance = np.diag(sd["pose"].values())  # 6x6

        self.X = None  # State (6 + 3 * k)x1
        self.P = None  # Covariance (6 + 3 * k)x(6 + 3 * k)
        self.X_predicted = None  # Predicted State (6 + 3 * k)x1
        self.P_predicted = None  # Predicted Covariance (6 + 3 * k)x(6 + 3 * k)

    def initialize_markers(self, measurement, pose):
        M = 3 * self.k

        # TODO Initialize Landmark using measurement and pose
        self.landmark = np.zeros((M, 1))  # (x, y, z) * num_markers x 1
        self.landmark_covariance = np.zeros((M, M))

        self.X = np.vstack((self.pose, self.landmark))
        self.P = np.block([[self.pose_covariance, np.zeros((6, M))], [np.zeros((M, 6)), self.landmark_covariance]])

    def get_kalman_gain(self, P_pre, H):
        meas = np.linalg.inv(H @ P_pre @ H.T + self.marker_covariance)
        K = P_pre @ H.T @ meas
        return K

    def predict(self, new_control_input):
        M = 3 * self.k
        self.X_predicted = np.zeros_like(self.X)
        self.P_predicted = np.zeros_like(self.P)
        F = np.hstack((np.eye(6), np.zeros(6, M)))

        # ? Predicted State
        x_prime, y_prime, z_prime = new_control_input[0:3]
        phi_prime, theta_prime, psi_prime = new_control_input[3:6]
        g = np.array([x_prime, y_prime, z_prime, phi_prime, theta_prime, psi_prime])  # Non-linear function
        self.X_predicted = self.X + F.T @ g

        # ? Predicted Covariance
        J = np.identity(6)
        J = np.identity(6 + M) + F.T @ J @ F
        self.P_predicted = J @ self.P @ J.T + F.T @ self.control_covariance @ F

        return self.X_predicted, self.P_predicted

    def update(self, new_measurement):
        ids = list(new_measurement.keys())
        positions = np.array(list(new_measurement.values()))

        for i, marker in enumerate(ids):
            print("Processing Marker {}".format(marker))

            # X_pre with x, y, z, phi, theta, psi of robot
            x_camera, y_camera, z_camera, phi, theta, psi = self.X_predicted[0:6]
            # Write the mesurement error
            measure_pre = self.X_predicted[(6+3*i):(9+3*i)]
            measure_tru = positions[(3*i):(3+3*i)]
            measure_error = measure_tru - measure_pre

            # Jacobian
            H_p = np.identity(6)    # 6 x 6
            H_l = np.vstack((np.identity(3), np.zeros((3, 3)))) # 6 x 3
            H_low = np.hstack((H_p, H_l)) # 6 x 9
            
            # Mapping matrix
            F = np.zeros((H_low.shape[1], 6 + positions.size))
            F[:6, :6] = np.identity(6)
            F[6:, (3*i+6):(3*i+9)] = np.identity(3)
            H = H_low @ F

            # Update state and covariances
            K = self.get_kalman_gain(self.P_predicted, H)
            self.X_predicted = K @ measure_error + self.X_predicted
            self.P_predicted = (np.identity(6 + positions.size) - K @ H) @ self.P_predicted

        return self.X_predicted, self.P_predicted
