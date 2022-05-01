from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np


def draw_covariance(waypoints, measurements, deviations):
    pass


def draw_wp_traj_l(ax, waypoints, measurements):
    pass


def visualization(ax, state, covariance, last_state, mode="init"):
    if mode == "init":
        ax.set_xlabel("$X$", fontsize=10)
        ax.set_ylabel("$Y$", fontsize=10)
        ax.set_zlabel("$Z$", fontsize=10)
        ax.scatter3D(state[0], state[1], state[2], c="k", marker="o", label="Start point")
        ax.scatter3D(state[3::3], state[4::3], state[5::3], c="b", marker="x", label="Markers")
    elif mode == "pred":
        ax.scatter3D(state[0], state[1], state[2], c="r", marker="1", label="Predicted Pose")
        ax.scatter3D(state[3::3], state[4::3], state[5::3], c="k", marker="x", label="Predicted Markers")
        # ax.plot([state[0], last_state[0]], [state[1], last_state[1]], [state[2], last_state[2]], label="Predicted Trajectory")
    elif mode == "update":

        ax.scatter3D(state[0], state[1], state[2], c="b", marker="^", label="Updated Pose")
        ax.scatter3D(state[3::3], state[4::3], state[5::3], c="y", marker="o", label="Updated Markers")

    plt.draw()
    plt.pause(0.5)
