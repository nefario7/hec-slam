import os
import time
import argparse
import numpy as np

from ekf import *
from utils import *

# from measurement import *


def argparser():
    args = argparse.ArgumentParser()

    args.add_argument("--num_waypoints", type=int, default=15)
    args.add_argument("--data", type=str, default="data")

    arguments = args.parse_args()
    return arguments


if __name__ == "__main__":
    params = argparser()

    if params.data is not None:
        data = read_data(params.data, show_data=False)

    if isinstance(data, tuple):
        # * Offline Localization
        waypoints, measurements, deviations = data

        ekf = HEC_SLAM_EKF(deviations, num_markers=6, landmarks=[1, 2, 3, 9, 10, 12])
        ekf.initialize_markers(measurements[1], waypoints[1])
        ekf_history = {}

        ax = plt.axes(projection="3d")
        visualization(ax, ekf.X, ekf.P, ekf.X)
        last_state = ekf.X
        final_waypoint = waypoints[1]

        for i in range(2, params.num_waypoints + 1):
            # print("-" * 100)
            print("Waypoint {} : \t\t{}".format(i, waypoints[i]))

            # * Predict
            predictions = ekf.predict(waypoints[i])
            # print("Preds : \t\t", predictions[0][:3].reshape(1, -1)[0])
            visualization(ax, predictions[0], predictions[0], last_state, mode="pred")
            last_state = predictions[0]

            # * Update
            updates = ekf.update(waypoints[i], measurements[i])
            visualization(ax, updates[0], updates[1], last_state, mode="update")

            # display_data(predictions, updates, measurements[i])
            ekf_history["{}".format(time.time())] = History(predictions, updates, measurements[i])

            localized_camera = updates[0][:6]
            final_waypoint += waypoints[i]

            if i == params.num_waypoints:
                print("Final Waypoint: \tx: {}, y: {}, z: {}".format(final_waypoint[0], final_waypoint[1], final_waypoint[2]))
        print(
            "Localization: \t\tx:{:.2f}, y:{:.2f}, z:{:.2f}".format(localized_camera[0][0], localized_camera[1][0], localized_camera[2][0])
        )

        # * Plot Data
        ax.scatter3D(final_waypoint[0], final_waypoint[1], final_waypoint[2], c="#17becf", marker="D", label="Final Waypoint")
        ax.scatter3D(localized_camera[0], localized_camera[1], localized_camera[2], c="g", marker="o", label="Localization")
        plt.legend(loc="upper right")
        plt.show()

    else:
        # * Online Localization
        deviations = data

        ekf = HEC_SLAM_EKF(deviations, params.num_waypoints)
        ekf.initialize_markers()

        pipe, profile = setup_realsense()
        ekf_history = {}
        k = 0
        while True:
            print("Waypoint {}".format(i))
            measurements = get_measurement(pipe, profile)

            predictions = ekf.predict(waypoints[i])
            updates = ekf.update(measurements[i])

            display_data(predictions, updates, measurement)
            ekf_history["{}".format(time.time())] = History(predictions, updates, measurement)

            localized_camera = updates[0][:6]
            print("Localization: {}".format(localized_camera))

            k += 1
            if k == params.num_waypoints:
                break

    print("Localization Complete!")

    # # TODO: Find Transformation Matrix from End Effector to Camera (Solve AX = Y)
