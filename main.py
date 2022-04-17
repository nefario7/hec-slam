import os
import time
import json
import argparse
import numpy as np


def argparser():
    args = argparse.ArgumentParser()

    args.add_argument("--num_markers", type=int, default=12, required=True)
    args.add_argument("--dev", type=str, default="deviations.json", required=True)
    args.add_argument("--data", type=str, default="dummy_data.txt")

    args = argparser.parse_args()
    return args


if __name__ == "__main__":
    params = argparser()

    if params.data is not None:
        data_file = open(params.data, "r")

    with open(params.dev, "r") as f:
        deviations = json.load(f)

    ekf = HEC_SLAM_EKF(deviations, params.num_markers)
    ekf.initialize_markers()  # ? How to initialize the EKF?

    pipe, profile = setup_realsense()

    ekf_history = {}

    while True:
        measurements = get_measurement(pipe, profile)

        predictions = ekf.predict()
        updates = ekf.update(measurements)

        display_data(predictions, updates, measurement)
        ekf_history["{}".format(time.time())] = History(predictions, updates, measurement)
