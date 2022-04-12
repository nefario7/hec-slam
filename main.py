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

    # ? How to initialize the EKF?
    ekf.initialize_markers()

    while True:
        measurement = get_sensor_data()

        # if measurement = control:
        self._predict()
        # elif measurement = marker:
        self._update()
