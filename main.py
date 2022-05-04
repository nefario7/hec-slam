import os
import time
import argparse
import numpy as np

from ekf import *
from factor import *
from utils import *


def argparser():
    args = argparse.ArgumentParser()

    args.add_argument("-m", "--method", type=str, default="ekf", help="ekf or factor", required=True)
    args.add_argument("-d", "--data", type=str, default="data", help="data directory")
    args.add_argument("--num_waypoints", type=int, default=15)

    arguments = args.parse_args()
    return arguments


if __name__ == "__main__":
    params = argparser()

    # * Read Data
    data = read_data(params.data, show_data=False)

    # * Run Localization
    if params.method == "ekf":
        run_ekf(data, params)
    elif params.method == "factor":
        run_factor(data, params)
    else:
        raise Exception(f"Invalid method : {params.method}")
    print("Localization Complete!")

    # * Calibrate

    # # TODO: Find Transformation Matrix from End Effector to Camera (Solve AX = Y)
