import os
import sys
import json

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def warp2pi(angle_rad):
    """_summary_

    Args:
        angle_rad (_type_): Input angle in radians

    Returns:
        _type_: Angle in radians in the range [-pi, pi]
    """
    if angle_rad > np.pi or angle_rad < -np.pi:
        angle_rad_warped = angle_rad - 2 * np.pi * np.floor((angle_rad + np.pi) / (2 * np.pi))
        return angle_rad_warped
    else:
        return angle_rad


def display_data(predictions, updates, measurement):
    """Prints data

    Args:
        predictions (_type_):
        updates (_type_):
        measurement (_type_):

    Returns:
        _type_: None
    """
    print("\nPredictions:")
    print("X : {} and P : {}".format(predictions[0], predictions[1]))
    print("\nUpdates:")
    print("X : {} and P : {}".format(updates[0], updates[1]))

    print("\nMeasurement:")
    print(measurement)


def read_data(data_path, show_data):
    """_summary_

    Args:
        data_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        with open(os.path.join(data_path, "data.csv"), "r") as f:
            data = pd.read_csv(data_path, header=None)
    except:
        print("CSV not found, processing data...")

    measurements = {}
    waypoints = {}
    deviations = {}

    with open(os.path.join(data_path, "deviations.json"), "r") as f:
        deviations = json.load(f)

    try:
        with open(os.path.join(data_path, "waypoints.txt"), "r") as f:
            data = f.readlines()
            prev = [0, 0, 0, 0, 0, 0]
            for i in range(len(data)):
                data[i] = data[i].split(",")
                data[i] = [float(x.strip()) for x in data[i]]
                waypoints[int(data[i][0])] = np.array(data[i][1:]) - np.array(prev).tolist()
                prev = data[i][1:]

        for wp in os.listdir(data_path):
            cur_path = os.path.join(data_path, wp)
            if os.path.isdir(cur_path):
                measurements[int(wp)] = dict()
                with open(os.path.join(data_path, wp, "measurements.txt"), "r") as f:
                    meas = f.readlines()
                for i in range(len(meas)):
                    meas[i] = meas[i].split(",")
                    meas[i] = [x.strip() for x in meas[i]]
                    landmark = int(meas[i][0].split(":")[0].split(" ")[1])
                    # Maybe can use the no. of values for confidence of sth
                    # n_vals = int(meas[i][0].split(':')[0].split(' ')[3])
                    x_meas = float(meas[i][0].split(":")[2]) * 10
                    y_meas = float(meas[i][1].split(":")[1]) * 10
                    z_meas = float(meas[i][2].split(":")[1]) * 10
                    measurements[int(wp)].update({landmark: [x_meas, y_meas, z_meas]})

        if show_data:
            for k, v in waypoints.items():
                print("\nwaypoint:", k, ":", v)
                print("\nmeasurements:")
                for l, d in measurements[k].items():
                    print("{}: {}".format(l, d))
            print("\ndeviations:", deviations)
        return waypoints, measurements, deviations
    except Exception as e:
        print("\nNo waypoints and measurements found because : ", e)
        if show_data:
            print("\ndeviations:", deviations)
        return deviations


def vec2rotmat(angle, axis, point=None):
    """Return matrix to rotate about axis defined by point and axis."""
    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = Tools.__unit_vector(axis[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    M = np.identity(3)
    M[:3, :3] = R
    if point is not None:
        M = np.identity(4)
        M[:3, :3] = R
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M
