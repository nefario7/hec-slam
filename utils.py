import numpy as np


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


def display_data(self, predictions, updates, measurement):
    """Prints data

    Args:
        predictions (_type_):
        updates (_type_):
        measurement (_type_):

    Returns:
        _type_: None
    """
    print("-" * 50)
    print("Waypoint: {}".format(self.waypoint))
    print("\nPredictions:")
    print("X : {} and P : {}".format(predictions[0], predictions[1]))
    print("\nUpdates:")
    print("X : {} and P : {}".format(updates[0], updates[1]))

    print("\nMeasurement:")
    print(measurement)
