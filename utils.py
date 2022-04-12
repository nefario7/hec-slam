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
