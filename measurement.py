import cv2
from pyzbar import pyzbar
import pyrealsense2 as rs
import numpy as np


take_measurement_flag = True  # make this flag True when the arm reaches the required waypoint
landmark_measurements = []


def get_qr_center(frame):
    barcodes = pyzbar.decode(frame)
    centerOfQRs = []
    landmark_id = "NIL"
    for barcode in barcodes:
        landmark_id = barcode.data.decode()
        x, y, w, h = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = (int(x + w / 2), int(y + h / 2))
        centerOfQRs.append([landmark_id, center[0], center[1]])

    return frame, centerOfQRs


def landmark_measurement(image_coords, depth_image, intr):
    width = intr.width
    h = intr.height
    ppx = intr.ppx
    ppy = intr.ppy
    fx = intr.fx
    fy = intr.fy
    landmark_id = image_coords[0]
    [ix, iy] = image_coords[1:]
    Zc = depth_image.get_distance(ix, iy)

    # homography
    P_transf = np.vstack(([Zc / (fx), 0, -Zc * ppx / fx], [0, Zc / fy, -Zc * ppy / fy], [0, 0, Zc]))
    I = np.transpose([ix, iy, 1])
    X = P_transf @ I  # transform x image_coordinates=camera coordinates
    X[0] = round(X[0] * 100, 2)
    X[1] = round(X[1] * 100, 2)
    X[2] = round(X[2] * 100, 2)

    return {landmark_id: [X[0], X[1], X[2]]}


def setup_realsense():
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 3)  # Low ambient light

    return pipe, profile


def get_measurement(pipe, profile):
    landmark_measurements = dict()
    # TODO: Run measurement for some iterations and average the results
    while take_measurement_flag:
        frameset = pipe.wait_for_frames()

        color_frame = frameset.get_color_frame()
        depth = frameset.get_depth_frame()

        color = np.asanyarray(color_frame.get_data())

        frame, centerOfQRs = get_qr_center(color)
        prof = profile.get_stream(rs.stream.depth)
        intr = prof.as_video_stream_profile().get_intrinsics()
        try:
            for center in centerOfQRs:
                [x, y] = center[1:]
                landmark_coordinates_current = landmark_measurement(center, depth, intr)
                landmark_measurements.update(landmark_coordinates_current)  #! Overwrites if landmark_id already exists
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, "landmark = " + str(landmark_coordinates_current), (x, y), font, 0.5, (0, 0, 0), 2)
            print(landmark_measurements)
        except:
            (x, y) = (0, 0)
        # image = cv2.circle(frame, center, radius=1, color=(0, 0, 255), thickness=5)
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, 'center='+str(center), (x + 6, y - 6), font, 1.0, (0,0,0), 1)
        cv2.imshow("Center of the QR Code", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # set take_measurement_flag to False here
    cv2.destroyAllWindows()
    return landmark_measurements
