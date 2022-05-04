import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyzbar import pyzbar
from scipy.sparse.linalg import lsqr

import cv2


# this class is to convert image coordinates into a point in 3D space
class Calibrator:
    def __init__(self):
        # initialize the camera outside the class the sample code is as follows:
        # pipe = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # profile = pipe.start(config)

        self.prof = profile.get_stream(rs.stream.depth)
        self.intr = self.prof.as_video_stream_profile().get_intrinsics()  # camera intrinsics

    def coords_relative_to_camera(self, coords, depth, intr):
        # this method takes image coordinates and corresponding depth image and maps it to camera coordinate system
        [ix, iy] = coords[1:]
        iz = depth.get_distance(ix, iy)  # get depth value the RGB and depth image is to be aligned
        point = rs.rs2_deproject_pixel_to_point(intr, [ix, iy], iz)
        [Cx, Cy, Cz] = point  # convert to centimeters
        Cx = round(Cx * 100, 2)
        Cy = round(Cy * 100, 2)
        Cz = round(Cz * 100, 2)
        point = (coords[0], Cx, Cy, Cz)
        return point

    def landmark_measurement(self, image_coords, depth_image, intr):
        # print('image coords=',image_coords)
        width = intr.width
        h = intr.height
        ppx = intr.ppx
        ppy = intr.ppy
        fx = intr.fx
        fy = intr.fy
        # landmark_id=image_coords[0]
        [ix, iy] = image_coords[:]
        # Zc=depth_image.get_distance(ix, iy)
        Zc = 0.5422  # manually entering the distance instead of reading from the depth image, the bug needs to be resolved
        # homography
        P_transf = np.vstack(([Zc / (fx), 0, -Zc * ppx / fx], [0, Zc / fy, -Zc * ppy / fy], [0, 0, Zc]))
        I = np.transpose([ix, iy, 1])
        X = P_transf @ I  # transform x image_coordinates=camera coordinates
        X[0] = round(X[0] * 100, 2)
        X[1] = round(X[1] * 100, 2)
        X[2] = round(X[2] * 100, 2)
        measurement = [X[0], X[1], X[2]]
        return measurement

    def get_Pi_inv(self, landmark_measurements, corners, intr_mat):  # function to calculate Pi_inv as per the referred paper
        print("lm shape", landmark_measurements.shape, "corner shape", corners.shape)
        landmark_measurements = np.array(landmark_measurements, dtype=np.float32)
        landmark_measurements = landmark_measurements.reshape(4, 3)
        corners = np.array(corners, dtype=np.float32)
        corners = corners.reshape(4, 2)
        retval, rvec, tvec, _ = cv2.solvePnPRansac(landmark_measurements, corners, np.array(intr_mat), None)
        [rot, jacob] = cv2.Rodrigues(rvec)
        last_row = [0, 0, 0, 1]
        Pi_inv = np.hstack((rot, tvec))
        Pi_inv = np.vstack((Pi_inv, last_row))
        print(Pi_inv)
        return Pi_inv

    def QRcode_corners(self, frame):
        # this method is used to detect the center of the QR code in an image,

        barcodes = pyzbar.decode(frame)
        centerOfQRs = []
        landmark_id = "NIL"
        corners = []
        for barcode in barcodes:
            landmark_id = barcode.data.decode()
            x, y, w, h = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = (int(x + w / 2), int(y + h / 2))
            centerOfQRs.append([landmark_id, center[0], center[1]])

            left = barcode.rect[0]
            top = barcode.rect[1]
            width = barcode.rect[2]
            height = barcode.rect[3]
            # get the rectangular contour corner coordinates
            top_left = [top, left]
            top_right = [top, left + width]
            bottom_left = [top - height, left]
            bottom_right = [top - height, left + width]
            corners.append([landmark_id, top_left, top_right, bottom_left, bottom_right])

        return frame, centerOfQRs, corners

    def intrinsic_mat(self, intr):
        width = intr.width
        h = intr.height
        ppx = intr.ppx
        ppy = intr.ppy
        fx = intr.fx
        fy = intr.fy
        intr_mat = np.vstack(([fx, 0, ppx], [0, fy, ppy], [0, 0, 1]))

        return intr_mat

    def get_B(self, Pi_inv, Pj_inv):
        # this method calculates B ie, transform from the first camera position to the second camera position
        Pj = np.linalg.inv(Pj_inv)
        B = Pi_inv @ Pj
        return B

    def decomp_R_t(self, A):
        # function to decompose a matrix in homogenous transformation to rotation and translation matrix
        A = np.array(A)
        R = A[:3, :3]
        t = A[:3, 3]
        return R, t

    def construct_final_matrices(self, Ai, Bi, Aj, Bj):
        # function to construct the final matrices of eqn 7 in the paper
        Rai, tai = self.decomp_R_t(Ai)
        Rbi, tbi = self.decomp_R_t(Bi)
        Raj, taj = self.decomp_R_t(Aj)
        Rbj, tbj = self.decomp_R_t(Bj)

        u = np.hstack((np.identity(9) - np.kron(Rai, Rbi), np.zeros((9, 3))))
        v = np.hstack((np.kron(np.identity(3), np.transpose(tbi)), np.identity(3) - Rai))
        w = np.hstack((np.identity(9) - np.kron(Raj, Rbj), np.zeros((9, 3))))
        x = np.hstack((np.kron(np.identity(3), np.transpose(tbj)), np.identity(3) - Raj))
        M = np.concatenate((u, v, w, x), axis=0)

        tai = tai.reshape(3, 1)
        tbi = tbi.reshape(3, 1)
        N = np.vstack((np.zeros((9, 1)), tai, np.zeros((9, 1)), tbi))
        return M, N

    def solve_for_x(self, M, N):
        # solves for x in equation 7, uses scipy library's least square method to solve
        x = lsqr(M, N)
        return x

    def choose_points(self, image):
        # this method is chiefly to trouble shoot, this method allows for points can be picked manually
        # on images and returns the list of image coordinates of the points chosen
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(image)
        text = ax.text(0, 0, "", va="bottom", ha="left")
        coords = []

        def onclick(event):
            tx = "button=%d, x=%d, y=%d, xdata=%f, ydata=%f" % (event.button, event.x, event.y, event.xdata, event.ydata)
            text.set_text(tx)
            coords.append([event.x, event.y])

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        return coords
