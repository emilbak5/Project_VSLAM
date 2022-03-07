import os
import numpy as np
import cv2

import pykitti
import sys
import os

sys.path.insert(1, os.getcwd()) #Use this to get the lib module to work (gets current working dir)

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm




class VisualOdometry():
    def __init__(self, dataset: pykitti.odometry):

        self.dataset = dataset
        self.cam0_K = self.dataset.calib.K_cam0
        self.cam0_P =  self.dataset.calib.P_rect_00

        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)


    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        # This function should detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
        # The descriptors should then be matched using the class flann object (knnMatch with k=2)
        # Remove the matches not satisfying Lowe's ratio test
        # Return a list of the good matches for each image, sorted such that the n'th descriptor in image i matches the n'th descriptor in image i-1
        # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

        img1 = np.array(self.dataset.get_cam0(i - 1))
        img2 = np.array(self.dataset.get_cam0(i))


        # compute the descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        matches = self.flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        kp1_list = np.float32([kp1[m.queryIdx].pt for m in good])
        kp2_list = np.float32([kp2[m.trainIdx].pt for m in good])

        return kp1_list, kp2_list

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Estimate the Essential matrix using built in OpenCV function
        # Use decomp_essential_mat to decompose the Essential matrix into R and t
        # Use the provided function to convert R and t to a transformation matrix T

        points1 = np.array(q1)
        points2 = np.array(q2)
        E, _ = cv2.findEssentialMat(points1, points2, cameraMatrix=self.cam0_K)

        r1, r2, t = cv2.decomposeEssentialMat(E)

        T1 = self._form_transf(r1, t.ravel())
        T2 = self._form_transf(r1, -t.ravel())
        T3 = self._form_transf(r2, t.ravel())
        T4 = self._form_transf(r2, -t.ravel())

        Ts = [T1, T2, T3, T4]

        max_count = 0
        T_highest_count = None
        P = self.cam0_P
        for T in Ts:
            Q = []
            Q_2 = []
            for i in range(int(points1.size/2)):
                Q.append(cv2.triangulatePoints(P, P@T, points1[i], points2[i]))
                Q[i] = Q[i] / Q[i][3]
                Q_2.append(T @ Q[i])
            only_pos_Q = [num for num in Q if num[2] >= 0]
            only_pos_Q2 = [num for num in Q_2 if num[2] >= 0]
            pos_count = len(only_pos_Q) + len(only_pos_Q2)
            if pos_count > max_count:
                max_count = pos_count
                T_highest_count = T

        return T_highest_count

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z
        pass


def visual_odemetry_mono():

    basedir = './data'
    sequence = '00'
    # data_dir = './data/sequences/00'  # Try KITTI_sequence_2 too

    frames = range(0, 50, 1) #Indicate how many frames to use
    dataset = pykitti.odometry(basedir, sequence, frames=frames)
    
    poses = dataset.poses

    vo = VisualOdometry(dataset)
    #play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))

        
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))


    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry",
                             file_out=os.path.basename(basedir) + sequence + ".html")

