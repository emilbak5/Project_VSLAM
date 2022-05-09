import cv2
from scipy.optimize import least_squares

from itertools import compress

from src.Graphwrapper import *

import matplotlib.pyplot as plt

import random


# def stereo_vo(kp, desc, dataset, graph: graphstructure, idx, prev_idx):

class VisualOdometry():
    def __init__(self, dataset: pykitti.odometry):
        # self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        # self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        # self.images_l = self._load_images(os.path.join(data_dir, 'image_l'))
        # self.images_r = self._load_images(os.path.join(data_dir, 'image_r'))
        self.dataset = dataset

        self.K_l = self.dataset.calib.K_cam0
        self.P_l =  self.dataset.calib.P_rect_00
        self.K_r = self.dataset.calib.K_cam1
        self.P_r =  self.dataset.calib.P_rect_10
        

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [np.divide(self.disparity.compute(np.array(self.dataset.get_cam0(0)), np.array(self.dataset.get_cam1(0))).astype(np.float32), 16)]

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        # create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.orb = cv2.ORB_create(nfeatures=100)

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, img1, img2, kp1, kp2, des1, des2):
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

        # Match descriptors.
        matches = self.bf.match(des1,des2)

        ### VISUALISATION ### 
        # # Sort them in the order of their distance.
        # matches = sorted(matches, key = lambda x:x.distance)
        # # Draw first 10 matches.
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show()
        # #cv2.waitKey(0)

        kp_matches = np.float32([ [kp1[m.queryIdx].pt, kp2[m.trainIdx].pt] for m in matches[:50] ])

        return kp_matches
    
    def find_essential_mat(self,img1, img2, K_camera_matrix, matches):
        kp1 = matches[:,0]
        kp2 = matches[:,1]

        confidence = 0.99
        ransacReprojecThreshold = 1
        essential_matrix, mask = cv2.findEssentialMat(
                kp1,
                kp2, 
                K_camera_matrix, 
                cv2.RANSAC, 
                confidence,
                ransacReprojecThreshold,
                maxIters = 100000)
        mask = mask.ravel()

        inlier_matches = np.array([match for index, match in enumerate(matches) if mask[index]])

        ### VISUALISATION ###
        # inlier_kp1 = inlier_matches[:,0]
        # inlier_kp2 = inlier_matches[:,1]

        # n_arr = []
        # for i in range(len(inlier_kp1)):
        #     random_color=list(np.random.choice(range(255),size=3))
        #     n_arr.append(random_color)
        # n_arr = np.array(n_arr)

        # i = 0
        # image = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
        # for kp in inlier_kp1:
        #     color = (int(n_arr[i,0]), int(n_arr[i,1]), int(n_arr[i,2]))
        #     image = cv2.circle(image, (int(kp[0]),int(kp[1])), radius=4, color=color, thickness=2)
        #     i += 1
        # cv2.imshow("Image 1", image)

        # i = 0
        # image = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
        # for kp in inlier_kp2:
        #     color = (int(n_arr[i,0]), int(n_arr[i,1]), int(n_arr[i,2]))
        #     image = cv2.circle(image, (int(kp[0]),int(kp[1])), radius=4, color=color, thickness=2)
        #     i += 1
        # cv2.imshow("Image 2", image)

        # cv2.waitKey(0)

        return essential_matrix, inlier_matches     

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def track_keypoints(self, img1, img2, trackpoints1_, max_error=4):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """

        trackpoints1 = np.expand_dims(trackpoints1_, axis=1) 

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Triangulate points from i-1'th image

        enough_points = True

        if len(q1_r) > 0:
            Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
            # Un-homogenize
            Q1 = np.transpose(Q1[:3] / Q1[3])

            # Triangulate points from i'th image
            Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
            # Un-homogenize
            Q2 = np.transpose(Q2[:3] / Q2[3])
            return Q1, Q2, enough_points
        else: 
            enough_points = False
            return None, None, False


    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self, kp2, desc2, dataset, graph: graphstructure, current_img_idx, prev_idx):

        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Get the i-1'th image and i'th image

        #img1_l, img2_l = self.images_l[i - 1:i + 1]

        img1_l = np.array(self.dataset.get_cam0(prev_idx))
        img2_l = np.array(self.dataset.get_cam0(current_img_idx))

        vertex_0 = graph.g.vertex(len(graph.g.get_vertices()) - 1)
        kp1 = graph.v_keypoints[vertex_0]
        kp1 = cv2.KeyPoint_convert(kp1)
        
        kp1, desc1 = self.orb.compute(img1_l, kp1)

        matches = self.get_matches(img1_l, img2_l, kp1, kp2, desc1, desc2)
        _, inlier_matches = self.find_essential_mat(img1_l, img2_l, self.K_l, matches)
        kp1 = inlier_matches[:,0]
        kp2 = inlier_matches[:,1]
        
        # Calculate the disparitie
        self.disparities.append(np.divide(self.disparity.compute(img2_l, np.array(self.dataset.get_cam1(current_img_idx))).astype(np.float32), 16))

        # Calculate the right keypoints

        idx = len(graph.g.get_vertices())
        kp1_l, kp1_r, kp2_l, kp2_r = self.calculate_right_qs(kp1_l, kp2_l, self.disparities[idx - 1], self.disparities[idx])

        # Calculate the 3D points
        Q1, Q2, enough_points = self.calc_3d(kp1_l, kp1_r, kp2_l, kp2_r)
        if enough_points:
        # Estimate the transformation matrix
            transformation_matrix = self.estimate_pose(kp1_l, kp2_l, Q1, Q2)

            return transformation_matrix, inlier_matches


