import os
import sys
#from sympy import true

from yaml import parse
#sys.path.insert(1, os.path.abspath(""))
sys.path.insert(1, os.getcwd()) #Use this to get the lib module to work (gets current working dir)

import bz2

import numpy as np
import time
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
import matplotlib.pyplot as plt
import pykitti
import random


from lib.visualization.plotting import plot_residual_results, plot_sparsity

def test_keypoints_out_frame(kp1, kp2, image1, image2):
    center_coordinates1 = (int(kp1[0]), int(kp1[1]))
    center_coordinates2 = (int(kp2[0]), int(kp2[1]))
    # Radius of circle
    radius = 5

    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    image1 = cv2.circle(image1, center_coordinates1, radius, color, thickness)
    image2 = cv2.circle(image2, center_coordinates2, radius, color, thickness)

    # Displaying the image
    window_name1 = "Image 1"
    window_name2 = "Image 2"
    cv2.imshow(window_name1, image1)
    cv2.imshow(window_name2, image2)
    cv2.waitKey(0)

def inverse_T(T):
    T[:3,:3] = np.transpose(T[:3,:3])
    T[:3, 3] = -np.matmul(T[:3,:3], T[:3, 3])
    return T

def reindex(idxs):
    keys = np.sort(np.unique(idxs))
    key_dict = {key: value for key, value in zip(keys, range(keys.shape[0]))}
    return [key_dict[idx] for idx in idxs]

def rotate(Qs, rot_vecs):
    """
    Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.

    Parameters
    ----------
    Qs (ndarray): The 3D points
    rot_vecs (ndarray): The rotation vectors

    Returns
    -------
    Qs_rot (ndarray): The rotated 3D points
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(Qs * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * Qs + sin_theta * np.cross(v, Qs) + dot * (1 - cos_theta) * v


def project(Qs, cam_params):
    """
    Convert 3-D points to 2-D by projecting onto images.

    Parameters
    ----------
    Qs (ndarray): The 3D points
    cam_params (ndarray): Initial parameters for cameras

    Returns
    -------
    qs_proj (ndarray): The projectet 2D points
    """
    # Rotate the points
    qs_proj = rotate(Qs, cam_params[:, :3])
    # Translat the points
    qs_proj += cam_params[:, 3:6]
    # Un-homogenized the points
    qs_proj = -qs_proj[:, :2] / qs_proj[:, 2, np.newaxis]
    # Distortion
    f, k1, k2 = cam_params[:, 6:].T
    n = np.sum(qs_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    qs_proj *= (r * f)[:, np.newaxis]
    return qs_proj


def objective(params, n_cams, n_Qs, cam_idxs, Q_idxs, qs):
    """
    The objective function for the bundle adjustment

    Parameters
    ----------
    params (ndarray): Camera parameters and 3-D coordinates.
    n_cams (int): Number of cameras
    n_Qs (int): Number of points
    cam_idxs (list): Indices of cameras for image points
    Q_idxs (list): Indices of 3D points for image points
    qs (ndarray): The image points

    Returns
    -------
    residuals (ndarray): The residuals
    """
    # Should return the residuals consisting of the difference between the observations qs and the reporjected points
    # Params is passed from bundle_adjustment() and contains the camera parameters and 3D points
    # project() expects an arrays of shape (len(qs), 3) indexed using Q_idxs and (len(qs), 9) indexed using cam_idxs
    # Copy the elements of the camera parameters and 3D points based on cam_idxs and Q_idxs

    # Get the camera extrinsic parameters
    cam_extrinsics_rodrigues = params[:n_cams * 6].reshape((n_cams, 6))
    #Define camera intrinsics (No distortion params)
    K = np.array([[718.856,0,607.1928],[0,718.856,185.2157],[0,0,1]])

    # Get the 3D points
    Qs = params[n_cams * 6:].reshape((n_Qs, 3))

    Qs_points = Qs[Q_idxs]
    
    qs_proj = np.empty((Q_idxs.size, 2))
    for i in range(Q_idxs.size):
        R_vec = params[cam_idxs[i]*6:cam_idxs[i]*6 + 3]
        t_vec = params[cam_idxs[i]*6 + 3:cam_idxs[i]*6 + 6]
        cam = cam_idxs[i]
        Q = Qs_points[i]
        pt, _ = cv2.projectPoints(Q, R_vec, t_vec, K, np.empty(0))
        qs_proj[i] = pt
        if pt[0][0][0] > 1241 or pt[0][0][1] > 376:
            #print("Error")
            pass

    # Calculate the residuals
    residuals = (qs_proj - qs).ravel()
    #print("sumres: ", np.mean(residuals, axis=0))
    
    return residuals

def sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs):
    """
    Create the sparsity matrix

    Parameters
    ----------
    n_cams (int): Number of cameras
    n_Qs (int): Number of points
    cam_idxs (list): Indices of cameras for image points
    Q_idxs (list): Indices of 3D points for image points

    Returns
    -------
    sparse_mat (ndarray): The sparsity matrix
    """
    m = cam_idxs.size * 2  # number of residuals
    n = n_cams * 6 + n_Qs * 3  # number of parameters
    sparse_mat = lil_matrix((m, n), dtype=int)
    # Fill the sparse matrix with 1 at the locations where the parameters affects the residuals

    i = np.arange(cam_idxs.size)
    # Sparsity from camera parameters
    for s in range(6):
        sparse_mat[2 * i, cam_idxs * 6 + s] = 1
        sparse_mat[2 * i + 1, cam_idxs * 6 + s] = 1

    # Sparsity from 3D points
    for s in range(3):
        sparse_mat[2 * i, n_cams * 6 + Q_idxs * 3 + s] = 1
        sparse_mat[2 * i + 1, n_cams * 6 + Q_idxs * 3 + s] = 1

    return sparse_mat

def bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs, qs, sparse_mat):
    """
    Preforms bundle adjustment with sparsity

    Parameters
    ----------
    cam_params (ndarray): Initial parameters for cameras
    Qs (ndarray): The 3D points
    cam_idxs (list): Indices of cameras for image points
    Q_idxs (list): Indices of 3D points for image points
    qs (ndarray): The image points
    sparse_mat (ndarray): The sparsity matrix

    Returns
    -------
    residual_init (ndarray): Initial residuals
    residuals_solu (ndarray): Residuals at the solution
    solu (ndarray): Solution
    """

    # Stack the camera parameters and the 3D points
    params = np.hstack((cam_params.ravel(), Qs.ravel()))

    # Save the initial residuals
    residual_init = objective(params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)

    # Perform the least_squares optimization with sparsity
    res = least_squares(objective, params, jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-2, method='trf',
                        args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))

    # Get the residuals at the solution and the solution
    residuals_solu = res.fun
    solu = res.x
    return residual_init, residuals_solu, solu

def bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs):
    # Create Sparsity Matrix
    n_cams = cam_params.shape[0]
    n_Qs = Qs.shape[0]
    sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
    start = time.process_time()
    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs,
                                                                                    qs, sparse_mat)
    print("Bundle Adjustment took: ", time.process_time() - start, " Seconds")


def hamming_dist(x, y):
    d = np.count_nonzero(x != y)
    return d



# Scuffed outlier detection avoiding use of matrix operations (NOT OPTIMIZED)
# Note: Should not be nessecary to use this method as hamming_threshold should be chosen so outliers are not occuring
# Note: Only works if images are not too far away from each other and taken in a situation like driving around where the camera is placed in one position on the car
# img_shift should be chosen as ex. width of image 2
# Takes 2 arrays of point correspondences from 2 images and removes the outliers
def fast_outlier_detection(pts1, pts2, img_shift, std_factor = 1):
    slopes = []
    for i in range(pts1):
        dx = (pts2[0] + img_shift) - pts1[0]
        dy = (pts2[1] + img_shift) - pts1[1]
        a = np.sqrt(dx^2 + dy^2) # Pythagoras to find slope
        slopes.append(a)
    std = np.std(np.array(slopes), ddof=1)
    mu = sum(slopes) / len(slopes)
    #mu = np.median(np.array(slopes)) # using median instead of mean because there are only few points and one outlier can fuck up the median
    # Nah the statistics still works with regular mean
    # Calculate d to be the distance between the mean slope and the current index slope
    pts1_new = []
    pts2_new = []
    for i in range(slopes):
        d = abs(mu - slopes[i])
        if d > (mu + std_factor * std) or d < (mu - std_factor * std):
            continue
        pts1_new.append(pts1[i])
        pts2_new.append(pts2[i])

    return pts1_new, pts2_new

def triangulation(kp1, kp2, T1, T2):
    # Function modified from: Project: DF-VO   Author: Huangying-Zhan   File: ops_3d.py    License: MIT License
    """Triangulation to get 3D points
    Args:
        kp1 (Nx2): keypoint in view 1 (normalized)
        kp2 (Nx2): keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (3xN): 3D coordinates of the keypoints w.r.t world coordinate
        X1 (3xN): 3D coordinates of the keypoints w.r.t view1 coordinate
        X2 (3xN): 3D coordinates of the keypoints w.r.t view2 coordinate
    """
    K = np.array([[718.856,0,607.1928],[0,718.856,185.2157],[0,0,1]], dtype=np.float64)

    P1 = np.matmul(K, T1[:3])
    P2 = np.matmul(K, T2[:3])

    Q = cv2.triangulatePoints(P1, P2, kp1, kp2)
    Q = Q[:4]/Q[3]

    return Q[:3]


# Pre processing for bundle adjustment, takes list of keypoints and descriptors
# kpdes = [[kp1, des1], [kp2, des2], [kp3, des3]] corresponding to camera 0, 1, 2.... n
# cam_extrinsics = list of initial estimated camera matrices for each camera. Needs to have same length of kpdes
# hamming_threshold = number describing how matching the descriptors should be before accepted, the lower the more picky and less (but strong) features will be accepted
# search_window = Can specify the search window if one wants to limit process time since it can theoretically be ex: 300 images before amount of found correspondences is lower then min_correspondences
#                 this will cause enormous matrix in bundle adjustment which is impractical for real time Bundle Adjustment, one could limit this to having a maximum overlap of 5 images 
#                 if minc_correspondences = 0 and search window = len(kpdes) then every combination of features are searched (aka complete search), this is fast for less than 10 images but time complexity of approx 500^N where N is the length of kpdes aka amount of images (or cameras)
# window_shift = variable that limits the search space and speeds up the process, instead of searching 1st image (image [0]) and matching it with all next, it only matches it with 2,4,6... images (if this value is = 2) (Should not be used if there are a few descriptors ex: 7 in our SLAM example)
# window_shift_jump = variable that makes this function faster so if we need to search 7 images, then we match 1st image (image [0]) with 1,2,3,4,5,6 but if this variable is 2 then we match the 1st image with 2,4,6 image reducing overhead. This value must be less than search_window
def BA_pre_processing_cam_n_points(kpdes, cam_extrinsics, hamming_threshold = 25, search_window = 7, window_shift = 1, window_shift_jump = 1):
    if window_shift >= search_window:
        print("ERROR IN BA: window_shift must be less than search_window")
    n_cams = len(kpdes)     # Number of Cameras
    n_Qs = 0                # Number of 3D points (aka features seen in more than 1 camera), the same feature seen in x different cameras will be counted as 1 n_Qs
    n_qs = 0                # Number total 2D points seen in all cameras, if x different cameras see the same feature it will be couted as 4 n_qs

    # Bottom arrays needs to be dynamically appended
    cam_idxs = np.empty(0, dtype=int)   
    Q_idxs = np.empty(0, dtype=int)     
    qs = np.empty((0, 2))               # Used for initial guesses
    Qs = np.empty((0, 3))               # Used for initial guesses

    # These for loops will go through all descriptor vectors and match the first descroptor with all descriptors of the next images, then do the same with the second descriptor in the first and so on, and after that move on to the next image and do the same to the images thereafter
    for mov_win in range(0, n_cams - search_window + 1, 5):
        for des_x in range(mov_win, mov_win + search_window - 1, window_shift):                      
            for i in range(len(kpdes[des_x][1])):
                n_qs_tmp = 0
                camidx_x = None
                camidx_y = None
                kp1 = None
                kp2 = None     
                for des_y in range(des_x + 1, mov_win + search_window, window_shift_jump):
                    for j in range(len(kpdes[des_y][1])):                           
                        d = hamming_dist(kpdes[des_x][1][i], kpdes[des_y][1][j])
                        if d < hamming_threshold: # Correspondence Found!
                            #print(d)
                            if n_qs_tmp == 0:
                                cam_idxs = np.append(cam_idxs, des_x)
                                camidx_x = des_x
                                Q_idxs = np.append(Q_idxs, n_Qs)
                                kp1 = np.asarray(kpdes[des_x][0][i].pt)
                                qs = np.append(qs, kp1)

                                cam_idxs = np.append(cam_idxs, des_y)
                                camidx_y = des_y
                                Q_idxs = np.append(Q_idxs, n_Qs)
                                kp2 = np.asarray(kpdes[des_y][0][j].pt)
                                qs = np.append(qs, kp2)
                                n_qs_tmp += 2
                            else:
                                cam_idxs = np.append(cam_idxs, des_y)
                                camidx_y = des_y
                                Q_idxs = np.append(Q_idxs, n_Qs)
                                kp2 = np.asarray(kpdes[des_y][0][j].pt)
                                qs = np.append(qs, np.asarray(kpdes[des_y][0][j].pt))
                                n_qs_tmp += 1

                                #Visualize for debugging
                                # img1 = cv2.imread("data/sequences/00/image_0/00000" + str(des_x) + ".png")
                                # img2 = cv2.imread("data/sequences/00/image_0/00000" + str(des_y) + ".png")
                                # img1 = cv2.circle(img1, (int(kpdes[des_x][0][i].pt[0]), int(kpdes[des_x][0][i].pt[1])), 5, (255,0,0), 3)
                                # img2 = cv2.circle(img2, (int(kpdes[des_y][0][j].pt[0]), int(kpdes[des_y][0][j].pt[1])), 5, (255,0,0), 3)
                                # print((int(kpdes[des_x][0][i].pt[0]), int(kpdes[des_x][0][i].pt[1])))
                                # print((int(kpdes[des_y][0][j].pt[0]), int(kpdes[des_y][0][j].pt[1])))
                                # cv2.imshow("img1", img1)
                                # cv2.imshow("img2", img2)
                                # cv2.waitKey(0)
                                
                            break   # No need to search any further since we assume no more descriptors match that same query descriptor
                if n_qs_tmp != 0:
                    # Triangulate 2 of the points and add it to Q_idxs list
                    cam_ex1 = cam_extrinsics[camidx_x]
                    cam_ex2 = cam_extrinsics[camidx_y]
                    Q = triangulation(kp1, kp2, cam_extrinsics[camidx_x], cam_extrinsics[camidx_y])
                    if (Q[0] > -4.81 and Q[0] < -4.79) or (Q[2] > 68 and Q[2] < 69):
                        if camidx_x ==75 or camidx_y == 75:
                            print("WAIT!")
                            #-4.8
                            #-2.31
                            #-68.57
                            pass
                    Qs = np.append(Qs, Q)

                    n_qs += n_qs_tmp
                    n_Qs += 1 # As the last thing, increment
        n_qs = np.size(qs, 0)
        #n_Qs = np.size(Q_idxs, 0)

    """
    Returns
    -------
    cam_params (ndarray): Shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a translation vector, then a focal distance and two distortion parameters.
    Qs (ndarray): Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
    cam_idxs (ndarray): Shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
    Q_idxs (ndarray): Shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
    qs (ndarray): Shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.
    """
    Qs = Qs.flatten()
    cam_idxs = cam_idxs.flatten()
    Q_idxs = Q_idxs.flatten()
    qs = qs.flatten()

    Qs = Qs.reshape(int(Qs.size/3), 3)
    qs = qs.reshape(int(qs.size/2), 2)
    return Qs, cam_idxs, Q_idxs, qs


def main():
    print("Starting bundle adjustment...")
    start_BA = time.process_time()

    #Get the camera and image data
    basedir = './data'
    sequence = '00'

    n_frames = 100
    frames = range(0, n_frames, 1) #Indicate how many frames to use
    dataset = pykitti.odometry(basedir, sequence, frames=frames)#, frames=frames)
    
    cam_K = dataset.calib.K_cam0    
    cam_poses = dataset.poses

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=100) # nFeatures is a very important number because the lower it is, the faster the computation time
    # find the keypoints and descriptors with ORB

    #Setup keypoint and extrinsic camera list
    frames_kp_and_des = []
    cam_extrinsics_rodrigues = []
    cam_extrinsics = []
    start = time.process_time()

    for i in range(n_frames):
        #Get camera extrinisc parameters
        cam_pose = cam_poses[i].copy()
        cam_extrinsic = inverse_T(cam_pose)

        R, _ = np.asarray(cv2.Rodrigues(cam_extrinsic[:3, :3]), dtype=object)
        t = cam_extrinsic[:3, 3].flatten()

        cam_extrinsics_rodrigues.append([R[0][0], R[1][0], R[2][0], t[0], t[1], t[2]])
        cam_extrinsics.append(cam_extrinsic)

        #Get keypoints and descriptors
        frame = np.array(dataset.get_cam0(i))
        kp_tmp, des_tmp = orb.detectAndCompute(frame, None)
        frames_kp_and_des.append([kp_tmp, des_tmp])


    print("DetectAndComputes plus computation of camera extrinsics took: ", time.process_time() - start, " Seconds")

    #Genearate between keypoints and triangulated points along with cameras used for triangulation
    print("Bundle adjustment preprocessing...")
    start_BA_preprocess = time.process_time()
    Qs, cam_idxs, Q_idxs, qs = BA_pre_processing_cam_n_points(frames_kp_and_des, cam_extrinsics, 20, 7, 1)
    print("Bundle adjustment preprocessing took: ", time.process_time() - start_BA_preprocess, " Seconds")

    print("cam_idxs: ", cam_idxs.shape)
    print("Q_idxs: ", Q_idxs.shape)
    print("qs: ", qs.shape)
    print("Qs: ", Qs.shape)
    print(cam_idxs)

    
    cam_extrinsics_rodrigues = np.array(cam_extrinsics_rodrigues).flatten()

    #Add noise to camera extrinsic parameters to simulate
    #imperfect Visual Odometry
    for i in range(int(len(cam_extrinsics_rodrigues)/6)):
        #cam_extrinsics_rodrigues[i*6 + 3] += round(random.uniform(-0.003, 0.003)*1, 6) # x
        #cam_params[i*6 + 4] += round(random.uniform(-0.01, 0.01)*1, 6)  # y
        #cam_extrinsics_rodrigues[i*6 + 5] += round(random.uniform(-0.5, 0.5)*1, 6)     # z
        pass

    cam_extrinsics_rodrigues = cam_extrinsics_rodrigues.reshape(int(cam_extrinsics_rodrigues.size/6), 6)
    n_cams = cam_extrinsics_rodrigues.shape[0]
    n_Qs = Qs.shape[0]
    print("n_cameras: {}".format(n_cams))
    print("n_points: {}".format(n_Qs))
    print("Total number of parameters: {}".format(6 * n_cams + 3 * n_Qs))
    print("Total number of residuals: {}".format(2 * qs.shape[0]))

    # residual_init, residual_minimized, opt_params = bundle_adjustment(cam_extrinsics_rodrigues, Qs, cam_idxs, Q_idxs, qs)
    sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
    plot_sparsity(sparse_mat)
    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_extrinsics_rodrigues, Qs, cam_idxs, Q_idxs,
                                                                                    qs, sparse_mat)
    print("Bundle Adjustment took: ", time.process_time() - start_BA, " Seconds")
    # plot_residual_results(qs, residual_init,
    #                       residual_minimized)

    x0 = np.hstack((cam_extrinsics_rodrigues.ravel(), Qs.ravel()))
    residual_init = objective(x0, n_cams, n_Qs, cam_idxs, Q_idxs, qs)


    # Plotting the results vs GT etc
    xs_gt = np.empty(0, dtype=float)  
    ys_gt = np.empty(0, dtype=float)  
    zs_gt = np.empty(0, dtype=float)
    xs_iniguess = np.empty(0, dtype=float)  
    ys_iniguess = np.empty(0, dtype=float)  
    zs_iniguess = np.empty(0, dtype=float) 
    xs_est = np.empty(0, dtype=float)  
    ys_est = np.empty(0, dtype=float)  
    zs_est = np.empty(0, dtype=float) 
    for i in range(n_cams):
        pos = poses[i][:3, 3]
        est = opt_params[3+(i*6):6+(i*6)]
        xs_gt = np.append(xs_gt, pos[0])
        ys_gt = np.append(ys_gt, pos[1])
        zs_gt = np.append(zs_gt, pos[2])
        xs_iniguess = np.append(xs_iniguess, cam_extrinsics_rodrigues[i][3])
        ys_iniguess = np.append(ys_iniguess, cam_extrinsics_rodrigues[i][4])
        zs_iniguess = np.append(zs_iniguess, cam_extrinsics_rodrigues[i][5])
        xs_est = np.append(xs_est, est[0])
        ys_est = np.append(ys_est, est[1])
        zs_est = np.append(zs_est, est[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs_gt,zs_gt)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs_iniguess,zs_iniguess)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs_est,zs_est)
    plt.show()

    plt.figure(1, figsize=(15, 5))
    plt.subplot(211)
    plt.plot(residual_init)
    plt.subplot(212)
    plt.plot(residual_minimized)
    plt.show()


if __name__ == "__main__":
    main()