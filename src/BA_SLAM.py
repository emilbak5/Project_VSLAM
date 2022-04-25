import os
import sys
from sympy import true

from yaml import parse
#sys.path.insert(1, os.path.abspath(""))
sys.path.append(r"C:\Users\Ulric\OneDrive - Syddansk Universitet\8. Semester\Advanced Computer Vision\Exercises\ProjectStructure\AdvancedComputerVisionExercises")

import bz2

import numpy as np
import time
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
import matplotlib.pyplot as plt
import pykitti

from lib.visualization.plotting import plot_residual_results, plot_sparsity


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

    # Get the camera parameters
    cam_params = params[:n_cams * 9].reshape((n_cams, 9))

    # Get the 3D points
    Qs = params[n_cams * 9:].reshape((n_Qs, 3))

    # Project the 3D points into the image planes
    qs_proj = project(Qs[Q_idxs], cam_params[cam_idxs])
    #print("proj", qs_proj[0])
    #print("qs: ", qs[0])


    # Calculate the residuals
    residuals = (qs_proj - qs).ravel()
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
    n = n_cams * 9 + n_Qs * 3  # number of parameters
    sparse_mat = lil_matrix((m, n), dtype=int)
    # Fill the sparse matrix with 1 at the locations where the parameters affects the residuals

    i = np.arange(cam_idxs.size)
    # Sparsity from camera parameters
    for s in range(9):
        sparse_mat[2 * i, cam_idxs * 9 + s] = 1
        sparse_mat[2 * i + 1, cam_idxs * 9 + s] = 1

    # Sparsity from 3D points
    for s in range(3):
        sparse_mat[2 * i, n_cams * 9 + Q_idxs * 3 + s] = 1
        sparse_mat[2 * i + 1, n_cams * 9 + Q_idxs * 3 + s] = 1

    return sparse_mat

def bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs, qs, sparse_mat, f_tol = 1e-1):
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
    res = least_squares(objective, params, jac_sparsity=sparse_mat, verbose=0, x_scale='jac', ftol=f_tol, method='trf',
                        args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))

    # Get the residuals at the solution and the solution
    residuals_solu = res.fun
    solu = res.x
    return residual_init, residuals_solu, solu

def bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs, ftol = 1e-1):
    # Create Sparsity Matrix
    n_cams = cam_params.shape[0]
    n_Qs = Qs.shape[0]
    sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
    start = time.process_time()
    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs,
                                                                                    qs, sparse_mat, f_tol = ftol)
    print("Bundle Adjustment took: ", time.process_time() - start, " Seconds")


def hamming_dist(x, y):
    d = np.count_nonzero(x != y)
    return d



# Scuffed outlier detection avoiding use of matrix operations (NOT OPTIMIZED)
# Note: Should not be nessecary to use this method as hamming_threshold should be chosen so outliers are not occuring
# Note: Only works if images are not too far away from each other and taken in a situation like driving around where the camera is placed in one position on the car
# img_shift should be chosen as ex. width of image 2
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

def triangulation(kp1, kp2, T_1w, T_2w):
    # Function taken from: Project: DF-VO   Author: Huangying-Zhan   File: ops_3d.py    License: MIT License
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
    kp1_3D = np.ones((3, 1))
    kp2_3D = np.ones((3, 1))
    kp1_3D[0], kp1_3D[1] = kp1[0], kp1[1]
    kp2_3D[0], kp2_3D[1] = kp2[0], kp2[1]
    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    X /= X[3]
    #X = np.transpose(X[:3] / X[3])
    #np.transpose(tri_result[:3]/tri_result[3])
    #X1 = T_1w[:3] @ X
    #X2 = T_2w[:3] @ X
    return X[:3]


# Pre processing for bundle adjustment, takes list of keypoints and descriptors
# kpdes = [[kp1, des1], [kp2, des2], [kp3, des3]] corresponding to camera 0, 1, 2.... n
# cam_matrices = list of initial estimated camera matrices for each camera. Needs to have same length of kpdes
# hamming_threshold = number describing how matching the descriptors should be before accepted, the lower the more picky and less (but strong) features will be accepted
# search_window = Can specify the search window if one wants to limit process time since it can theoretically be ex: 300 images before amount of found correspondences is lower then min_correspondences
#                 this will cause enormous matrix in bundle adjustment which is impractical for real time Bundle Adjustment, one could limit this to having a maximum overlap of 5 images 
#                 if minc_correspondences = 0 and search window = len(kpdes) then every combination of features are searched (aka complete search), this is fast for less than 10 images but time complexity of approx 500^N where N is the length of kpdes aka amount of images (or cameras)
# window_shift = variable that limits the search space and speeds up the process, instead of searching 1st image (image [0]) and matching it with all next, it only matches it with 2,4,6... images (if this value is = 2) (Should not be used if there are a few descriptors ex: 7 in our SLAM example)
# window_shift_jump = variable that makes this function faster so if we need to search 7 images, then we match 1st image (image [0]) with 1,2,3,4,5,6 but if this variable is 2 then we match the 1st image with 2,4,6 image reducing overhead. This value must be less than search_window
def BA_pre_processing_cam_n_points(kpdes, cam_matrices, hamming_threshold = 25, search_window = 7, window_shift = 1, window_shift_jump = 1):
    start = time.process_time()
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
    for des_x in range(0, search_window - 1, window_shift):                      
        for i in range(len(kpdes[des_x][1])):
            n_qs_tmp = 0
            camidx_x = None
            camidx_y = None
            kp_x = None
            kp_y = None     
            for des_y in range(des_x + 1, search_window, window_shift_jump):
                for j in range(len(kpdes[des_y][1])):                           
                    d = hamming_dist(kpdes[des_x][1][i], kpdes[des_y][1][j])
                    if d < hamming_threshold: # Correspondence Found!
                        #print(d)
                        if n_qs_tmp == 0:
                            cam_idxs = np.append(cam_idxs, des_x)
                            camidx_x = des_x
                            Q_idxs = np.append(Q_idxs, n_Qs)
                            kp_x = np.asarray(kpdes[des_x][0][i].pt)
                            qs = np.append(qs, kp_x)

                            cam_idxs = np.append(cam_idxs, des_y)
                            camidx_y = des_y
                            Q_idxs = np.append(Q_idxs, n_Qs)
                            kp_y = np.asarray(kpdes[des_y][0][j].pt)
                            qs = np.append(qs, kp_y)
                            n_qs_tmp += 2
                        else:
                            cam_idxs = np.append(cam_idxs, des_y)
                            Q_idxs = np.append(Q_idxs, n_Qs)
                            qs = np.append(qs, np.asarray(kpdes[des_y][0][j].pt))
                            n_qs_tmp += 1
                        break   # No need to search any further since we assume no more descriptors match that same query descriptor
            if n_qs_tmp != 0:
                # Triangulate 2 of the points and add it to Q_idxs list
                point3D = triangulation(kp_x, kp_y, cam_matrices[camidx_x], cam_matrices[camidx_y])
                Qs = np.append(Qs, point3D)
                #print("point3D: ", point3D)
                #print(Qs[n_Qs])
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

    print("Bundle Adjustment Pre Processing took: ", time.process_time() - start, " Seconds")
    return Qs, cam_idxs, Q_idxs, qs


def SLAM_bundle_adjustment(img_kp_and_des, cam_transformations, fk1k2 = [-5.14862520e+00, -1.14997375e-06, -6.67297786e-13], hamming_threshold = 25, search_window = 7, window_shift = 1, window_shift_jump = 1, ftol= 1e-1):
    """
    Performs bundle adjustment to optimize camera transformations (poses) and 3D points using keypoints and descriptors

    Parameters
    ----------
    - img_kp_and_des (list): (Nx2) where N = Number of cameras or images - ex: [[kp1, des1], [kp2, des2], [kp3, des3]....]
    - cam_transformations (list or i think ndarray works as well): (Nx3x4) where N = Number of cameras or images - List of camera poses (transformation matrices) ex [[[r11,r12,r13,tx],[r21,r22,r23,ty],[r31,r32,r33,tz]], [Transform2], [Transform3].....]
    - fk1k2 (list): (3x1) - List containing initial guesses of focal length and 2 distortion parameters k1 and k2
    - hamming_threshold (int) - Threshold used for matching binary orb descriptors, goes from 0 to 32, the lower the number means we only keep the strongest descriptors which also results in fewer matches
    - search_window (int) - Value determining how many images (looking ahead) we will look for matching correspondences, this is biquadratic aka O(N^4) meaning this should not be too high or performance will be slow, 7 is a good number when using less than 200 orb features
    - [ignore this] window_shift (int) - Value (always less than search_window) that can be used to skip images ex if the next 7 images are searched instead of comparing every descriptor in the first one with the next we can skip ahead by window_shift aka if it is = 2 then we skip the second image and instead match every second up until we hit the value defined by search_window
    - [ignore this] window_shift_jump (int) - Can be used if
    - ftol (float) - Determining the stopping criteria for the least squares optimization problem, the less the better the parameters will be optimized 

    Returns
    -------
    - cam_transformations_optimized (ndarray): (Nx3x4) - ndarray of optimized camera transformations
    - Q_optimized (ndarray): (Nx3) - Optimized 3D points used in the bundle adjustment reprojection error minimization
    """
    Qs, cam_idxs, Q_idxs, qs = BA_pre_processing_cam_n_points(img_kp_and_des, cam_transformations, 25, 7, 1)
    
    cam_params = []
    for i in range(len(img_kp_and_des)):
        R, _ = np.asarray(cv2.Rodrigues(cam_transformations[i][:3, :3]))
        t = cam_transformations[i][:3, 3].flatten()
        cam_params.append([R[0][0], R[1][0], R[2][0], t[0], t[1], t[2], fk1k2[0], fk1k2[1], fk1k2[2]])

    cam_params = np.array(cam_params).flatten()
    cam_params = cam_params.reshape(int(cam_params.size/9), 9)
    n_cams = cam_params.shape[0]
    n_Qs = Qs.shape[0]

    # residual_init, residual_minimized, opt_params = bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs)
    sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
    #plot_sparsity(sparse_mat)
    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs,
                                                                                    qs, sparse_mat, f_tol = ftol)

    Q_optimized = opt_params[len(img_kp_and_des) * 9:]
    cam_transformations_optimized = np.empty((len(img_kp_and_des), 3, 4))
    for i in range(len(img_kp_and_des)):
        T_tmp = np.empty((3, 4))
        R, _ = cv2.Rodrigues(opt_params[i + 9*i: i + 3 + (i*9)])
        #print(R)
        T_tmp[:3, :3] = R
        T_tmp[:3, 3] = opt_params[i + 3 + 9*i: i + 6 + (i*9)]
        #print(opt_params[i + 3 + 9*i: i + 6 + (i*9)])
        cam_transformations_optimized[i] = T_tmp

    #cam_transformations_optimized = np.reshape(cam_transformations_optimized, (len(img_kp_and_des), 3, 4))
    Q_optimized = np.reshape(Q_optimized, (n_Qs, 3))

    #print(cam_transformations_optimized.shape)
    #print(Q_optimized.shape)

    return cam_transformations_optimized, Q_optimized

# Example Usage:

def main():
    basedir = './data'
    sequence = '00'

    num_images = 100
    frames = range(0, num_images, 1) #Indicate how many frames to use
    dataset = pykitti.odometry(basedir, sequence, frames=frames)#, frames=frames)
    
    poses = dataset.poses

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=100) # nFeatures is a very important number because the lower it is, the faster the computation time
    # find the keypoints and descriptors with ORB
    img_kp_and_des = []
    cam_params = []
    cam_transformations = []
    start = time.process_time()
    for i in range(num_images):
        img = np.array(dataset.get_cam0(i))
        kp_tmp, des_tmp = orb.detectAndCompute(img, None)
        img_kp_and_des.append([kp_tmp, des_tmp])

        R, _ = np.asarray(cv2.Rodrigues(poses[i][:3, :3]))
        t = poses[i][:3, 3].flatten()
        fndis = [-5.14862520e+00, -1.14997375e-06, -6.67297786e-13]
        cam_params.append([R[0][0], R[1][0], R[2][0], t[0], t[1], t[2], fndis[0], fndis[1], fndis[2]])
        cam_transformations.append(poses[i])
    
    camT_new, Q_new = SLAM_bundle_adjustment(img_kp_and_des, cam_transformations)
    print("new Cams: ", camT_new)
    print("new Qs: ", Q_new)


if __name__ == "__main__":
    main()