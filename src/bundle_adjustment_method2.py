import os
import sys
#from sympy import true

from yaml import parse
sys.path.append(r"/mnt/c/Users/Thobi/Documents/Exersises/Project_VSLAM") #Use this to get the lib module to work (gets current working dir)
#sys.path.append(r"C:\Users\Ulric\OneDrive - Syddansk Universitet\8. Semester\Advanced Computer Vision\Exercises\ProjectStructure\AdvancedComputerVisionExercises")

import bz2

import numpy as np
import time
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
import matplotlib.pyplot as plt
import pykitti
import random


from lib.visualization.plotting import VisualDataSource


def objective(params, n_cams, n_Qs, cam_idxs, Q_idxs, qs, first_cam):
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
    first_cam (ndarray): First camera parameters

    Returns
    -------
    residuals (ndarray): The residuals
    """
    # Should return the residuals consisting of the difference between the observations qs and the reporjected points
    # Params is passed from bundle_adjustment() and contains the camera parameters and 3D points

    params = np.hstack((first_cam.ravel(), params))
    cam_params = np.array(params[:n_cams * 6])
    cam_params = cam_params.reshape((n_cams, 6))

    Qs = params[n_cams * 6:].reshape((n_Qs, 3))

    Qs_points = Qs[Q_idxs]
    K = np.array([[718.856,0,607.1928],[0,718.856,185.2157],[0,0,1]])
    qs_proj = np.empty((Q_idxs.size, 2))
    for i in range(Q_idxs.size):
        pt, _ = cv2.projectPoints(Qs_points[i], params[cam_idxs[i]*6:cam_idxs[i]*6 + 3], params[cam_idxs[i]*6 + 3:cam_idxs[i]*6 + 6], K, np.empty(0))
        qs_proj[i] = pt

    # Calculate the residuals
    residuals = (qs_proj - qs).ravel()

    return residuals

def BA_pre_processing_cam_n_points(images, kp_and_des, matches_list, cam_transformations):
    def find_match(train_idx, matches): # Fast search for match in sorted list, by dividing list in half each iteration
        # Return index of match (where it is in the list) if found, else return -1
        a = 0
        b = len(matches) - 1
        middle_idx = -1
        found = False

        while not found:
            if b - a <= 10:
                for i in range(a, b + 1):
                    query_idx = matches[i][0].queryIdx
                    if train_idx == query_idx:
                        return i
                return -1

            middle_idx = round((b - a) / 2) + a
            query_idx = matches[middle_idx][0].queryIdx
            if train_idx == query_idx:
                return middle_idx
            if train_idx < query_idx:
                b = middle_idx
            if train_idx > query_idx:
                a = middle_idx

    def match_linked_list(matches_list):
        # Linked list creation
        old_bins = []       # [[[cam_idx, Q_idxs, kp],[cam_idx, Q_idxs, kp],...], [bin2], [bin3]]
        active_bins = []    # [
                            #    [c0, query_idx_1], [c1, train_idx_3], [c2, train_idx_5] #Both query_idx and train_idx are keypoint indexes
                            #    [c0, query_idx_2], [c1, train_idx_2], [c2, train_idx_4]    
                            # ]
                            #

        for edge in range(len(matches_list)):
            matches : cv2.DMatch = matches_list[edge]
            

            #match2 = matches[edge+1]

            # Sort to make search faster
            matches = sorted(matches, key=lambda x: x[0].queryIdx)
            no_longer_active_bins = []

            #kp1, kp2 = get_image_points(match1, kp_and_des[edge], kp_and_des[edge+1])
            #_, kp3 = get_image_points(match2, kp_and_des[edge+1], kp_and_des[edge+2])

            n_active_bins = len(active_bins)
            not_found_matches = np.ones(len(matches))

            for m in range(n_active_bins): # Loop every row in the active bins
                match_idx = find_match(active_bins[m][-1][1], matches)  # Find the correspondence pair in match, where trainIdx in active bins matches queryIdx in match
                if match_idx != -1:
                    not_found_matches[match_idx] = 0 #Match found in matches at index match_idx
                    active_bins[m].append([edge + 1, matches[match_idx][0].trainIdx])#Is edges right index?
                else:
                    no_longer_active_bins.append(m)
                    
            for idx in reversed(no_longer_active_bins): #If match sequences are no longe ractive they are sent to the old bins
                old_bins.append(active_bins.pop(idx))
            
            for idx in range(len(not_found_matches)): #Matches not found in active bins are appended
                if not_found_matches[idx] == 1:
                    active_bins.append([[edge, matches[idx][0].queryIdx], [edge + 1, matches[idx][0].trainIdx]])
        return old_bins, active_bins

    old_bins, active_bins = match_linked_list(matches_list)

    all_bins = old_bins

    for idx in range(len(active_bins)): #Put all remaining active bins on old bins
        all_bins.append(active_bins[idx])
    
    #For each row of old bins Compute Q, log each camera seing point, log Q_idx, log keypoint position
    Qs = []
    cam_idxs = []
    Qs_idxs = []
    qs = []

    for bin in all_bins:
        cam_1 = bin[0][0]
        cam_2 = bin[1][0]
        kp1_idx = bin[0][1]
        kp2_idx = bin[1][1]
        kp1 = kp_and_des[cam_1][0][kp1_idx]
        kp2 = kp_and_des[cam_2][0][kp2_idx]
        Q = triangulation(np.asarray(kp1.pt), np.asarray(kp2.pt), cam_transformations[cam_1], cam_transformations[cam_2])
        # #Visualisation
        # img_1 = cv2.drawKeypoints(images[cam_1], [kp1],0, (255,0,0))
        # img_2 = cv2.drawKeypoints(images[cam_2], [kp2],0, (255,0,0))
        # numpy_horizontal = np.hstack((img_1, img_2))
        # winname = "keypoint"
        # cv2.imshow(winname, numpy_horizontal)
        # cv2.waitKey(0)
        Qs.append(Q)
        Qs_idx = len(Qs)-1
        for item in bin:
            cam_idxs.append(item[0])
            Qs_idxs.append(Qs_idx)
            kp = kp_and_des[item[0]][0][item[1]]
            qs.append(kp.pt)

    return np.array(Qs), np.array(cam_idxs), np.array(Qs_idxs), np.array(qs)



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

    # Remove the first camera from the matrix because those parameters are not included in the optimizer
    sparse_mat = sparse_mat[:,6:]
    
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
    n_cams = cam_params.shape[0]
    # Save the initial residuals
    residual_init = objective(params[6:], cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs, params[:6])

    # Perform the least_squares optimization with sparsity
    res = least_squares(objective, params[6:], jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-5, method='trf',
                        args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs, params[:6]))

    # Get the residuals at the solution and the solution
    residuals_solu = res.fun
    solu = res.x
    return residual_init, residuals_solu, solu

def hamming_dist(x, y):
    d = np.count_nonzero(x != y)
    return d

def triangulation(kp1, kp2, T_1w, T_2w):
    # Function modified from: Project: DF-VO   Author: Huangying-Zhan   File: ops_3d.py    License: MIT License
    """Triangulation to get 3D points
    Args:
        kp1 (Nx2): keypoint in view 1 (normalized)
        kp2 (Nx2): keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (3xN): 3D coordinates of the keypoints w.r.t world coordinate
    """
    kp1_3D = np.ones((3, 1))
    kp2_3D = np.ones((3, 1))
    kp1_3D[0], kp1_3D[1] = kp1[0], kp1[1]
    kp2_3D[0], kp2_3D[1] = kp2[0], kp2[1]

    K = np.array([[718.856,0,607.1928],[0,718.856,185.2157],[0,0,1]])

    T_1w = np.matmul(K, T_1w[:3])
    T_2w = np.matmul(K, T_2w[:3])
    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    X /= X[3]

    return X[:3]

def main():
    print("HEY")

    basedir = './data'
    sequence = '00'

    num_images = 7
    frames = range(0, num_images, 1) #Indicate how many frames to use
    dataset = pykitti.odometry(basedir, sequence, frames=frames)#, frames=frames)
    
    #K = dataset.calib.K_cam0
    #print(K)
    
    poses = dataset.poses

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=100) # nFeatures is a very important number because the lower it is, the faster the computation time
    # find the keypoints and descriptors with ORB

    images = []             
    img_kp_and_des = []
    cam_params = []
    cam_transformations = []
    start = time.process_time()
    for i in range(num_images):
        posnew = poses[i].copy()
        posnew[:3, :3] = np.transpose(posnew[:3, :3])
        posnew[:3, 3] = np.matmul(-posnew[:3, :3], posnew[:3, 3]) 
        #print("i=",i," ",poses[i][:3, 3])

        img = np.array(dataset.get_cam0(i))
        images.append(img)
        kp_tmp, des_tmp = orb.detectAndCompute(img, None)
        img_kp_and_des.append([kp_tmp, des_tmp])
        R, _ = np.asarray(cv2.Rodrigues(posnew[:3, :3]))
        t = posnew[:3, 3].flatten()

        cam_params.append([R[0][0], R[1][0], R[2][0], t[0], t[1], t[2]])
        cam_transformations.append(posnew)

    print("DetectAndComputes took: ", time.process_time() - start, " Seconds")

    #print(hamming([1,1,0,2,0,1],[0,1,1,2,1,0])*6) # Hamming distance = 4 because 4 bits differ
        # Create matches
    matcher = cv2.BFMatcher()
    matches_list = []
    for i in range(num_images-1):
        matches_tmp = matcher.knnMatch(img_kp_and_des[i][1], img_kp_and_des[i+1][1], k=2)
        # Do lowes test:
        good = []
        for m,n in matches_tmp:
            if m.distance < 0.75*n.distance:
                good.append([m])
        matches_list.append(good)

    Qs, cam_idxs, Q_idxs, qs = BA_pre_processing_cam_n_points(images, img_kp_and_des, matches_list, cam_transformations)


    print("cam_idxs: ", cam_idxs.shape)
    print("Q_idxs: ", Q_idxs.shape)
    print("qs: ", qs.shape)
    print("Qs: ", Qs.shape)
    print(cam_idxs)

    
    cam_params = np.array(cam_params).flatten()
    print("old_cam1t: ", cam_params[3:6])
    print("old_cam2t: ", cam_params[3+6:6+6])
    for i in range(int(len(cam_params)/6)):
        cam_params[i*6 + 3] += round(random.uniform(-0.005, 0.005)*1, 6) # x
        #cam_params[i*6 + 4] += round(random.uniform(-0.01, 0.01)*1, 6)  # y
        cam_params[i*6 + 5] += round(random.uniform(-1.9, 1.9)*1, 6)     # z
    cam_params_noise = cam_params.copy()

    cam_params = cam_params.reshape(int(cam_params.size/6), 6)
    n_cams = cam_params.shape[0]
    n_Qs = Qs.shape[0]
    print("n_cameras: {}".format(n_cams))
    print("n_points: {}".format(n_Qs))
    print("Total number of parameters: {}".format(6 * n_cams + 3 * n_Qs))
    print("Total number of residuals: {}".format(2 * qs.shape[0]))

    # residual_init, residual_minimized, opt_params = bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs)
    sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
    VisualDataSource.plot_sparsity(sparse_mat)
    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs,
                                                                                    qs, sparse_mat)
    # plot_residual_results(qs, residual_init,
    #                       residual_minimized)

    x0 = np.hstack((cam_params.ravel(), Qs.ravel()))
    residual_init = objective(x0[6:], n_cams, n_Qs, cam_idxs, Q_idxs, qs, x0[:6])


    opt_params = np.hstack((cam_params.ravel()[:6], opt_params))

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
        est_rot, _ = cv2.Rodrigues(opt_params[0+(i*6):3+(i*6)])
        est_t = np.matmul(-np.transpose(est_rot), est) 

        noise = cam_params_noise[3+(i*6):6+(i*6)]
        noise_rot, _ = cv2.Rodrigues(cam_params_noise[0+(i*6):3+(i*6)])
        noise_t = np.matmul(-np.transpose(noise_rot), noise)

        xs_gt = np.append(xs_gt, pos[0])
        ys_gt = np.append(ys_gt, pos[1])
        zs_gt = np.append(zs_gt, pos[2])
        xs_iniguess = np.append(xs_iniguess, noise_t[0])
        ys_iniguess = np.append(ys_iniguess, noise_t[1])
        zs_iniguess = np.append(zs_iniguess, noise_t[2])
        xs_est = np.append(xs_est, est_t[0])
        ys_est = np.append(ys_est, est_t[1])
        zs_est = np.append(zs_est, est_t[2])

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Axes values are scaled individually by default')
    ax1.scatter(xs_gt,zs_gt)
    ax2.scatter(xs_iniguess,zs_iniguess)
    ax3.scatter(xs_est,zs_est)
    plt.show()


    plt.figure(1, figsize=(15, 5))
    plt.subplot(211)
    plt.plot(residual_init)
    plt.subplot(212)
    plt.plot(residual_minimized)
    plt.show()


if __name__ == "__main__":
    main()
