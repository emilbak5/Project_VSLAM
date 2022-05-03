import cv2
from Graphwrapper import *





def get_next_keyframe(dataset, current_img_idx: int, graph: graphstructure, orb, graph_size):



    img_iter = 1
    img_curr = np.array(dataset.get_cam0(current_img_idx))
    vertex_prev = graph.g.vertex(graph_size - 1)

    trackpoints = graph.v_keypoints[vertex_prev]

    
    

    while True:
        
        if current_img_idx + img_iter < dataset.frames.stop:
            img_next = np.array(dataset.get_cam0(current_img_idx + img_iter))
        else:
            img_next = np.array(dataset.get_cam0(dataset.frames.stop - 1))
        
        tp1_l, tp2_l = track_keypoints(img_curr, img_next, trackpoints)

        if len(tp1_l) < 100:
            return current_img_idx + img_iter
        
        img_iter += 1





def track_keypoints(img1, img2, trackpoints1_, max_error=4):
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
    lk_params = dict(winSize=(15, 15),
                    flags=cv2.MOTION_AFFINE,
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    trackpoints1 = np.expand_dims(trackpoints1_, axis=1) 

    # Use optical flow to find tracked counterparts
    trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **lk_params)

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
        




