import cv2



def stero_vo(kp, desc, graph):
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
    img1_l = np.array(dataset.get_cam0(i - 1))
    img2_l = np.array(dataset.get_cam0(i))
    #img1_l, img2_l = images_l[i - 1:i + 1]

    # Get teh tiled keypoints
    trackpoints1 = prev_frame.points

    tp1_l = graph.g.get_vertex(i)
    tp2_l = cv2.KeyPoint_convert(kp)

    # Track the keypoints
    
    # Calculate the disparitie
    disparities.append(np.divide(disparity.compute(img2_l, np.array(dataset.get_cam1(i))).astype(np.float32), 16))

    # Calculate the right keypoints
    tp1_l, tp1_r, tp2_l, tp2_r = calculate_right_qs(tp1_l, tp2_l, disparities[i - 1], disparities[i])

    # Calculate the 3D points
    Q1, Q2, enough_points = calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)
    if enough_points:
    # Estimate the transformation matrix
        transformation_matrix = estimate_pose(tp1_l, tp2_l, Q1, Q2)
        ##### Keyframe descision
        kp2_l = np.array([])
        if(len(tp2_l) < 200):
            kp2_l = get_tiled_keypoints(img2_l, 10, 20)
            kp2_l = kp_left_in_right(kp2_l, disparities[i])
            tp2_l = cv2.KeyPoint_convert(kp2_l)
        
        prev_frame = dummy_graph.Frame(tp2_l, transformation_matrix)

        frame = dummy_graph.Frame(kp2_l, transformation_matrix)

        return frame, enough_points
    else: 
        return dummy_graph.Frame(np.ndarray([]), np.identity(4)), enough_points
