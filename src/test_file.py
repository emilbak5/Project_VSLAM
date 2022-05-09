
def get_image_points(matches):
    points_in_frame_1 = np.array(
            [match.keypoint1 for match in matches], dtype=np.float64)
    points_in_frame_2 = np.array(
            [match.keypoint2 for match in matches], dtype=np.float64)
    return points_in_frame_1, points_in_frame_2

def determine_essential_matrix(matches, K_camera_matrix):
    points_in_frame_1, points_in_frame_2 = get_image_points(matches)

    confidence = 0.99
    ransacReprojecThreshold = 1
    essential_matrix, mask = cv2.findEssentialMat(
            points_in_frame_1,
            points_in_frame_2, 
            K_camera_matrix, 
            cv2.FM_RANSAC, 
            confidence,
            ransacReprojecThreshold)

    inlier_matches = [match 
            for match, inlier in zip(matches, mask.ravel() == 1)
            if inlier]

    return inlier_matches