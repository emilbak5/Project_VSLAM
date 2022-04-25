from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import cv2
import pykitti
import numpy as np
from collections import Counter
from src.Graphwrapper import *



# def find_most_similar_image(flann_matcher: cv2.FlannBasedMatcher, image_idx_to_search_for: int):
#     print ("Matching...")

#     img = np.array(dataset.get_cam0(image_idx_to_search_for))
#     _, des_test = orb.detectAndCompute(img, None)
#     dmatches = flann.match(des_test)

#     index_list = [dmatch.imgIdx for dmatch in dmatches]
#     occurence_count = Counter(index_list)
#     most_occuring = occurence_count.most_common(1)[0][0]
#     return most_occuring

# basedir = './data'
# sequence = '00'
# # data_dir = './data/sequences/00'  # Try KITTI_sequence_2 too

# number_of_images = 500

# frames = range(0, number_of_images, 1) #Indicate how many frames to use
# dataset = pykitti.odometry(basedir, sequence, frames=frames)


# orb = cv2.ORB_create(600)
# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
# search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams={})


# for i in range(number_of_images):
#     img = np.array(dataset.get_cam0(i))
#     keypoints, des = orb.detectAndCompute(img, None)
#     flann.add([des])


# print (str(len(flann.getTrainDescriptors())))


# img_to_look_for = 10
# most_similar_idx = find_most_similar_image(flann, img_to_look_for)



# print("done")
# cv2.imshow("Image to search for", np.array(dataset.get_cam0(img_to_look_for)))
# cv2.imshow("Most similar image", np.array(dataset.get_cam0(most_similar_idx)))


# cv2.waitKey()
def find_most_similar_image(graph_size, graph: graphstructure, lsh_table: cv2.FlannBasedMatcher):

    most_occuring = None
    loop_closure_found = False

    if graph_size > 10:
        print ("Matching...")

        vertex = graph.g.vertex(graph_size - 1)

        desc = graph.v_descriptors[vertex]

        all_desc = lsh_table.getTrainDescriptors()
        all_desc = all_desc[5:]

        flann = lsh_table.clone()
        flann.clear(True)
        flann.add(all_desc)

        dmatches = flann.match(desc)

        

        index_list = [dmatch.imgIdx for dmatch in dmatches]
        occurence_count = Counter(index_list)
        most_occuring = occurence_count.most_common(1)[0][0]

    ### Insert track keypointsfunc to see how many is in the same image
    return most_occuring

def add_to_lsh_table(desc, flann):
    flann.add([desc])




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

