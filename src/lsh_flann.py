from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import cv2
import pykitti
import numpy as np
from collections import Counter
from src.Graphwrapper import *
from matplotlib import pyplot as plt




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
def find_most_similar_image(graph_size, graph: graphstructure, lsh_table: cv2.FlannBasedMatcher, dataset, current_img_idx):
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams={})
    most_occuring = None
    loop_closure_found = False
    lsh_table.clear()
    test = len(lsh_table.getTrainDescriptors())

    graph_size_before_starting = 100

    if graph_size > graph_size_before_starting + 2:
        # print ("Matching...")

        vertex = graph.g.vertex(graph_size - 1)

        desc = graph.v_descriptors[vertex]
        kp = graph.v_keypoints[vertex]
        current_img_idx_idx = graph.v_image_idx[vertex]

        edge = graph.g.edge(graph_size - 2, graph_size - 1)
        transform = graph.e_trans[edge]
        esti_path = transform[0, 3], transform[2, 3]
        current_x, current_y = [esti_path[0], esti_path[1]]

        close_enough_points_idx = []        

        edges = graph.g.get_edges()
        edges = edges[0:graph_size - graph_size_before_starting]

        for i, edge in enumerate(edges):
            edge_ = graph.g.edge(edge[0], edge[1])
            transform = graph.e_trans[edge_]
            esti_path = transform[0, 3], transform[2, 3]
            esti_path_x, esti_path_y = [esti_path[0], esti_path[1]]


            max_dist_allowed = 30
            if (esti_path_x < current_x + max_dist_allowed and esti_path_x > current_x - max_dist_allowed):
                if (esti_path_y < current_y + max_dist_allowed and esti_path_y > current_y - max_dist_allowed):
                    close_enough_points_idx.append(i)
        
        for i in close_enough_points_idx:
            vertex = graph.g.vertex(i)
            lsh_table.add([graph.v_descriptors[vertex]])
            

        # desc_to_compare = np.array(desc_to_compare)

        nr_of_desc = len(lsh_table.getTrainDescriptors())



        #lsh_table.add(all_desc)
        if nr_of_desc > 0:
            dmatches = lsh_table.knnMatch(desc, k=2)
            dmatches = [dmatch for dmatch in dmatches if len(dmatch) == 2]
            matchesMask = [[0,0] for i in range(len(dmatches))]

            good = []
            for i, (m,n) in enumerate(dmatches):
                if m.distance < 0.75*n.distance:
                    good.append([m])
                    matchesMask[i]=[1,0]



        
            draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)


        


            index_list = [dmatch[0].imgIdx for dmatch in dmatches]
            occurence_count = Counter(index_list)
            all_occurences = occurence_count.values()
            print(max(all_occurences))
            if max(all_occurences) > 10:
                most_occuring = occurence_count.most_common(1)[0][0]
                most_occuring = close_enough_points_idx[most_occuring]

                vertex = graph.g.vertex(most_occuring)

                idx_match = graph.v_image_idx[vertex]
                keypoints_match = graph.v_keypoints[vertex]

                img3 = cv2.drawMatchesKnn(np.array(dataset.get_cam0(idx_match)), cv2.KeyPoint_convert(keypoints_match), np.array(dataset.get_cam0(current_img_idx_idx)), cv2.KeyPoint_convert(kp), dmatches, None, **draw_params)
                cv2.imshow("test", img3)
                cv2.waitKey(0)
                # plt.imshow(img3,),plt.show()
                # cv2.waitKey(0)

                loop_closure_found = True
                print("Loop closure found!")
        else:
            print(0)


    ### Insert track keypointsfunc to see how many is in the same image
    return most_occuring, loop_closure_found

def add_to_lsh_table(desc, flann, graph: graphstructure):
    if len(graph.g.get_vertices()) > 20:
        vertex = graph.g.vertex(len(graph.g.get_vertices()) - 20)   
        descriptor = graph.v_descriptors[vertex]
 
        flann.add([descriptor])


