from distutils.log import info
from visual_odometry_mono_template import visual_odemetry_mono
import Graphwrapper
from graph_tool.all import *
import numpy as np


from src.descriptors import *
from src.helper_functions import *
from src.keyframe import *
from src.lsh_flann import *
from src.stereo_vo_cleaned import *
from src.graph_functions import *


def main():
    print("Project in VSLAM")


    num_images = 500
    dataset = get_dataset(num_images)
    gt_poses = dataset.poses
    infotest=np.array([1,2,3])
    graph = Graphwrapper.graphstructure(gt_poses[0], infotest)

    VO = VisualOdometry(dataset)

    orb = cv2.ORB_create(1000)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams={})

    


    prev_idx = 0
    ## First image
    keypoints, desc = orb.detectAndCompute(np.array(dataset.get_cam0(0)), None)
    keypoints = cv2.KeyPoint_convert(keypoints)
    vertex_0 = graph.g.vertex(0)
    graph.v_keypoints[vertex_0] = keypoints
    graph.v_descriptors[vertex_0] = desc




    current_img_idx = 1
    graph_size = 1
    
    while current_img_idx < num_images:        
        
        keyframe_idx = get_next_keyframe(dataset, current_img_idx, graph, orb, graph_size)


        kp, desc = get_descripters(keyframe_idx, dataset, orb)
        add_to_lsh_table(desc, flann)
        transform, _ = VO.get_pose(kp, desc, dataset, graph, current_img_idx, prev_idx)
        add_to_graph(transform, desc, kp, graph_size, keyframe_idx, graph)
        graph_size += 1


        if graph_size % 7 == 0:
            pass
            #bundle adjustment

        # visualize_path(graph)
        idx, loop_closure_found_bool = find_most_similar_image(graph_size, graph, flann) # Is not done
        if loop_closure_found_bool:
            pass
        #     perform_loop_closure(idx, graph)
        #     update_visualize_path(graph)
        prev_idx = current_img_idx
        current_img_idx = keyframe_idx
        print (graph_size)





    


    







if __name__ == "__main__":
    main()