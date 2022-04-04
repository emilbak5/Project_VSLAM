from distutils.log import info
from visual_odometry_mono_template import visual_odemetry_mono
import Graphwrapper
from graph_tool.all import *
import numpy as np


from src.helper_functions import *
from src.keyframe import *


def main():
    print("Project in VSLAM")
    #test
    #visual_odemetry_mono()
    
    # transtest=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    # transtest2=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
    # transtest3=[1,2,3,4,5,6,7,8,9,10]
    # infotest=np.array([1,2,3])

    # test=Graphwrapper.graphstructure(transtest2, infotest)

    # test.g.add_vertex(4)
    # test.g.add_edge(test.g.vertex(3),test.g.vertex(2))
    # edg = test.g.get_edges()
    # ver=test.g.get_vertices()
    # print(ver)


    num_images = 500
    dataset = get_dataset(num_images)
    gt_poses = dataset.poses
    infotest=np.array([1,2,3])
    graph = Graphwrapper.graphstructure(gt_poses[0], infotest)
    




    for i in range(num_images):
    
        
        idx = get_keyframe(dataset, i, graph)

    #     if key_frame_found:
    #         desc = get_descripters(idx)
    #         add_to_lsh_table(desc)
    #         transform = stereo_vo(desc)
    #         add_to_graph(transform, desc, i, graph)
    #         visualize_path(graph)
    #         idx, loop_closure_found_bool = check_for_loop_closure(i, graph, lsh_table)
    #         if loop_closure_found_bool:
    #             perform_loop_closure(idx, graph)
    #             visualize_path(graph)




    


    







if __name__ == "__main__":
    main()