from src.Graphwrapper import *




def add_to_graph(transform, desc, kp, i, keyframe_idx, graph: graphstructure, current_pose):
    

    #kp = cv2.KeyPoint_convert(kp)
    
    prev_vertex = graph.g.vertex(i-1)
    vertex = graph.g.add_vertex()
    edge = graph.g.add_edge(prev_vertex, vertex)
    
    graph.v_descriptors[vertex] = desc
    graph.v_keypoints[vertex] = kp
    graph.v_image_idx[vertex] = keyframe_idx
    
    graph.e_trans[edge] = np.matmul(current_pose, transform)


    x = 5
    

    


