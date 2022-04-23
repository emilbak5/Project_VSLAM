from src.Graphwrapper import *




def add_to_graph(transform, desc, kp, i, graph: graphstructure):


    kp = cv2.KeyPoint_convert(kp)
    
    prev_vertex = graph.g.vertex(i-1)
    vertex = graph.g.add_vertex()
    edge = graph.g.add_edge(prev_vertex, vertex)
    
    graph.v_descriptors[vertex] = desc
    graph.v_keypoints[vertex] = desc
    
    graph.e_trans[edge] = transform

    x = 5
    

    


