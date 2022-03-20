from visual_odometry_mono_template import visual_odemetry_mono
import Graphwrapper
from graph_tool.all import *
import numpy as np


def main():
    print("Project in VSLAM")
    #test
    #visual_odemetry_mono()
    
    transtest=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    transtest2=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
    transtest3=[1,2,3,4,5,6,7,8,9,10]
    infotest=np.array([1,2,3])

    test=Graphwrapper.graphstructure(transtest2, infotest)

    test.g.add_vertex(4)
    test.g.add_edge(test.g.vertex(3),test.g.vertex(2))
    edg = test.g.get_edges()
    ver=test.g.get_vertices()
    print(ver)






if __name__ == "__main__":
    main()