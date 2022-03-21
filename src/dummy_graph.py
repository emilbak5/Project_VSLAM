import numpy as np
class Frame():
    def __init__(self, points, pose_):
        self.points = points
        self.pose = pose_
 
class Graph():
    def __init__(self):
        self.vertexes = np.array([])
        self.edges = np.array([])

    def init(self, keyframe0):
        self.vertexes = np.append(self.vertexes, keyframe0)

    def add_vertex(self, keyframe):
        self.vertexes = np.append(self.vertexes, keyframe)

    def add_edge(self, pose):
        self.edges = np.append(self.edges, pose)