import os
from graph_tool.all import *
import numpy as np
import cv2

import pykitti
import sys
import os

from pyrsistent import v

# loop-closure detecting might have to keep the vertex descriptor in order to indicate where the graph have to be corrected
# No idea how to handle when moving down a already discovered path. Might have to say that within loop closure, the already known path is the correct one
#   Could be that the graph just does not get updated while in known territory 
# Might have to do the graph directed. Depends how the edge are defined. If they are defined as (last_V, V), there is no need. If they are getting ordered in acending order, thats a problem, since you can't tell which way the transformation is
#   It is ordered
# Måske hashtable? (kigge på x,y)
#   HVer gang vi laver ny vertex, skla den ind på et hashtable
#   Måske løser visual words det hele
#   Bruge dict, kan bruge (x,y) til key, hver værdi der hører til er descriptoren



# useful commands
# g.get_edges(), gives you a matrix of the different edges, and their vertices in order
class graphstructure():
    def __init__(self,pose,virtuelInfo):

        self.g=Graph(directed=False)

        self.v_pose=self.g.new_vertex_property("object")
        self.v_Vinfo=self.g.new_vertex_property("object")
        self.v_keypoints = self.g.new_vertex_property("object")
        self.v_descriptors = self.g.new_vertex_property("object")
        
        self.v_image_idx = self.g.new_vertex_property("int")


        self.e_unc=self.g.new_edge_property("float")
        self.e_trans=self.g.new_edge_property("object")
        self.e_kp_matches = self.g.new_edge_property("object")


        self.edgelist=[]


        self.last_v=self.g.add_vertex()
        self.v_pose[self.last_v]=pose
        self.v_Vinfo[self.last_v]=virtuelInfo

        #print(self.v_pose(self.last_v))
    def add_vertex (self, pose, virtuelInfo, uncertanity, transformation, kp_matches, keypoints):
        v=self.g.add_vertex()

        self.v_pose[v]=pose
        self.v_Vinfo[v]=virtuelInfo
        self.v_keypoints[v] = keypoints

        e=self.g.add_edge(self.last_v,v)
        self.e_unc[e]=uncertanity
        self.e_trans[e]=transformation
        self.e_kp_matches[e]= kp_matches
        self.last_v=v

    def add_edge (self,uncertainty, transformation, kp_matches, targetV):
        e=self.g.add_edge(self.last_v,targetV)
        self.e_unc[e]=uncertainty
        self.e_trans[e]=transformation
        self.e_kp_matches[e]= kp_matches
        
        self.edgelist.append(e)
        self.last_v=targetV


    def helpGraph ():
        """
        Useful commands
        ----------
        g.get_edges(): returns the edge matrix, with the edges and their vertices
        g.edges(): returns the descriptor of a edge. Can be used as the index, but only for deletion
        g.get_vertices(): returns the index of the vertices
        g.vertices(): returns the descriptor of a vertex. Can be used as the index, but have more options, you just cant visualise it
        """
        pass


