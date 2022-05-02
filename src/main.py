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

from bokeh.models import Button
from bokeh.server.server import Server


from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show, save
from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.layouts import column, layout, gridplot, row
from bokeh.models import Div, WheelZoomTool, Slider
from bokeh.models.widgets import Panel, Tabs
from bokeh.driving import count

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


print("Project in VSLAM")


num_images = 4000
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

gt_path = []
estimated_path = []

gt_path.append((gt_poses[0][0, 3], gt_poses[0][2, 3]))
estimated_path.append((gt_poses[0][0, 3], gt_poses[0][2, 3]))

# Visu=plotting.VisualDataSource(gt_path, estimated_path)
# Visu.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry",
#                 file_out=os.path.basename("data") + 'll'".html")



current_img_idx = 1
graph_size = 1

# while current_img_idx <= num_images - 2:        
    
#     keyframe_idx = get_next_keyframe(dataset, current_img_idx, graph, orb, graph_size)


#     kp, desc = get_descripters(keyframe_idx, dataset, orb)
#     add_to_lsh_table(desc, flann)
#     transform, _ = VO.get_pose(kp, desc, dataset, graph, current_img_idx, prev_idx)
#     add_to_graph(transform, desc, kp, graph_size, keyframe_idx, graph)
#     graph_size += 1


#     if graph_size % 7 == 0:
#         pass
#         #bundle adjustment

#     # visualize_path(graph)
#     idx, loop_closure_found_bool = find_most_similar_image(graph_size, graph, flann) # Is not done
#     if loop_closure_found_bool:
#         pass
#     #     perform_loop_closure(idx, graph)
#     #     update_visualize_path(graph)
#     prev_idx = current_img_idx
#     current_img_idx = keyframe_idx

#     print (graph_size)





fig, ax = plt.subplots()
gt_data_x, gt_data_y = [], []
esti_data_x, esti_data_y = [], []
error_x, error_y = [], []


gt_line, = plt.plot([], [], marker='o', color='b')
esti_line, = plt.plot([], [], marker='o', color='r')
lines = [gt_line, esti_line]


i = 0
prev_max_x = 0
prev_min_x = 0
prev_max_y = 0
prev_min_y = 0

def init():
    for line in lines:
        line.set_data([],[])
    return lines

i = 0
def update(_):
    global i
    global prev_max_x
    global prev_max_y
    global prev_min_x
    global prev_min_y
    global gt_poses


    gt_path = gt_poses[i][0, 3], gt_poses[i][2, 3]
    gt_path_x, gt_path_y = [gt_path[0], gt_path[1]]

    # esti_path_x = gt_data_x + 10
    # esti_path_y = gt_data_y + 10

    gt_data_x.append(gt_path_x)
    gt_data_y.append(gt_path_y)
    esti_data_x.append(gt_path_x + 10)
    esti_data_y.append(gt_path_y + 10)


    gt_line.set_data(gt_data_x, gt_data_y)
    esti_line.set_data(esti_data_x, esti_data_y)
    # if len(xdata) % 10 == 0:
    #     xdata[i-5] = 4
    #     ydata[i-5] = -0.5
    

    border = 10
    if gt_path_x + border > prev_max_x:
        ax.set_xlim(prev_min_x, gt_path_x + border)
        prev_max_x = gt_path_x + border

    if gt_path_x - border < prev_min_x:
        ax.set_xlim(gt_path_x - border, prev_max_x)
        prev_min_x = gt_path_x - border

    if gt_path_y + border > prev_max_y:
        ax.set_ylim(prev_min_y, gt_path_y + border)
        prev_max_y = gt_path_y + border

    if gt_path_y - border < prev_min_y:
        ax.set_ylim(gt_path_y - border, prev_max_y)
        prev_min_y = gt_path_y - border

    i += 20

    test = 0
    for i in range(10):
        test += 10
    
    return lines,

ani = FuncAnimation(fig, update, frames=None, interval=10,
                    init_func=init, blit=False)
plt.show()

    



# current_img_idx = 1
# graph_size = 1
# end_of_program = False

# #@count()
# def callback():
#     global current_img_idx
#     global graph_size
#     global num_images
#     global prev_idx
#     global end_of_program
#     #for i in range(len(Visu.gt_path)-1):
#     # gt_path = np.array(Visu.gt_path[t+1])
#     # pred_path = np.array(Visu.pred_path[t+1])
# # while current_img_idx <= num_images - 2:        
#     if current_img_idx < num_images - 2:
#         keyframe_idx = get_next_keyframe(dataset, current_img_idx, graph, orb, graph_size)


#         kp, desc = get_descripters(keyframe_idx, dataset, orb)
#         add_to_lsh_table(desc, flann)
#         transform, _ = VO.get_pose(kp, desc, dataset, graph, current_img_idx, prev_idx)
#         add_to_graph(transform, desc, kp, graph_size, keyframe_idx, graph)
#         graph_size += 1


#         if graph_size % 7 == 0:
#             pass
#             #bundle adjustment

#         # visualize_path(graph)
#         idx, loop_closure_found_bool = find_most_similar_image(graph_size, graph, flann) # Is not done
#         if loop_closure_found_bool:
#             pass
#         #     perform_loop_closure(idx, graph)
#         #     update_visualize_path(graph)
#         prev_idx = current_img_idx
#         current_img_idx = keyframe_idx

#         #print (graph_size)
#         print(current_img_idx)

#     else:
#         if end_of_program == False:
#             print("End of program")
#         end_of_program = True


#     # gt_x, gt_y = gt_path.T
#     #pred_x, pred_y = transform.T
 
#     pred_path = np.array([transform[0, 3], transform[2, 3]])
#     pred_path_x, pred_path_y = pred_path[0], pred_path[1]


#     gt_path = gt_poses[keyframe_idx]
#     gt_path = np.array([gt_path[0, 3], gt_path[2, 3]])


#     gt_path_x, gt_path_y = [gt_path[0], gt_path[1]]

    
#     pred_path=pred_path.reshape(-1,2)
#     gt_path=gt_path.reshape(-1,2)
    
    
#     temp1=np.array([gt_path_x, pred_path_x]).T
#     xs = list(temp1.reshape(-1,2))

#     temp2=np.array([gt_path_y, pred_path_y]).T
#     ys = list(temp2.reshape(-1,2))

#     # print("test")

#     diff = np.linalg.norm(gt_path - pred_path, axis=1)
#     print(len(diff))
    
#     colorArray=['blue']

#     data=dict(gtx=gt_path[:, 0], gty=gt_path[:, 1],
#                                         px=pred_path[:, 0], py=pred_path[:, 1],
#                                         diffx=np.arange(len(diff)), diffy=diff,
#                                         disx=xs, disy=ys, color=colorArray)

#     #print(data)
#     Visu.source.stream(data,100)

# curdoc().theme='dark_minimal'

# curdoc().add_root(Visu.plot)
# curdoc().title = "OHLC"
# curdoc().add_periodic_callback(callback,5000)
# print("here")
















