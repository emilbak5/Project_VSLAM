from visual_odometry_mono_template import visual_odemetry_mono
import Graphwrapper
from graph_tool.all import *
import numpy as np
from time import time
import math

from src.descriptors import *
from src.helper_functions import *
from src.keyframe import *
from src.lsh_flann import *
from src.stereo_vo_cleaned import *
from src.graph_functions import *



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import rcParams
rcParams['animation.convert_path'] = r'/usr/local/bin/convert'


import matplotlib


print("Project in VSLAM")


num_images = 4000
dataset = get_dataset(num_images)
gt_poses = dataset.poses
infotest=np.array([1,2,3])
graph = Graphwrapper.graphstructure(gt_poses[0], infotest)

VO = VisualOdometry(dataset)

orb = cv2.ORB_create(300)
#orb = cv2.SIFT_create(nfeatures=2000)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)




prev_idx = 0
## First image
keypoints, desc = orb.detectAndCompute(np.array(dataset.get_cam0(0)), None)
keypoints = cv2.KeyPoint_convert(keypoints)
vertex_0 = graph.g.vertex(0)
graph.v_keypoints[vertex_0] = keypoints
graph.v_descriptors[vertex_0] = desc

current_pose = gt_poses[0]

gt_path = []
estimated_path = []

gt_path.append((gt_poses[0][0, 3], gt_poses[0][2, 3]))
estimated_path.append((gt_poses[0][0, 3], gt_poses[0][2, 3]))


current_img_idx = 1
graph_size = 1


fig, ax = plt.subplots(1, 2 , figsize=(10,10))
#ax[1] = plt.subplot(132, figsize=(15,15))

gt_data_x, gt_data_y = [], []
esti_data_x, esti_data_y = [], []
error_x, error_y = [], []


gt_line, = ax[0].plot([], [], marker='o', color='b')
esti_line, = ax[0].plot([], [], marker='o', color='r')
error_line, = ax[1].plot([], [], marker='o', color='b')
lines = [gt_line, esti_line, error_line]


i = 0
prev_max_x = 0
prev_min_x = 0
prev_max_y = 0
prev_min_y = 0

prev_max_x_error = 0
prev_max_y_error = 0

def init():
    for line in lines:
        line.set_data([],[])

    ax[0].set_xlim(-10, 10)
    ax[0].set_ylim(-10, 10)
    ax[0].grid()

    ax[1].set_xlim(0, 10)
    ax[1].set_ylim(0, 10)
    ax[1].grid()
    return lines

i = 0
frame = range(0, num_images-1)



def gen():
    global current_img_idx
    i = 0
    while current_img_idx <= num_images:
        i += 1
        yield i


def update(_):
    global i
    global prev_max_x
    global prev_max_y
    global prev_min_x
    global prev_min_y
    global prev_max_x_error
    global prev_max_y_error
    global gt_poses
    global num_images
    global graph_size
    global current_img_idx
    global prev_idx
    global current_pose
    global keyframe_idx



    if current_img_idx < num_images - 1:

        #print()
        #print('---------------------------------')
        #print(f'Current img index : {current_img_idx}')
        start_time = time()
        keyframe_idx = get_next_keyframe(dataset, current_img_idx, graph, orb, graph_size)
        #print(f'Keyframe : {time() - start_time}')

        start_time = time()
        kp, desc = get_descripters(keyframe_idx, dataset, orb)
        #print(f'Getting desc : {time() - start_time}')

        add_to_lsh_table(desc, flann)

        start_time = time()
        transform, enough_points = VO.get_pose(kp, desc, dataset, graph, current_img_idx, keyframe_idx)
        #print(f'Visual odometry : {time() - start_time}')


        if enough_points:

            add_to_graph(transform, desc, kp, graph_size, keyframe_idx, graph, current_pose)
            current_pose = np.matmul(current_pose, transform)

            graph_size += 1


            if graph_size % 7 == 0:
                pass
                #bundle adjustment

            # visualize_path(graph)
            idx, loop_closure_found_bool = find_most_similar_image(graph_size, graph, flann) # Is not done
            if loop_closure_found_bool:
                print("Loop closure found mf")
            #     pass
            #     perform_loop_closure(idx, graph)
            #     update_visualize_path(graph)
            # prev_idx = current_img_idx
            # current_img_idx = keyframe_idx

            
            start_time = time()

            gt_path = gt_poses[current_img_idx][0, 3], gt_poses[current_img_idx][2, 3]
            gt_path_x, gt_path_y = [gt_path[0], gt_path[1]]
            gt_data_x.append(gt_path_x)
            gt_data_y.append(gt_path_y)


            esti_data_x.clear()
            esti_data_y.clear()

            edges = graph.g.edges()
            for edge in edges:
                transform = graph.e_trans[edge]
                esti_path = transform[0, 3], transform[2, 3]
                esti_path_x, esti_path_y = [esti_path[0], esti_path[1]]
                esti_data_x.append(esti_path_x)
                esti_data_y.append(esti_path_y)

            
            

            esti_line.set_data(esti_data_x, esti_data_y)
            gt_line.set_data(gt_data_x, gt_data_y)

            error_y.clear()
            for i in range(len(gt_data_x)):
                error_y.append(math.dist([gt_data_x[i], gt_data_y[i]], [esti_data_x[i], esti_data_y[i]]))
            error_x.append(graph_size)
            error_line.set_data(error_x, error_y)


            # l = matplotlib.lines.Line2D([gt_path_x, gt_path_x + 10], [gt_path_y, gt_path_y + 10])
            # ax.add_line(l)
            

            border_path = 50
            if gt_path_x + border_path > prev_max_x:
                ax[0].set_xlim(prev_min_x, gt_path_x + border_path)
                prev_max_x = gt_path_x + border_path

            if gt_path_x - border_path < prev_min_x:
                ax[0].set_xlim(gt_path_x - border_path, prev_max_x)
                prev_min_x = gt_path_x - border_path

            if gt_path_y + border_path > prev_max_y:
                ax[0].set_ylim(prev_min_y, gt_path_y + border_path)
                prev_max_y = gt_path_y + border_path

            if gt_path_y - border_path < prev_min_y:
                ax[0].set_ylim(gt_path_y - border_path, prev_max_y)
                prev_min_y = gt_path_y - border_path

            
            if esti_path_x + border_path > prev_max_x:
                ax[0].set_xlim(prev_min_x, esti_path_x + border_path)
                prev_max_x = esti_path_x + border_path

            if esti_path_x - border_path < prev_min_x:
                ax[0].set_xlim(esti_path_x - border_path, prev_max_x)
                prev_min_x = esti_path_x - border_path

            if esti_path_y + border_path > prev_max_y:
                ax[0].set_ylim(prev_min_y, esti_path_y + border_path)
                prev_max_y = esti_path_y + border_path

            if esti_path_y - border_path < prev_min_y:
                ax[0].set_ylim(esti_path_y - border_path, prev_max_y)
                prev_min_y = esti_path_y - border_path


            border_error = 10

            if graph_size + border_error > prev_max_x_error:
                ax[1].set_xlim(0, graph_size + border_error)
                prev_max_x_error = graph_size + border_error

            if max(error_y) + border_error > prev_max_y_error:
                ax[1].set_ylim(0, max(error_y) + border_error)
                prev_max_y_error = max(error_y) + border_error - 5

            #print(f'Visualizing : {time() - start_time}')


        else:
            print(enough_points)




    prev_idx = current_img_idx
    current_img_idx = keyframe_idx
    
    
    return lines

ani = FuncAnimation(fig, update, frames=gen, interval=100,
                    init_func=init, blit=False, repeat=False, save_count=num_images)
plt.show()
print("Saving animation as GIF")
writergif = PillowWriter(fps=30) 
ani.save("animation.mp4", fps=30)
print("Annimation saved")

    



















