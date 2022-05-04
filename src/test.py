from cv2 import repeat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
rcParams['animation.convert_path'] = r'/usr/local/bin/convert'

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')
i = 0
prev_max_x = 0
prev_min_x = 0
prev_max_y = 0
prev_min_y = 0

def init():
    # ax.set_xlim(0, 2*np.pi)
    # ax.set_ylim(-1, 1)
    return ln,



def update(frame):
    global i
    global prev_max_x
    global prev_max_y
    global prev_min_x
    global prev_min_y
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    # if len(xdata) % 10 == 0:
    #     xdata[i-5] = 4
    #     ydata[i-5] = -0.5
    

    if frame + 0.5 > prev_max_x:
        ax.set_xlim(prev_min_x, frame + 0.5)
        prev_max_x = frame + 0.5

    if frame - 0.5 < prev_min_x:
        ax.set_xlim(frame - 0.5, prev_max_x)
        prev_min_x = frame - 0.5

    if np.sin(frame) + 0.5 > prev_max_y:
        ax.set_ylim(prev_min_y, np.sin(frame) + 0.5)
        prev_max_y = np.sin(frame) + 0.5

    if np.sin(frame) - 0.5 < prev_min_y:
        ax.set_ylim(np.sin(frame) - 0.5, prev_max_y)
        prev_min_y = np.sin(frame) - 0.5

    i += 1
    
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=False, repeat=False, interval = 10)# save_count=1000)
ani.save("test.mp4")