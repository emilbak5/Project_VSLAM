import cv2
import numpy as np

def get_descripters(idx, dataset, orb):
    img = np.array(dataset.get_cam0(idx))

    kp, des = orb.detectAndCompute(img, None)

    return kp, des


