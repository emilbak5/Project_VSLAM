import cv2





def get_next_keyframe(dataset, i, graph):

    img_l = dataset.get_cam0(i)


    if i == 0:
        orb = cv2.ORB_create(500)
        keypoints, des1 = orb.detectAndCompute(img_l, None)

