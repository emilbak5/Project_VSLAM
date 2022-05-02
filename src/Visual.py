import numpy as np
import cv2

# function to display the coordinates of
# of the points clicked on the image

# def click_event(event, x, y, flags, params):
# # checking for left mouse clicks
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, ' ', y)

# # path
# path_l = r'test\img\000000_l.png'
# path_r = r'test\img\000000_r.png'	
# img_l = cv2.imread(path_l)
# img_r = cv2.imread(path_r)

# cv2.imshow("image_l", img_l)
# cv2.setMouseCallback('image_l', click_event)
# # wait for a key to be pressed to exit
# cv2.waitKey(0)

# cv2.imshow("image_r", img_r)
# cv2.setMouseCallback('image_r', click_event)
# # wait for a key to be pressed to exit
# cv2.waitKey(0)
# # close the window
# cv2.destroyAllWindows()

kp1 = np.array([561, 341], dtype=np.float64)
kp2 = np.array([505, 341], dtype=np.float64)
#kp1 = np.array([505, 341], dtype=np.float64)
#kp2 = np.array([561, 341], dtype=np.float64)
K = np.array([[718.856,0,607.1928],[0,718.856,185.2157],[0,0,1]], dtype=np.float64)
Pcal=np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00], [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]], dtype=np.float64)
Pcal2=np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02], [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]],dtype=np.float64)
T1 = np.array([[1.000000e+00, 9.043680e-12, 2.326809e-11, 5.551115e-17], [9.043683e-12, 1.000000e+00, 2.392370e-10, 3.330669e-16], [2.326810e-11, 2.392370e-10, 9.999999e-01, -4.440892e-16]], dtype=np.float64)
T2 = np.array([[1.000000e+00, 9.043680e-12, 2.326809e-11, 5.551115e-17 - 0.54], [9.043683e-12, 1.000000e+00, 2.392370e-10, 3.330669e-16], [2.326810e-11, 2.392370e-10, 9.999999e-01, -4.440892e-16]], dtype=np.float64)
#T2 = np.array([[9.999978e-01, 5.272628e-04, -2.066935e-03, -4.690294e-02], [-5.296506e-04, 9.999992e-01, -1.154865e-03, -2.839928e-02], [2.066324e-03, 1.155958e-03, 9.999971e-01, 8.586941e-01
#]], dtype=np.float64)
#T1 = np.eye(4)
#T1[0, 3] = Pcal[0, 3] / Pcal[0, 0]
#T2 = np.eye(4)
#T2[0, 3] = Pcal2[0, 3] / Pcal2[0, 0]
#print(T1)
#print(T2)
P1 = np.matmul(K,T1)
P2 = np.matmul(K,T2)


M = cv2.triangulatePoints(P1, P2, kp1, kp2)
print(M)
M = M[:4]/M[3]
print("K: \n {0}\n...\n".format(K))
print("T1: \n {0}\n...\n".format(T1))
print("T2: \n {0}\n...\n".format(T2))
print("P1: \n {0}\n...\n".format(P1))
print("P2: \n {0}\n...\n".format(P2))
print("M: \n {0}\n...\n".format(M))

# Python program to explain cv2.circle() method


# path
path1="./test/img/000000_l.png"
path2="./test/img/000000_r.png"
#path1 = r'test\img\000000_l.png'
#path2 = r'.\test\img\000000_r.png'

# Reading an image in default mode
image1 = cv2.imread(path1)
image2 = cv2.imread(path2)


# Window name in which image is displayed
window_name1 = 'Image1'
window_name2 = 'Image2'

# Center coordinates
center_coordinates1 = (561, 341)
center_coordinates2 = (505, 341)

# Radius of circle
radius = 5

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 1

# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
image1 = cv2.circle(image1, center_coordinates1, radius, color, thickness)
image2 = cv2.circle(image2, center_coordinates2, radius, color, thickness)

# Displaying the image
cv2.imshow(window_name1, image1)
cv2.imshow(window_name2, image2)
cv2.waitKey(0)