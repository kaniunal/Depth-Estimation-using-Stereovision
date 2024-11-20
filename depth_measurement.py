import math
import os
import cv2
import random as rd
import numpy as np  
import matplotlib.pyplot as plt

from calibration import draw_keypoints_and_match, drawlines, RANSAC_F_matrix, calculate_E_matrix, extract_camerapose, disambiguate_camerapose
from rectification import rectification
from correspondence import ssd_correspondence
from depth import disparity_to_depth


img1_path = "Depth-Estimation/im0.png"
img2_path = "Depth-Estimation/im1.png"

img1 = cv2.imread(img1_path, 0)
img2 = cv2.imread(img2_path, 0)

width = int(img1.shape[1] * 0.3) # 0.3
height = int(img1.shape[0] * 0.3) # 0.3

img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)

cv2.imwrite("left_image.png", img1)
cv2.imwrite("right_image.png", img2)

## Display the images in popup windows
#cv2.imshow("Left Image", img1)
#cv2.imshow("Right Image", img2)
#
## Wait for a key press and close the windows
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Initiate ORB detector
orb = cv2.ORB_create()
 
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
 
# Match descriptors.
matches = bf.match(des1,des2)
 
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
 
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3),plt.show()
