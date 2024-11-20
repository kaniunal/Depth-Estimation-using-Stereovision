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

# Its all about resolution - Its a trade off between resolution and time of computation

def main():
    ## Print the current working directory
    #print("Current Working Directory:", os.getcwd())
    #number = 1
    ## Check if the images exist
    #img1_path = "Depth-Estimation\im0.png"
    #img2_path = "Depth-Estimation\im1.png"
    #
    #if not os.path.exists(img1_path):
    #    print(f"Error: {img1_path} does not exist.")
    #    return
    #if not os.path.exists(img2_path):
    #    print(f"Error: {img2_path} does not exist.")
    #    return
    #
    #img1 = cv2.imread(img1_path, 0)
    #img2 = cv2.imread(img2_path, 0)
    #
    #if img1 is None or img2 is None:
    #    print(f"Error: Could not load images from {img1_path} and/or {img2_path}")
    #    return
    #
    #width = int(img1.shape[1] * 0.3) # 0.3
    #height = int(img1.shape[0] * 0.3) # 0.3
    
    #open beide camera's
    img1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    img2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

    frame_rate = 120 #zelf te kiezen (max is 120fps)

    B = 9               #Afstand tussen de 2 camera's (cm)
    f = 6               #De lens van de camera zijn brandpunt lengte (cm)
    alpha = 49.99        #field of view van de camera (horizontale vlak [49.99 graden])

    img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
    
    #__________________Camera Parameters________________________________
    K11 = np.array([[5299.313,  0,   1263.818], 
                [0,       5299.313, 977.763],
                [0,          0,       1   ]])
    K12 = np.array([[5299.313,   0,    1438.004],
                [0,       5299.313,  977.763 ],
                [0,           0,      1     ]])

    K21 = np.array([[4396.869, 0, 1353.072],
                    [0, 4396.869, 989.702],
                    [0, 0, 1]])
    K22 = np.array([[4396.869, 0, 1538.86],
                [0, 4396.869, 989.702],
                [0, 0, 1]])
    
    K31 = np.array([[5806.559, 0, 1429.219],
                    [0, 5806.559, 993.403],
                    [ 0, 0, 1]])
    K32 = np.array([[5806.559, 0, 1543.51],
                    [ 0, 5806.559, 993.403],
                    [ 0, 0, 1]])
    camera_params = [(K11, K12), (K21, K22), (K31, K32)]

    while(1):
        try:
            list_kp1, list_kp2 = draw_keypoints_and_match(img1, img2)
            
            #_______________________________Calibration_______________________________

            F = RANSAC_F_matrix([list_kp1, list_kp2])
            print("F matrix", F)
            print("=="*20, '\n')
            K1, K2 = camera_params[number-1]
            E = calculate_E_matrix(F, K1, K2)
            print("E matrix", E)
            print("=="*20, '\n')
            camera_poses = extract_camerapose(E)
            best_camera_pose = disambiguate_camerapose(camera_poses, list_kp1)
            print("Best_Camera_Pose:")
            print("=="*20)
            print("Roatation", best_camera_pose[0])
            print()
            print("Transaltion", best_camera_pose[1])
            print("=="*20, '\n')
            pts1 = np.int32(list_kp1)
            pts2 = np.int32(list_kp2)

            #____________________________Rectification________________________________
            
            rectified_pts1, rectified_pts2, img1_rectified, img2_rectified = rectification(img1, img2, pts1, pts2, F)
            break
        except Exception as e:
            # print("error", e)
            continue
    
    # Find epilines corresponding to points in right image (second image) and drawing its lines on left image
    
    lines1 = cv2.computeCorrespondEpilines(rectified_pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1_rectified, img2_rectified, lines1, rectified_pts1, rectified_pts2)

    # Find epilines corresponding to points in left image (first image) and drawing its lines on right image

    lines2 = cv2.computeCorrespondEpilines(rectified_pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2_rectified, img1_rectified, lines2, rectified_pts2, rectified_pts1)

    cv2.imwrite("left_image.png", img5)
    cv2.imwrite("right_image.png", img3)
    
    #____________________________Correspondance________________________________
    
    disparity_map_unscaled, disparity_map_scaled = ssd_correspondence(img1_rectified, img2_rectified)
    # cv2.imwrite(f"disparity_map_{number}.png", disparity_map_scaled)

    # img_n = cv2.normalize(src=disparity_map_scaled, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # heatmap1 = cv2.applyColorMap(img_n, cv2.COLORMAP_HOT)
    # cv2.imwrite(f"disparity_heat_map_{number}.png", heatmap1)
    plt.figure(1)
    plt.title('Disparity Map Graysacle')
    plt.imshow(disparity_map_scaled, cmap='gray')
    plt.figure(2)
    plt.title('Disparity Map Hot')
    plt.imshow(disparity_map_scaled, cmap='hot')
    

    #________________________________Depth______________________________________
    baseline1, f1 = 177.288, 5299.313
    baseline2, f2 = 144.049, 4396.869
    baseline3, f3 = 174.019, 5806.559
    
    params = [(baseline1, f1), (baseline2, f2), (baseline3, f3)]
    baseline, f = params[number-1]
    depth_map, depth_array = disparity_to_depth(baseline, f, disparity_map_unscaled)
    
    plt.figure(3)
    plt.title('Depth Map Graysacle')
    plt.imshow(depth_map, cmap='gray')
    plt.figure(4)
    plt.title('Depth Map Hot')
    plt.imshow(depth_map, cmap='hot')
    plt.show()

    print("=="*20)
    # print("Depth values", depth_array)

    #____________________________________________________________________________

if __name__ == "__main__":
    main()
