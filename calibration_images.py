import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

num = 0

while cap.isOpened():
    success1, img = cap.read()
    success2, img2 = cap2.read()
    
    k = cv2.waitKey(5)
    
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite("left_image" + str(num) +  ".png", img)
        cv2.imwrite("right_image" + str(num) +  ".png", img2)
        print("Images saved")
        num += 1
        
    cv2.imshow("Camera 1", img)
    cv2.imshow("Camera 2", img2)
    