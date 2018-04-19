
import cv2
import os
import numpy as np

for cards in os.listdir("D:/tf-resnet/test"):
    print(cards)
    dirname = cards
    path = os.path.join("D:/tf-resnet/test",cards)
    I = 0
    for pics in os.listdir(path):
        img_path = os.path.join(path,pics)
        print(pics)
        image = cv2.imread(img_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 120])
        upper_white = np.array([179, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        if not os.path.exists("D:/test/"+dirname):
            os.makedirs("D:/test/"+dirname)
        cv2.imwrite("D:/test/"+dirname+"/"+pics,mask)