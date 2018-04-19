import cv2

image = cv2.imread('poker/HQ.png')
# Find perimeter of card and use it to approximate corner points
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, image = cv2.threshold(image,255,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
image = cv2.resize(image,(200,300))
cv2.imshow('resized',image)
Qcorner = image[0:84, 0:34]
Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)
Qrank = Qcorner_zoom[20:185, 0:136]

cv2.imshow('rank',Qrank)
Qsuit = Qcorner_zoom[186:336, 0:136]
cv2.imshow('suit',Qsuit)

cv2.waitKey(0)