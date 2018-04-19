
import cv2

im = cv2.imread('randtest.jpg')

im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret, im = cv2.threshold(im,100,255,cv2.THRESH_OTSU)
im = cv2.Canny(im, 5, 10)

cv2.imshow('canny',im)
cv2.waitKey(0)