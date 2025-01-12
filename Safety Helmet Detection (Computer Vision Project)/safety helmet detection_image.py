from ultralytics import YOLO
import cv2

model = YOLO('models/best.pt')

img = cv2.imread('images/img1.jpg')

result = model(img)

detect = result[0].plot()

cv2.imshow('detection',detect)
cv2.waitKey(0)