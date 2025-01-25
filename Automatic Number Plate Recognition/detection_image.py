from ultralytics import YOLO
import cv2

model  = YOLO('models/best.pt')

img    = cv2.imread('image/img1.jpeg')

resize = cv2.resize(img, (600,700))

result = model(resize)

detect = result[0].plot()

cv2.imshow('detection',detect)
cv2.waitKey(0)