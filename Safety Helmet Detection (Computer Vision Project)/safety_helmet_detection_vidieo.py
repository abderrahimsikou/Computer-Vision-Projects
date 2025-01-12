from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('vidieo/vid1.mp4')

model = YOLO('models/best.pt')

while True:
    _ , frame = cap.read()
    
    result = model(frame)

    detect = result[0].plot()
    
    cv2.imshow('Safety Helmet Detection', detect)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break