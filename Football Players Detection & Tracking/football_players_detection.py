from ultralytics import YOLO
import cv2

model    = YOLO('models/best.pt')
cap      = cv2.VideoCapture('vidieo/vid2.mp4')

while True:
    ret , frame = cap.read()
    
    results = model.track(frame)
    
    detection = results[0].plot()
          
    cv2.imshow('Detection',detection)
    if cv2.waitKey(3) & 0xff == ord('q'):
        break