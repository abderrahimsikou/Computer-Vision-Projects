from ultralytics import YOLO
import cv2
from playsound import playsound

model    = YOLO('model/best.pt')
                
def detect(path):
    cap = cv2.VideoCapture(path)
    while True:
        ret , frame = cap.read()  
        if not ret:
            break
    
        results   = model(frame)
        for result in results:
            for box in result.boxes: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf           = box.conf[0]
                label          = model.names[int(box.cls[0])]
                
                cv2.rectangle(frame, (x1, y1),(x2, y2), (255,0,0),3)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0), 3)
                
                if label == 'fire':
                    playsound('audio.mp3')
        
        cv2.imshow('YOLO Player Detection', frame)
        if cv2.waitKey(2) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect('vidieo/vid1.mp4')