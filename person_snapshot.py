from ultralytics import YOLO
import cv2, os
from datetime import datetime

model, cap = YOLO('yolov8n.pt'), cv2.VideoCapture(0)
os.makedirs('snapshots', exist_ok=True)
last_save=0
confidence_threshold = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame)[0]
    person_count = sum(int(b.cls[0]) == 0  for b in r.boxes)  # person class_id=0
    frame = r.plot()
    cv2.putText(frame, f'Person Count: {person_count}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    if person_count > 2 and person_count<11 and (datetime.now().timestamp() - last_save) >= 3:  # Save snapshot every 3 seconds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.rectangle(frame,(0,0),(450,60),(255,0,0),2)
        cv2.putText(frame,f"{timestamp} | Persons: {person_count}",(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imwrite(f'snapshots/snapshot_{timestamp}.jpg', frame)
        last_save = datetime.now().timestamp()

    cv2.imshow('Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()