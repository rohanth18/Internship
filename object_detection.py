from ultralytics import YOLO
import cv2

model=YOLO('yolov8n.pt')

cap=cv2.VideoCapture(0)
"""cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    'YOLOv8 Detection',
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)"""

while True:
    ret,frame=cap.read()
    if not ret:
        break   

    r = model(frame)[0]

    # bottle class_id=39
    count = sum(int(b.cls[0]) == 39 for b in r.boxes)
    count1 = sum(int(b.cls[0]) == 67 for b in r.boxes)
    frame = r.plot()
    cv2.putText(frame, f'Bottle Count: {count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Cellphone Count: {count1}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

     # cellphone class_id=67
    

"""frame = r.plot()
    cv2.putText(frame, f'Cellphone Count: {count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Cellphone Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break"""

"""  results=model(frame)

    annotated_frame=results[0].plot()
    annotated_frame=cv2.resize(annotated_frame,(1040,720))

    cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break"""

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()