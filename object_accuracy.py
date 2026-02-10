from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

conf_threshold = 0.70 # confidence threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame, conf=0.7)[0]   # get detection result
    cellphone_count = 0

    # count person with confidence > 0.7
    for b in r.boxes:
        cls_id = int(b.cls[0])  # check if class_id is 67 (cellphone)
        conf = float(b.conf[0])

        if cls_id == 67 and conf > conf_threshold:  # cellphone class
            cellphone_count += 1

    annotated_frame = r.plot()
    annotated_frame = cv2.resize(annotated_frame, (1040, 720))

    cv2.putText(annotated_frame,
                f'CellPhone Count (conf > 0.7): {cellphone_count}',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2)

    cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()