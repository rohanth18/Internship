from ultralytics import YOLO
import cv2, os, requests
from datetime import datetime

FIREBASE_URL = 'https://internship-1e19c-default-rtdb.firebaseio.com/'  # Replace with your Firebase URL

model, cap = YOLO('yolov8n.pt'), cv2.VideoCapture(0)

os.makedirs('snapshots', exist_ok=True)

last_save = 0
last_firebase_update = 0


def send_to_firebase(bottle_detected, bottle_count):
    try:
        data = {
            'bottle_detected': 1 if bottle_detected else 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'bottle_count': bottle_count
        }

        response = requests.put(f"{FIREBASE_URL}/bottle_detection.json", json=data)

        if response.status_code == 200:
            print(f"Firebase updated: bottle_detected = {1 if bottle_detected else 0}")
            print(f"Bottle count: {bottle_count}")
        else:
            print(f"Firebase error {response.status_code}")

    except Exception as e:
        print(f"Error sending to Firebase: {e}")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame)[0]

    # 🔹 Bottle class_id = 39 (COCO dataset)
    bottle_count = sum(int(b.cls[0]) == 39 for b in r.boxes)

    frame = r.plot()

    cv2.putText(frame,
                f'Bottles: {bottle_count}',
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    current_time = datetime.now().timestamp()

    # 🔹 Send to Firebase every 5 seconds
    if bottle_count > 0 and (current_time - last_firebase_update) >= 5:
        send_to_firebase(True, bottle_count)
        last_firebase_update = current_time

    # 🔹 Save snapshot every 20 seconds if bottles > 2
    if bottle_count > 2 and (current_time - last_save) >= 20:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cv2.rectangle(frame, (0, 0), (450, 60), (255, 0, 0), 2)
        cv2.putText(frame,
                    f"{timestamp} | Bottles: {bottle_count}",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imwrite(f'snapshots/snapshot_{timestamp}.jpg', frame)
        print("Saved snapshot")

        last_save = current_time

    cv2.imshow('Bottle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When closing program
send_to_firebase(False, 0)

cap.release()
cv2.destroyAllWindows()
