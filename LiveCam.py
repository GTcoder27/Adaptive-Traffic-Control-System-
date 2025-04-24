import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Vehicle classes: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# ✅ Correct IP camera stream URL
ip_cam_url = 'http://192.0.0.4:8080/video'

# Open the video stream
cap = cv2.VideoCapture(ip_cam_url)

if not cap.isOpened():
    print("❌ Error: Could not open video stream")
    exit()

print("✅ Connected to phone camera!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Failed to grab frame")
        break

    # Run YOLOv8 detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0].item()
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.model.names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ✅ Resize and show
    resized_frame = cv2.resize(frame, (640, 360))  # smaller window size
    cv2.imshow("🚗 Live YOLOv8 Vehicle Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()