import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define vehicle classes
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']

# Load videos for each lane
caps = [
    cv2.VideoCapture("lane1.mp4"),
    cv2.VideoCapture("lane2.mp4"),
    cv2.VideoCapture("lane3.mp4"),
    cv2.VideoCapture("lane4.mp4"),
]

# Traffic signal timings
green_time = 30
yellow_time = 5
total_time = green_time + yellow_time

# Signal control
current_lane = 0
last_switch_time = time.time()

# Define refined horizontal ROI bounds (y_min, y_max) for each lane
y_bounds = [
    (360, 100),
    (360, 100),
    (420, 120),
    (420, 120),
]

def get_signal_state(elapsed):
    if elapsed < green_time:
        return "green"
    elif elapsed < total_time:
        return "yellow"
    else:
        return "switch"

def draw_traffic_light(frame, state, x=10, y=40):

    colors = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
    dark = (50, 50, 50)

    states = ["red", "yellow", "green"]
    for i, s in enumerate(states):
        color = colors[s] if s == state else dark
        cx, cy = x + 20, y + i * 30
        cv2.circle(frame, (cx, cy), 10, color, -1)

while True:
    ret_flags, frames = [], []

    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            ret_flags.append(False)
        else:
            frame = cv2.resize(frame, (640, 360))
            ret_flags.append(True)
        frames.append(frame)

    if not any(ret_flags):
        break

    # Determine current signal state
    elapsed = time.time() - last_switch_time
    signal_state = get_signal_state(elapsed)

    if signal_state == "switch":
        current_lane = (current_lane + 1) % 4
        last_switch_time = time.time()
        signal_state = "green"

    processed_frames = []

    for i, frame in enumerate(frames):
        count = 0
        y_min, y_max = y_bounds[i]

        # Draw ROI lines
        cv2.line(frame, (180, y_min), (500, y_min), (255, 0, 0), 2)
        cv2.line(frame, (180, y_max), (500, y_max), (255, 0, 0), 2)

        results = model.predict(source=frame, conf=0.4, device="cpu", verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if y_min > cy > y_max:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    count += 1

        # Vehicle count display
        cv2.putText(frame, f"Count: {count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Determine light state for this lane
        if i == current_lane:
            lane_state = "green" if signal_state == "green" else "yellow"
        else:
            lane_state = "red"

        draw_traffic_light(frame, lane_state, x=10, y=40)
        processed_frames.append(frame)

    # Merge frames into a 2x2 grid
    top = np.hstack((processed_frames[0], processed_frames[1]))
    bottom = np.hstack((processed_frames[2], processed_frames[3]))
    final_display = np.vstack((top, bottom))

    cv2.namedWindow("Traffic Signal Simulation", cv2.WINDOW_NORMAL)
   
    cv2.imshow("Traffic Signal Simulation", final_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
