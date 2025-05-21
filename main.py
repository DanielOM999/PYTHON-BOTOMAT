import cv2
import torch
import tkinter as tk
from tkinter import ttk
import warnings
import time
import math

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# Initialize Tkinter
root = tk.Tk()
root.withdraw()

# Create lock control window
lock_window = tk.Toplevel(root)
lock_window.title("Vase Lock Control")

# Global variables
locked_positions = []
latest_vase_positions = []
KNOWN_DISTANCE = 10  # 10 meters between vases
CAR_SPEEDS = []

# Speed calculation variables
class CarTimer:
    def __init__(self):
        self.entry_times = {}  # vase_index: timestamp
        self.last_position = None
        self.last_update = time.time()  # Add timestamp for timeout tracking

car_timers = {}  # car_id: CarTimer

# Lock buttons callbacks (same as before)
def lock_current():
    for pos in latest_vase_positions:
        if pos not in locked_positions:
            locked_positions.append(pos)
    print_locked_positions()

def clear_locks():
    locked_positions.clear()
    CAR_SPEEDS.clear()
    print("All locks cleared.")

def print_locked_positions():
    print("Locked vase positions:")
    for idx, (x1, y1, x2, y2) in enumerate(locked_positions, 1):
        print(f"{idx}: {x1, y1, x2, y2}")

# Create lock buttons
lock_button = ttk.Button(lock_window, text="Lock Current Vases", command=lock_current)
lock_button.pack(padx=20, pady=5)
clear_button = ttk.Button(lock_window, text="Clear All Locks", command=clear_locks)
clear_button.pack(padx=20, pady=5)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.25  # Set fixed confidence threshold
model.classes = [2, 3, 75]

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

cv2.namedWindow('Car-Speed-Detector', cv2.WINDOW_NORMAL)

def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_point_in_rect(point, rect):
    x, y = point
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= x <= rx2 and ry1 <= y <= ry2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break

    # Detect objects
    results = model(frame)
    detections = results.xyxy[0]

    # Process detections
    latest_vase_positions.clear()
    current_cars = {}
    
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        label_name = model.names[class_id]
        
        if class_id in [2, 3]:  # Vehicles
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{label_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Track cars for speed calculation
            car_center = get_center(x1, y1, x2, y2)
            car_id = f"{car_center[0]}_{car_center[1]}"
            current_cars[car_id] = car_center

        elif class_id == 75:  # Vases
            latest_vase_positions.append((x1, y1, x2, y2))
            # Draw detected vases in blue before locking
            color = (255, 0, 0)  # Correct blue color in BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{label_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Speed calculation logic
    if len(locked_positions) >= 2:
        # Match cars with previous timers
        for car_id, center in current_cars.items():
            # Find closest existing timer
            closest = None
            min_dist = float('inf')
            for existing_id, timer in car_timers.items():
                if timer.last_position:
                    dx = center[0] - timer.last_position[0]
                    dy = center[1] - timer.last_position[1]
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance < min_dist and distance < 50:  # 50px threshold
                        min_dist = distance
                        closest = existing_id
            
            # Update or create timer
            if closest and closest in car_timers:
                timer = car_timers[closest]
                timer.last_position = center
                timer.last_update = time.time()  # Update timestamp on position update
                del car_timers[closest]
                car_timers[car_id] = timer
            else:
                car_timers[car_id] = CarTimer()
                car_timers[car_id].last_position = center
                car_timers[car_id].last_update = time.time()  # Initialize timestamp

            # Check vase intersections
            timer = car_timers[car_id]
            for idx, vase in enumerate(locked_positions[:2]):  # Only use first 2 vases
                if is_point_in_rect(center, vase):
                    if idx not in timer.entry_times:
                        timer.entry_times[idx] = time.time()
                        print(f"Car {car_id} entered vase {idx+1}")

            # Calculate speed if passed both vases (moved inside car loop)
            if len(timer.entry_times) == 2:
                times = sorted(timer.entry_times.values())
                time_diff = times[1] - times[0]
                if time_diff > 0:
                    speed_mps = KNOWN_DISTANCE / time_diff
                    speed_kmh = speed_mps * 3.6
                    CAR_SPEEDS.append(speed_kmh)
                    print(f"Speed: {speed_kmh:.1f} km/h")
                    label = f"{speed_kmh:.1f} km/h"
                    cv2.putText(frame, label, (center[0]-20, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                del car_timers[car_id]

    # Cleanup old car timers to prevent memory bloat
    current_time = time.time()
    for car_id in list(car_timers.keys()):
        if current_time - car_timers[car_id].last_update > 5:  # 5 seconds timeout
            del car_timers[car_id]

    # Draw locked vase boxes
    for idx, (x1, y1, x2, y2) in enumerate(locked_positions):
        color = (0, 0, 255)  # Red color for all locked vases
        label = f"Vase {idx+1} (Locked)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update lock window
    if latest_vase_positions or locked_positions:
        lock_window.deiconify()
    else:
        lock_window.withdraw()

    cv2.imshow('Car-Speed-Detector', frame)
    root.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
lock_window.destroy()
root.destroy()