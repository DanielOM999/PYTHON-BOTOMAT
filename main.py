import cv2
import torch
import easyocr
import tkinter as tk
from tkinter import ttk
import warnings
import time
import math
from collections import Counter, defaultdict

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# Tkinter setup for vase lock control
root = tk.Tk()
root.withdraw()
lock_window = tk.Toplevel(root)
lock_window.title("Vase Lock Control")

# Globals
locked_positions = []
latest_vase_positions = []
KNOWN_DISTANCE = 10  # meters between vases
CAR_SPEEDS = []

# Enhanced sign recognition data structure
class SignData:
    def __init__(self):
        self.letters = []  # store recognized letters
        self.numbers = []  # store recognized numbers
        self.last_seen = time.time()
        self.final_result_printed = False

sign_histories = defaultdict(SignData)  # car_id -> SignData
active_cars = set()  # track which cars are currently visible

# Speed & timing
def current_time():
    return time.time()

class CarTimer:
    def __init__(self):
        self.entry_times = {}  # vase_index: timestamp
        self.last_position = None
        self.last_update = current_time()

car_timers = {}  # car_id -> CarTimer

def extract_letters_numbers(text):
    """Extract letters and numbers from OCR text"""
    letters = ''.join(c for c in text if c.isalpha()).upper()
    numbers = ''.join(c for c in text if c.isdigit())
    return letters, numbers

def build_norwegian_sign(sign_data):
    """Build Norwegian sign from accumulated letters and numbers"""
    if not sign_data.letters and not sign_data.numbers:
        return None
    
    # Find most common letters (should be 2-3 characters)
    from collections import Counter
    letter_counter = Counter(sign_data.letters)
    number_counter = Counter(sign_data.numbers)
    
    # Build letters part (most common sequence)
    best_letters = ""
    if letter_counter:
        # Try to find 2-3 letter combinations
        all_letters = ''.join(sign_data.letters)
        for length in [2, 3]:
            for i in range(len(all_letters) - length + 1):
                candidate = all_letters[i:i+length]
                if len(set(candidate)) >= 2:  # at least 2 different letters
                    best_letters = candidate
                    break
            if best_letters:
                break
        
        # Fallback to most common letters
        if not best_letters:
            sorted_letters = letter_counter.most_common()
            best_letters = ''.join([letter for letter, count in sorted_letters[:2]])
    
    # Build numbers part (should be 5 digits)
    best_numbers = ""
    if number_counter:
        all_numbers = ''.join(sign_data.numbers)
        # Look for 5-digit sequences
        for i in range(len(all_numbers) - 4):
            candidate = all_numbers[i:i+5]
            if len(set(candidate)) >= 3:  # at least 3 different digits
                best_numbers = candidate
                break
        
        # Fallback to most common numbers
        if not best_numbers:
            sorted_numbers = number_counter.most_common()
            best_numbers = ''.join([num for num, count in sorted_numbers[:5]])
    
    # Combine if we have both parts
    if best_letters and best_numbers and len(best_numbers) >= 4:
        return f"{best_letters} {best_numbers}"
    
    return None

def cleanup_old_cars():
    """Clean up cars that are no longer visible and print their final sign results"""
    current_time_val = current_time()
    cars_to_process = []
    
    # Find cars that are no longer active
    for car_id, sign_data in sign_histories.items():
        if car_id not in active_cars and not sign_data.final_result_printed:
            time_since_seen = current_time_val - sign_data.last_seen
            if time_since_seen > 1.0:  # Wait 1 second after car disappears
                cars_to_process.append(car_id)
    
    # Process cars that should get final results
    for car_id in cars_to_process:
        sign_data = sign_histories[car_id]
        final_sign = build_norwegian_sign(sign_data)
        
        if final_sign:
            print(f"FINAL SIGN RECOGNITION - Car {car_id}: '{final_sign}'")
        
        sign_data.final_result_printed = True
    
    # Clean up very old entries
    cars_to_remove = []
    for car_id, sign_data in sign_histories.items():
        time_since_seen = current_time_val - sign_data.last_seen
        if time_since_seen > 10.0:
            cars_to_remove.append(car_id)
    
    for car_id in cars_to_remove:
        del sign_histories[car_id]

# Lock control callbacks
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
        print(f"{idx}: {x1}, {y1}, {x2}, {y2}")

# Build lock control UI
ttk.Button(lock_window, text="Lock Current Vases", command=lock_current).pack(padx=20, pady=5)
ttk.Button(lock_window, text="Clear All Locks", command=clear_locks).pack(padx=20, pady=5)

# Load YOLOv5 for vehicles + vases
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.25
model.classes = [2, 3, 5, 7, 75]  # car, motorcycle, bus, truck, vase

# OCR reader
reader = easyocr.Reader(['en', 'no'], gpu=False)

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()
cv2.namedWindow('Detection System', cv2.WINDOW_NORMAL)

# Helpers
def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_point_in_rect(pt, rect):
    x, y = pt
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= x <= rx2 and ry1 <= y <= ry2

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]
    latest_vase_positions.clear()
    current_cars = {}
    active_cars.clear()  # Reset active cars for this frame

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        label = model.names[cls]

        # Vase detection
        if cls == 75:
            latest_vase_positions.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Vehicle detection + OCR + speed
        elif cls in [2, 3, 5, 7]:
            # draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            center = get_center(x1, y1, x2, y2)
            car_id = f"{center[0]}_{center[1]}"
            current_cars[car_id] = center
            active_cars.add(car_id)  # Mark this car as active

            # Update last seen time for this car
            sign_histories[car_id].last_seen = current_time()

            # OCR region
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                ocr_results = reader.readtext(crop)
                for bbox, text, confidence in ocr_results:
                    if confidence > 0.5:  # Higher confidence threshold
                        letters, numbers = extract_letters_numbers(text)
                        if letters:
                            sign_histories[car_id].letters.extend(letters)
                        if numbers:
                            sign_histories[car_id].numbers.extend(numbers)
                    
                    # Draw OCR bounding box
                    tl = (int(bbox[0][0]) + x1, int(bbox[0][1]) + y1)
                    br = (int(bbox[2][0]) + x1, int(bbox[2][1]) + y1)
                    cv2.rectangle(frame, tl, br, (0, 0, 255), 2)

            # Show live recognition status
            total_letters = len(sign_histories[car_id].letters)
            total_numbers = len(sign_histories[car_id].numbers)
            if total_letters > 0 or total_numbers > 0:
                cv2.putText(frame, f"L:{total_letters} N:{total_numbers}", (center[0], center[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Clean up old cars and print final results
    cleanup_old_cars()

    # Speed calc if vases locked - FIXED VERSION
    if len(locked_positions) >= 2:
        for car_id, center in current_cars.items():
            # Find closest existing timer (improved matching)
            closest_timer_id = None
            min_distance = float('inf')
            
            for existing_id, timer in car_timers.items():
                if timer.last_position:
                    dx = center[0] - timer.last_position[0]
                    dy = center[1] - timer.last_position[1]
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance < min_distance and distance < 50:  # 50px threshold
                        min_distance = distance
                        closest_timer_id = existing_id
            
            # Update or create timer
            if closest_timer_id:
                timer = car_timers.pop(closest_timer_id)  # Remove old key
                timer.last_position = center
                timer.last_update = current_time()
            else:
                timer = CarTimer()
                timer.last_position = center
                timer.last_update = current_time()
            
            car_timers[car_id] = timer  # Assign with current car_id

            # Check vase intersections for first 2 locked vases only
            for idx, vase in enumerate(locked_positions[:2]):
                if is_point_in_rect(center, vase) and idx not in timer.entry_times:
                    timer.entry_times[idx] = current_time()
                    print(f"Car {car_id} entered vase {idx+1}")

            # Calculate speed if passed both vases
            if len(timer.entry_times) == 2:
                times = sorted(timer.entry_times.values())
                time_diff = times[1] - times[0]
                if time_diff > 0:
                    speed_mps = KNOWN_DISTANCE / time_diff
                    speed_kmh = speed_mps * 3.6
                    CAR_SPEEDS.append(speed_kmh)
                    cv2.putText(frame, f"{speed_kmh:.1f} km/h", (center[0] - 20, center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print(f"Speed: {speed_kmh:.1f} km/h")
                # Remove timer after calculation
                del car_timers[car_id]

    # cleanup old timers
    now = current_time()
    for cid in list(car_timers):
        if now - car_timers[cid].last_update > 5:
            del car_timers[cid]

    # draw locked vases
    for i, v in enumerate(locked_positions):
        x1, y1, x2, y2 = v
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Vase {i+1} (Locked)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # toggle lock window
    if latest_vase_positions or locked_positions:
        lock_window.deiconify()
    else:
        lock_window.withdraw()

    cv2.imshow('Detection System', frame)
    root.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
lock_window.destroy()
root.destroy()