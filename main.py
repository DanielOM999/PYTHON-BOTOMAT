import cv2
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.5  # Confidence threshold
model.classes = [2, 3, 5, 7]  # Filter: cars, bikes, buses, trucks

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

# Create window ONCE (outside loop)
cv2.namedWindow('Car Detector', cv2.WINDOW_NORMAL)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break

    # Detect objects (YOLOv5)
    results = model(frame)
    detections = results.xyxy[0]  # x1, y1, x2, y2, conf, cls

    # Draw bounding boxes
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Car Detector', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
