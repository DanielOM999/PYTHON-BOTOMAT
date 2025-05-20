import cv2
import torch
import easyocr
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# Load YOLOv5 model for vehicle detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.5  # confidence threshold
model.classes = [2, 3, 5, 7]  # classes: car(2), motorcycle(3), bus(5), truck(7)

# Initialize EasyOCR reader for sign/plate text detection
reader = easyocr.Reader(['en', 'no'], gpu=False)  # English + Norwegian

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

cv2.namedWindow('Vehicle & Sign Detector', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break

    # Run YOLOv5 detection on the frame
    results = model(frame)
    detections = results.xyxy[0]  # x1, y1, x2, y2, conf, cls

    # Prepare list to store all sign detections
    signs_data = []

    # Process each vehicle detection
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"

        # Draw vehicle bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Crop the detected vehicle region
        cropped_car = frame[y1:y2, x1:x2]
        if cropped_car.size == 0:
            continue

        # Run OCR on the cropped car image
        ocr_results = reader.readtext(cropped_car)
        for bbox, text, prob in ocr_results:
            # Convert bbox to integer coordinates relative to cropped image
            tl, tr, br, bl = bbox
            tl = (int(tl[0]) + x1, int(tl[1]) + y1)
            br = (int(br[0]) + x1, int(br[1]) + y1)

            # Store detected sign data
            signs_data.append({
                'text': text,
                'confidence': prob,
                'bbox': (tl, br)
            })

            # Draw bounding box for the sign on the original frame
            cv2.rectangle(frame, tl, br, (0, 0, 255), 2)
            cv2.putText(frame, f"{text} {prob:.2f}", (tl[0], tl[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display detected sign information in console
    if signs_data:
        print("Detected signs in current frame:")
        for sign in signs_data:
            print(f"Text: {sign['text']}, Confidence: {sign['confidence']:.2f}, BBox: {sign['bbox']}")

    # Show result
    cv2.imshow('Vehicle & Sign Detector', frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
