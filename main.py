import cv2
import easyocr
import matplotlib.pyplot as plt

# Load image
image_path = 'blurr.png'
image = cv2.imread(image_path)

# Initialize EasyOCR
reader = easyocr.Reader(['no'])  # Add 'no' for Norwegian if needed

# Run OCR on image
results = reader.readtext(image)

# Filter and display license plate-like text
for (bbox, text, prob) in results:
    print(f"Detected Text: {text} (Confidence: {prob:.2f})")
    
    # Draw bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Show image with plate highlighted
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()