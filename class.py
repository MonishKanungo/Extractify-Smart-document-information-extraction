import cv2
import re
from ultralytics import YOLO

model = YOLO("best.pt")
image_path = "test_aadhar.jpeg"
image = cv2.imread(image_path)

results = model(image)

for r in results:
    for box in r.boxes:
        confidence = box.conf[0].item()
        class_id = int(box.cls[0])
        class_name = r.names[class_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_image = image[y1:y2, x1:x2]

        print(f"🔍 Detected Document: {class_name} (Confidence: {confidence:.2f})")

if len(results) == 0:
    print(" No document detected. Try a clearer image.")
