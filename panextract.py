import easyocr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import re

reader = easyocr.Reader(['en'])
model_path = "pan.pt"
model = YOLO(model_path)
model.overrides['conf'] = 0.25
model.overrides['iou'] = 0.45
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = 1000
image_path = 'pan.jpg'
results = model.predict(image_path)

def extract_text_from_bbox(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray, detail=0)
    return " ".join(result) if result else None

def clean_extracted_text(text, class_name):
    if text is None:
        return None
    text = text.strip()

    if class_name == "name":
        text = re.sub(r'^\d+x\s*Name\s*', '', text, flags=re.IGNORECASE)
    elif class_name == "dob":
        text = re.sub(r'[^0-9/]', '', text)
    elif class_name == "pan_number":
        return text
    elif class_name == "father-s name":
        text = re.sub(r'^\d+x\s*Father\'?s?\s*Name\s*', '', text, flags=re.IGNORECASE)

    return text.strip() if text else None

extracted_data = {"Name": None, "DOB": None, "PAN Number": None, "Father's Name": None}
highest_confidence_detections = {
    "name": {"confidence": 0, "text": None},
    "dob": {"confidence": 0, "text": None},
    "pan_number": {"confidence": 0, "text": None},
    "father-s name": {"confidence": 0, "text": None},
}
cleaned_pan_number = None

for result in results:
    boxes = result.boxes

    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            img = Image.open(image_path)
            img_np = np.array(img)

            extracted_text = extract_text_from_bbox(img_np, box.xyxy[0])
            cleaned_text = clean_extracted_text(extracted_text, class_name)

            if class_name in highest_confidence_detections:
                if confidence > highest_confidence_detections[class_name]["confidence"]:
                    highest_confidence_detections[class_name]["confidence"] = confidence
                    highest_confidence_detections[class_name]["text"] = cleaned_text

                    if class_name == "pan_number":
                        cleaned_pan_number = cleaned_text

            print(f"Class: {class_name}, Confidence: {confidence:.2f}")
            print(f"Extracted Text: {extracted_text}")
            print(f"Cleaned Text: {cleaned_text}")
            print(f"Bounding Box: [{x1}, {y1}, {x2}, {y2}]\n")

    else:
        print("No PAN cards detected in the image.")

extracted_data["Name"] = highest_confidence_detections["name"]["text"]
extracted_data["DOB"] = highest_confidence_detections["dob"]["text"]
extracted_data["PAN Number"] = highest_confidence_detections["pan_number"]["text"]
extracted_data["Father's Name"] = highest_confidence_detections["father-s name"]["text"]

print("\nExtracted PAN Card Information:")
print(f"Name: {extracted_data['Name']}")
print(f"DOB: {extracted_data['DOB']}")
print(f"PAN Number: {cleaned_pan_number}")
print("Father's Name:", extracted_data["Father's Name"])
print("Inference and processing complete.")