import cv2
import easyocr
import numpy as np
from inference_sdk import InferenceHTTPClient
import json

reader = easyocr.Reader(['en'])
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="sFM2tWZMm4xZoYxI46WJ"
)

image_path = "test_aadhar.jpeg"
image = cv2.imread(image_path)

response = CLIENT.infer(image_path, model_id="annotation-xsy5d/3")

print("API Response:\n", json.dumps(response, indent=4))

def extract_text_from_bbox(image, bbox):
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
    x1, y1, x2, y2 = max(0, x - w//2), max(0, y - h//2), min(image.shape[1], x + w//2), min(image.shape[0], y + h//2)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=0)
    return " ".join(results) if results else None

def parse_annotations(api_response, image):
    extracted_data = {"Name": None, "DOB": None, "Aadhar Number": None}

    for prediction in api_response.get("predictions", []):
        label = prediction["class"]
        extracted_text = extract_text_from_bbox(image, prediction)

        if label == "name":
            extracted_data["Name"] = extracted_text
        elif label == "dob":
            extracted_data["DOB"] = extracted_text
        elif label == "aadharNumber":
            extracted_data["Aadhar Number"] = extracted_text

    return extracted_data

parsed_info = parse_annotations(response, image)

print("\nExtracted Aadhaar Information:")
print(json.dumps(parsed_info, indent=4))