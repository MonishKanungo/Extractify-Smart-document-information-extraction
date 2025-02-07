import cv2
import easyocr
import numpy as np
import json
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
import gradio as gr


reader = easyocr.Reader(['en'])


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="sFM2tWZMm4xZoYxI46WJ"  
)


document_model = YOLO("best.pt")  
pan_model = YOLO("pan.pt")  


def detect_document_type(image_path):
    image = cv2.imread(image_path)
    results = document_model(image)
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            return class_name  
    
    return "Unknown"  


def extract_text_from_bbox(image, bbox):
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
    x1, y1, x2, y2 = max(0, x - w//2), max(0, y - h//2), min(image.shape[1], x + w//2), min(image.shape[0], y + h//2)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=0)
    return " ".join(results) if results else None


def extract_aadhaar_info(image_path):
    image = cv2.imread(image_path)
    response = CLIENT.infer(image_path, model_id="annotation-xsy5d/3")
    
    extracted_data = {"Name": None, "DOB": None, "Aadhar Number": None}
    
    for prediction in response.get("predictions", []):
        label = prediction["class"]
        extracted_text = extract_text_from_bbox(image, prediction)
        
        if label == "name":
            extracted_data["Name"] = extracted_text
        elif label == "dob":
            extracted_data["DOB"] = extracted_text
        elif label == "aadharNumber":
            extracted_data["Aadhar Number"] = extracted_text
    
    return extracted_data


def extract_pan_info(image_path):
    image = cv2.imread(image_path)
    results = pan_model.predict(image_path)
    
    extracted_data = {"Name": None, "DOB": None, "PAN Number": None, "Father's Name": None}
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = pan_model.names[class_id]
            extracted_text = extract_text_from_bbox(image, [x1, y1, x2, y2])
            
            if class_name == "name":
                extracted_data["Name"] = extracted_text
            elif class_name == "dob":
                extracted_data["DOB"] = extracted_text
            elif class_name == "pan_number":
                extracted_data["PAN Number"] = extracted_text
            elif class_name == "father-s name":
                extracted_data["Father's Name"] = extracted_text
    
    return extracted_data


def process_image(image_path):
    document_type = detect_document_type(image_path)
    
    if document_type == "Aadhaar":
        data = extract_aadhaar_info(image_path)
    elif document_type == "PAN":
        data = extract_pan_info(image_path)
    else:
        data = {
            "Name": "Not Found",
            "Aadhar Number": "Not Found",
            "DOB": "Not Found",
            "Father's Name": "Not Found",
            "PAN Number": "Not Found",
        }
    
    return {"Document Type": document_type, "Extracted Data": data}


def gradio_interface(image):
    result = process_image(image)
    return result

# Create Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="filepath", label="Upload Document"),
    outputs=gr.JSON(label="Extracted Information"),
    title="Extractify",
    description="Upload an Aadhaar or PAN card to extract information.",
)

# Launch the app
iface.launch()
