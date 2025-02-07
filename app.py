import cv2
import easyocr
import numpy as np
import re
import json
from roboflow import Roboflow  
from ultralytics import YOLO
import gradio as gr

reader = easyocr.Reader(['en'])

# Initialize Roboflow client
rf = Roboflow(api_key="sFM2tWZMm4xZoYxI46WJ")  
project = rf.workspace().project("annotation-xsy5d")  
aadhaar_model = project.version(3).model 


document_model = YOLO("best.pt")  n
pan_model = YOLO("pan.pt")  # Model for PAN card extraction

# Function to detect document type
def detect_document_type(image_path):
    image = cv2.imread(image_path)
    results = document_model(image)
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            return class_name  # Return the detected document type
    
    return None  # No document detected

# Function to extract information from Aadhaar card
def extract_aadhaar_info(image_path):
    image = cv2.imread(image_path)
    
    # Send request to Roboflow API
    response = aadhaar_model.predict(image_path, confidence=40, overlap=30).json()
    
    def extract_text_from_bbox(image, bbox):
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
        x1, y1, x2, y2 = max(0, x - w//2), max(0, y - h//2), min(image.shape[1], x + w//2), min(image.shape[0], y + h//2)
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=0)
        return " ".join(results) if results else None
    
    extracted_data = {"Name": None, "DOB": None, "Aadhar Number": None}
    
    for prediction in response.get("predictions", []):
        label = prediction["class"]
        bbox = prediction  # Roboflow bounding box format
        extracted_text = extract_text_from_bbox(image, bbox)
        
        if label == "name":
            extracted_data["Name"] = extracted_text
        elif label == "dob":
            extracted_data["DOB"] = extracted_text
        elif label == "aadharNumber":
            extracted_data["Aadhar Number"] = extracted_text
    
    return extracted_data

# Function to extract information from PAN card
def extract_pan_info(image_path):
    image = cv2.imread(image_path)
    results = pan_model.predict(image_path)
    
    def extract_text_from_bbox(image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        result = reader.readtext(gray, detail=0)
        return " ".join(result) if result else None
    
    extracted_data = {"Name": None, "DOB": None, "PAN Number": None, "Father's Name": None}
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = pan_model.names[class_id]  # Get class name from class ID
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

# Function to process the uploaded image
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
    
    return data

# Gradio interface
def gradio_interface(image):
    result = process_image(image)
    return result

# Create Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="filepath", label="Upload Document"),
    outputs=gr.JSON(label="Extracted Information"),
    title="Document Information Extractor",
    description="Upload an Aadhaar or PAN card to extract information.",
)

# Launch the app
iface.launch()
