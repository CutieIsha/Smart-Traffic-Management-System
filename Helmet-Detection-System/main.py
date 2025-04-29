import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
import os
from datetime import datetime
import csv

# Specify your video file here
VIDEO_FILE = 'video/clip.mp4'  # Change this to your video filename

# Initialize PaddleOCR
ocr = PaddleOCR()

# Function to perform OCR on an image array
def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")
    
    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
        for result in results[0]:
            text = result[1][0]
            detected_text.append(text)

    # Join all detected texts into a single string
    return ''.join(detected_text)

# Mouse callback function for RGB window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load YOLOv8 model
model = YOLO("best.pt")
names = model.names

# Create directory for current date
current_date = datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(current_date):
    os.makedirs(current_date)

# Initialize CSV file path in the current date folder
csv_file_path = os.path.join(current_date, f"{current_date}.csv")

# Create CSV file with headers if it doesn't exist
csv_file_exists = os.path.exists(csv_file_path)
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not csv_file_exists:
        writer.writerow(["Number Plate", "Date", "Time", "Helmet Status"])

# Track processed track IDs
processed_track_ids = set()

# Color definitions for visualization
RED = (0, 0, 255)    # BGR format for no helmet (red)
GREEN = (0, 255, 0)  # BGR format for helmet (green)

# Open the video file
cap = cv2.VideoCapture(VIDEO_FILE)
print(f"Processing video: {VIDEO_FILE}")

if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_FILE}")
    exit()

# Get video dimensions for dynamic processing
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True)

    # Dictionary to match numberplates to riders
    rider_plates = {}  # {rider_track_id: (plate_track_id, plate_box)}
    helmet_status = {}  # {rider_track_id: status}

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        # First pass: collect all detections
        riders = []  # List to store rider information (track_id, box, class)
        plates = []  # List to store plate information (track_id, box)
        
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            if c in ['helmet', 'no-helmet']:
                riders.append((track_id, box, c))
                helmet_status[track_id] = c
            elif c == 'numberplate':
                plates.append((track_id, box))

        # Process riders
        for rider_id, rider_box, rider_class in riders:
            # Draw color-coded overlay for rider
            x1, y1, x2, y2 = rider_box
            color = GREEN if rider_class == 'helmet' else RED
            
            # Add semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Filled rectangle
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # Blend with original frame
            
            # Draw border and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{rider_id}: {rider_class}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Find closest numberplate
            if rider_class == 'no-helmet':
                # Calculate center of rider
                rider_cx = (x1 + x2) // 2
                rider_cy = (y1 + y2) // 2
                
                # Find the closest numberplate
                closest_plate = None
                min_distance = float('inf')
                
                for plate_id, plate_box in plates:
                    px1, py1, px2, py2 = plate_box
                    plate_cx = (px1 + px2) // 2
                    plate_cy = (py1 + py2) // 2
                    
                    # Calculate distance between rider and plate
                    distance = np.sqrt((rider_cx - plate_cx)**2 + (rider_cy - plate_cy)**2)
                    
                    # Update if this is the closest plate
                    if distance < min_distance and distance < 200:  # Threshold for association
                        min_distance = distance
                        closest_plate = (plate_id, plate_box)
                
                # If we found a close plate
                if closest_plate and closest_plate[0] not in processed_track_ids:
                    plate_id, plate_box = closest_plate
                    px1, py1, px2, py2 = plate_box
                    
                    # Draw connection line between rider and plate
                    cv2.line(frame, (rider_cx, rider_cy), ((px1+px2)//2, (py1+py2)//2), (255, 0, 255), 2)
                    
                    # Process the plate
                    crop = frame[py1:py2, px1:px2]
                    if crop.size > 0:  # Ensure valid crop
                        crop = cv2.resize(crop, (120, 85))
                        
                        # Show plate ID
                        cvzone.putTextRect(frame, f"Plate {plate_id}", (px1, py1-10), 1, 1)
                        
                        # Draw plate rectangle
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 255), 2)
                        
                        # Perform OCR on the plate
                        text = perform_ocr(crop)
                        print(f"Detected Number Plate: {text} for rider {rider_id}")
                        
                        current_time = datetime.now().strftime('%H-%M-%S-%f')[:12]
                        crop_image_path = os.path.join(current_date, f"{text}_{current_time}.jpg")
                        cv2.imwrite(crop_image_path, crop)
                        
                        # Write to CSV with helmet status
                        with open(csv_file_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([text, current_date, current_time, "No Helmet"])
                        
                        processed_track_ids.add(plate_id)

    # Add legend to the frame
    cv2.putText(frame, "Green: Helmet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(frame, "Red: No Helmet", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    
    # Display count of detected riders
    helmet_count = sum(1 for status in helmet_status.values() if status == 'helmet')
    no_helmet_count = sum(1 for status in helmet_status.values() if status == 'no-helmet')
    cv2.putText(frame, f"With Helmet: {helmet_count}", (frame.shape[1] - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(frame, f"No Helmet: {no_helmet_count}", (frame.shape[1] - 200, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()