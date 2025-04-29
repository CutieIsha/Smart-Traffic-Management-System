import cv2
import os
import numpy as np
from ultralytics import YOLO

# Determine the correct model path based on the environment
local_model_path = "C:/Users/hp/OneDrive/Documents/numberplate_detection/license_plate_detector.pt"
chat_model_path = "/mnt/data/license_plate_detector.pt"

# Use the appropriate path
model_path = local_model_path if os.path.exists(local_model_path) else chat_model_path

# Check if model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Model file not found at {model_path}")

# Load the YOLO model
model = YOLO(model_path)

# Directory containing input videos
input_dir = "C:/Users/hp/OneDrive/Documents/numberplate_detection/archive"
output_dir = os.path.join(input_dir, "output")
plates_dir = os.path.join(output_dir, "plates")  # Folder for extracted plates

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plates_dir, exist_ok=True)

# Set the video file name manually
video_name = "mycarplate.mp4"  # Replace with your actual video file name
video_path = os.path.join(input_dir, video_name)

# List to store already saved plates (to prevent duplicates)
saved_plates = []

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_video_path = os.path.join(output_dir, f"processed_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        display_plate = None  # Holds the plate to display in the corner
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                
                if conf > 0.5:
                    # Extract plate region
                    plate_img = frame[y1:y2, x1:x2]

                    # Convert to grayscale for easy duplicate checking
                    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    
                    # Check if the plate is already saved
                    if not any(np.array_equal(plate_gray, p) for p in saved_plates):
                        # Save the plate
                        plate_filename = os.path.join(plates_dir, f"plate_{len(saved_plates)}.jpg")
                        cv2.imwrite(plate_filename, plate_img)
                        saved_plates.append(plate_gray)  # Store to prevent duplicates

                    # Store plate to display in the corner
                    display_plate = plate_img

                    # Draw a rectangle around the detected plate
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "License Plate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the extracted plate in the top-left corner
        if display_plate is not None:
            plate_resized = cv2.resize(display_plate, (200, 80))  # Resize for display
            frame[10:90, 10:210] = plate_resized  # Overlay plate at (10,10) position
        
        out.write(frame)
        cv2.imshow("License Plate Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Video saved at: {output_video_path}")
    print(f"Extracted {len(saved_plates)} unique number plates. Saved in: {plates_dir}")

# Check if video file exists before processing
if os.path.exists(video_path):
    process_video(video_path)
else:
    print("Error: File not found. Please check the file name and try again.")
