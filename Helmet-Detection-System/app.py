from flask import Flask, render_template, request, Response, redirect, url_for, flash, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
import os
from datetime import datetime
import csv
import time
import threading
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "license_plate_detection_secret_key"

# Global variables
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global variables for video processing
video_path = None
detection_active = False
output_frame = None
processing_thread = None
current_date_folder = None
csv_file_path = None

# Initialize PaddleOCR
ocr = PaddleOCR()

# Function to determine which model to use based on video filename
def get_model_for_video(video_filename):
    # Use best.pt for final.mp4 video
    if video_filename.lower() == "final.mp4":
        return YOLO("best.pt"), "best"
    else:
        # Default model for all other videos
        return YOLO("helmet.pt"), "helmet"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_path, current_date_folder, csv_file_path
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Set the video path for processing
        video_path = file_path
        
        # Create directory for current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_date_folder = os.path.join(RESULTS_FOLDER, current_date)
        os.makedirs(current_date_folder, exist_ok=True)
        
        # Initialize CSV file path
        csv_file_path = os.path.join(current_date_folder, f"{current_date}.csv")
        
        # Create CSV file with headers if it doesn't exist
        csv_file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not csv_file_exists:
                writer.writerow(["Number Plate", "Date", "Time", "Helmet Status"])
        
        # Start detection in a new thread
        start_detection()
        
        return redirect(url_for('detect'))
    else:
        flash('File type not allowed. Please upload a video file (mp4, avi, mov, mkv)')
        return redirect(url_for('index'))

def perform_ocr(image_array):
    if image_array is None:
        return ""
    
    try:
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
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def process_video():
    global video_path, detection_active, output_frame, current_date_folder, csv_file_path
    
    # Set active flag
    detection_active = True
    
    # Get the filename from the path
    video_filename = os.path.basename(video_path)
    
    # Load the appropriate model based on the video
    model, model_type = get_model_for_video(video_filename)
    names = model.names
    
    print(f"Using model: {model_type} for video: {video_filename}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        detection_active = False
        return
    
    # Track processed track IDs
    processed_track_ids = set()
    
    # Color definitions for visualization
    RED = (0, 0, 255)    # BGR format for no helmet (red)
    GREEN = (0, 255, 0)  # BGR format for helmet (green)
    
    while detection_active:
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
                            crop_image_path = os.path.join(current_date_folder, f"{text}_{current_time}.jpg")
                            cv2.imwrite(crop_image_path, crop)
                            
                            # Write to CSV with helmet status
                            with open(csv_file_path, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([text, datetime.now().strftime('%Y-%m-%d'), current_time, "No Helmet"])
                            
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
        
        # Update the output frame
        output_frame = frame.copy()
        
        # Sleep to reduce CPU usage
        time.sleep(0.01)
    
    cap.release()
    print("Video processing stopped")

def generate_frames():
    global output_frame
    
    while True:
        if output_frame is not None:
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Return a blank frame if output_frame is None
            blank_frame = np.ones((500, 1020, 3), dtype=np.uint8) * 255
            cv2.putText(blank_frame, "Waiting for video to process...", (300, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

def start_detection():
    global processing_thread, detection_active
    
    if processing_thread is None or not processing_thread.is_alive():
        detection_active = True
        processing_thread = threading.Thread(target=process_video)
        processing_thread.daemon = True
        processing_thread.start()

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False
    return {'status': 'success', 'message': 'Detection stopped'}

@app.route('/view_plates')
def view_plates():
    # Get the current date folder
    if current_date_folder is None:
        flash('No detection results available')
        return redirect(url_for('index'))
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(current_date_folder) if f.endswith('.jpg')]
    
    # Read CSV data for display
    csv_data = []
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            # Skip header row
            next(reader, None)
            for row in reader:
                if len(row) >= 4:  # Ensure row has all expected columns
                    csv_data.append({
                        'plate': row[0],
                        'date': row[1],
                        'time': row[2],
                        'helmet': row[3]
                    })
    
    return render_template('plates.html', 
                           images=image_files, 
                           folder=os.path.basename(current_date_folder),
                           csv_data=csv_data)

@app.route('/download_csv')
def download_csv():
    if csv_file_path and os.path.exists(csv_file_path):
        return send_file(csv_file_path, 
                         mimetype='text/csv',
                         download_name=f"license_plates_{os.path.basename(current_date_folder)}.csv",
                         as_attachment=True)
    else:
        flash('CSV file not found')
        return redirect(url_for('index'))

@app.route('/image/<folder>/<filename>')
def get_image(folder, filename):
    folder_path = os.path.join(RESULTS_FOLDER, folder)
    return send_file(os.path.join(folder_path, filename))

if __name__ == '__main__':
    app.run(debug=True, port=7000)