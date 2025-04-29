from flask import Flask, render_template, request, Response, redirect, url_for, flash, send_from_directory
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort import Sort
import csv
import random
import string
import datetime
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'wrongway_detection_secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for detection
global_cap = None
global_out_frame = None
global_processing = False
global_csv_filename = None

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize SORT Tracker
tracker = Sort()

# Thresholds for movement and IoU filtering
MIN_MOVEMENT_THRESHOLD = 15  # Pixels
IOU_THRESHOLD = 0.6  # IoU threshold for duplicate filtering
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_maharashtra_license_plate():
    """Generate a random Maharashtra license plate."""
    district_codes = [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
        "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
        "41", "42", "43", "44", "45", "46", "47", "48", "49", "50"
    ]
    
    district_code = random.choice(district_codes)
    series_code = ''.join(random.choices(string.ascii_uppercase, k=2))
    number = ''.join(random.choices(string.digits, k=4))
    
    # Format: MH-NN-XX-NNNN (e.g., MH-01-AB-1234)
    return f"MH-{district_code}-{series_code}-{number}"

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def draw_text_with_background(img, text, position, font, font_scale, text_color, bg_color, thickness=1, padding=5):
    """Draw text with a colored background rectangle."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle dimensions with padding
    bg_left = position[0] - padding
    bg_top = position[1] - text_height - padding
    bg_right = position[0] + text_width + padding
    bg_bottom = position[1] + padding
    
    # Draw background rectangle
    cv2.rectangle(img, (bg_left, bg_top), (bg_right, bg_bottom), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, position, font, font_scale, text_color, thickness)

def process_video(video_path):
    global global_cap, global_out_frame, global_processing, global_csv_filename
    
    # Set processing flag
    global_processing = True
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    global_cap = cap
    
    if not cap.isOpened():
        global_processing = False
        return "Error: Could not open video file."
    
    # Get video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        global_processing = False
        cap.release()  # Close the video if we can't read the first frame
        return "Error: Could not read the first frame."
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Object tracking history
    object_tracks = {}  # Stores vehicle positions
    wrong_way_vehicles = set()  # Stores IDs of wrong-way vehicles
    wrong_way_count = 0
    vehicle_first_entry = {}  # Tracks first entry positions
    wrong_way_recorded = set()  # Tracks which vehicles have been recorded in CSV
    
    # CSV file setup
    program_start_time = datetime.datetime.now()
    formatted_date = program_start_time.strftime("%Y-%m-%d")
    formatted_time = program_start_time.strftime("%H:%M:%S")
    csv_filename = f"wrong_way_vehicles_{program_start_time.strftime('%Y%m%d_%H%M%S')}.csv"
    global_csv_filename = csv_filename
    
    # Create and initialize CSV file with headers
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Detection_ID', 'License_Plate', 'Detection_Timestamp', 'Program_Date', 'Program_Time'])
    
    # Optical Flow Parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Store latest tracking data for consistent drawing
    latest_trackers = []
    latest_wrong_way_ids = set()
    
    def record_wrong_way_vehicle(vehicle_id):
        """Record wrong-way vehicle details to CSV file."""
        if vehicle_id in wrong_way_recorded:
            return  # Skip if already recorded
        
        wrong_way_recorded.add(vehicle_id)
        license_plate = generate_maharashtra_license_plate()
        detection_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([vehicle_id, license_plate, detection_time, formatted_date, formatted_time])
        
        print(f"Recorded wrong-way vehicle {vehicle_id} with license plate {license_plate}")
    
    # For frame skip to improve performance
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
    
    # Variable to track if we've wrapped around the video
    video_looped = False
    
    # Add a timeout mechanism to close the video after a certain duration
    max_playbacks = 1  # Maximum number of times to play the video
    current_playbacks = 0
    
    try:
        while cap.isOpened() and current_playbacks < max_playbacks:
            ret, frame = cap.read()
            if not ret:
                # Reached end of video
                current_playbacks += 1
                if current_playbacks < max_playbacks:
                    # Reset position to beginning of the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    video_looped = True
                    continue
                else:
                    # We've played the video the maximum number of times, exit the loop
                    break
            
            frame_count += 1
            process_this_frame = frame_count % PROCESS_EVERY_N_FRAMES == 0 or frame_count == 1
            
            # Make a clean copy of the frame for drawing
            display_frame = frame.copy()
            
            if process_this_frame and not video_looped:  # Only process new detections if we haven't completed one loop
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
                # Run YOLOv8 inference
                results = model(frame)
                detections = []
        
                for result in results:
                    for box in result.boxes.data:
                        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                        label = int(cls)
        
                        # Only detect vehicles (Cars, Trucks, Motorcycles, Buses) with confidence >= threshold
                        if label in [2, 3, 5, 7] and conf >= CONFIDENCE_THRESHOLD:
                            detections.append([x1, y1, x2, y2, conf])
        
                detections = np.array(detections) if detections else np.empty((0, 5))
        
                # Track objects using SORT
                trackers = tracker.update(detections)
                latest_trackers = trackers.copy()  # Store for non-processed frames
        
                # Optical Flow for Motion Detection
                moving_objects = []
                if len(trackers) > 0:
                    object_points = np.array([[((x1 + x2) // 2, (y1 + y2) // 2)] for x1, y1, x2, y2, _ in trackers], dtype=np.float32)
                    new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, object_points, None, **lk_params)
        
                    for i, (new, old) in enumerate(zip(new_points, object_points)):
                        if status[i] == 1:  # If tracking is successful
                            dx, dy = new[0] - old[0]
                            movement_distance = np.sqrt(dx**2 + dy**2)
        
                            if movement_distance > MIN_MOVEMENT_THRESHOLD:
                                moving_objects.append(i)  # Store moving objects index
        
                # Process each tracked object
                for i, track in enumerate(trackers):
                    if i not in moving_objects:
                        continue  # Ignore stationary objects
        
                    x1, y1, x2, y2, track_id = track.astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
                    # IoU-based duplicate filtering
                    is_duplicate = False
                    for prev_id, prev_box in vehicle_first_entry.items():
                        if track_id != prev_id and compute_iou(prev_box, (x1, y1, x2, y2)) > IOU_THRESHOLD:
                            is_duplicate = True
                            break
        
                    if is_duplicate:
                        continue  # Ignore duplicate entry
        
                    # Store first entry as full bounding box
                    if track_id not in vehicle_first_entry:
                        vehicle_first_entry[track_id] = (x1, y1, x2, y2)
        
                    # Track movement direction
                    if track_id in object_tracks:
                        prev_cx, prev_cy = object_tracks[track_id]
                        dx, dy = cx - prev_cx, cy - prev_cy
        
                        # Ignore minor movements
                        if abs(dx) < MIN_MOVEMENT_THRESHOLD and abs(dy) < MIN_MOVEMENT_THRESHOLD:
                            continue
        
                        # Detect wrong-way movement (Assuming RIGHT is WRONG)
                        if dx > 0:  
                            if track_id not in wrong_way_vehicles:
                                wrong_way_count += 1
                                wrong_way_vehicles.add(track_id)
                                # Record this wrong-way vehicle in CSV
                                record_wrong_way_vehicle(track_id)
                            latest_wrong_way_ids.add(track_id)  # Update for consistent display
        
                    object_tracks[track_id] = (cx, cy)
                
                prev_gray = curr_gray.copy()
            
            # Always draw tracking boxes and text (for every frame)
            for track in latest_trackers:
                x1, y1, x2, y2, track_id = track.astype(int)
                
                # Choose color based on whether this is a wrong-way vehicle
                if track_id in wrong_way_vehicles:
                    # Draw red box for wrong-way vehicles
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Draw "Wrong-Way!" text with red background and white text
                    draw_text_with_background(
                        display_frame, 
                        "Wrong-Way!", 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (255, 255, 255),  # White text
                        (0, 0, 255),      # Red background
                        2,                # Thickness
                        5                 # Padding
                    )
                    
                    # Add the license plate if it's a wrong-way vehicle
                    if track_id in wrong_way_recorded:
                        # Reopen the CSV to find this vehicle's license plate
                        license_plate = None
                        with open(csv_filename, 'r') as csvfile:
                            csv_reader = csv.reader(csvfile)
                            next(csv_reader)  # Skip header
                            for row in csv_reader:
                                if int(row[0]) == track_id:
                                    license_plate = row[1]
                                    break
                        
                        if license_plate:
                            # Display license plate with yellow background
                            draw_text_with_background(
                                display_frame, 
                                f"Plate: {license_plate}", 
                                (x1, y2 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (0, 0, 0),       # Black text
                                (255, 255, 0),   # Yellow background
                                2,               # Thickness
                                5                # Padding
                            )
                else:
                    # Draw green box for correct-way vehicles
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw ID with black background and white text
                draw_text_with_background(
                    display_frame, 
                    f"ID: {track_id}", 
                    (x1, y1 - 5 if track_id not in wrong_way_vehicles else y1 - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255),  # White text
                    (0, 0, 0),        # Black background
                    2,                # Thickness
                    3                 # Padding
                )
        
            # Show Wrong-Way Count with dark background and white text
            draw_text_with_background(
                display_frame, 
                f"Wrong-Way Vehicles: {wrong_way_count}", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255),  # White text
                (0, 0, 150),      # Dark blue background
                3,                # Thickness
                10                # Padding
            )
        
            # Update global frame for streaming
            global_out_frame = display_frame.copy()
            
            # Slight delay to reduce CPU usage
            time.sleep(0.03)
    
    except Exception as e:
        print(f"Error during video processing: {e}")
    finally:
        # Always clean up resources when done
        if cap:
            cap.release()
        global_processing = False
        print("Video processing complete. Resources cleaned up.")

def generate_frames():
    global global_cap, global_out_frame, global_processing
    
    # Wait for processing to start
    while not global_processing:
        time.sleep(0.1)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + b'\r\n')
    
    while global_processing:
        if global_out_frame is not None:
            ret, buffer = cv2.imencode('.jpg', global_out_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + b'\r\n')
        else:
            # Empty frame while waiting
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global global_csv_filename
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Start processing in a new thread
        import threading
        processing_thread = threading.Thread(target=process_video, args=(filepath,))
        processing_thread.daemon = True
        processing_thread.start()
        
        return redirect(url_for('detect'))
    else:
        flash('File type not allowed. Please upload MP4, AVI, MOV, or MKV files.')
        return redirect(url_for('index'))

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_csv')
def download_csv():
    global global_csv_filename
    if global_csv_filename and os.path.exists(global_csv_filename):
        directory = os.path.dirname(global_csv_filename)
        filename = os.path.basename(global_csv_filename)
        return send_from_directory(directory or '.', filename, as_attachment=True)
    else:
        flash('No CSV file available for download.')
        return redirect(url_for('detect'))

if __name__ == '__main__':
    app.run(debug=True)