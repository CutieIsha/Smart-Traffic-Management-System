from flask import Flask, render_template, Response, request, redirect, url_for, flash, send_from_directory, jsonify
import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue

app = Flask(__name__)
app.secret_key = "license_plate_detection_secret_key"

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
PLATES_FOLDER = os.path.join(OUTPUT_FOLDER, "plates")
MODEL_PATH = os.path.join(BASE_DIR, "license_plate_detector.pt")

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLATES_FOLDER, exist_ok=True)

# Global variables
processing_video = False
current_frame = None
frame_queue = queue.Queue(maxsize=10)
saved_plates = []
processing_complete = False
stop_processing = False  # Flag to stop processing

# Load the YOLO model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def process_video_thread(video_path):
    global processing_video, saved_plates, processing_complete, stop_processing
    processing_video = True
    processing_complete = False
    stop_processing = False  # Reset stop flag
    saved_plates = []  # Reset saved plates for new video
    plate_data = []  # Store plate images and metadata together
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            processing_video = False
            processing_complete = True
            return
        
        frame_count = 0
        last_detection_time = time.time()
        
        while cap.isOpened() and processing_video and not stop_processing:
            ret, frame = cap.read()
            if not ret:
                # Video has ended
                break
            
            frame_count += 1
            
            # Process every 3rd frame to reduce load
            if frame_count % 3 != 0:
                continue
                
            # Process with YOLO model
            if model is not None:
                results = model(frame)
                display_plate = None
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        
                        if conf > 0.5:  # Confidence threshold
                            # Extract plate region
                            plate_img = frame[y1:y2, x1:x2]
                            
                            # Skip invalid plates
                            if plate_img.size <= 0 or plate_img.shape[0] <= 0 or plate_img.shape[1] <= 0:
                                continue
                                
                            # Normalize plate image
                            h, w = plate_img.shape[:2]
                            if h < 20 or w < 20 or h > 500 or w > 500:
                                continue  # Skip unusually small or large detections
                            
                            # Convert to grayscale and normalize
                            plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                            
                            # Check if the plate region is mostly uniform (might be a false detection)
                            if np.std(plate_gray) < 20:  # Skip low-variance images (likely not a plate)
                                continue
                                
                            # Apply preprocessing for better comparison
                            plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 0)
                            plate_gray = cv2.equalizeHist(plate_gray)  # Enhance contrast
                            
                            # Calculate key features for comparison - FIXED: Ensure this works correctly
                            descriptors = None
                            has_features = False
                            try:
                                # Create ORB feature detector
                                orb = cv2.ORB_create(nfeatures=500)  # Increase number of features
                                keypoints, descriptors = orb.detectAndCompute(plate_gray, None)
                                has_features = descriptors is not None and len(descriptors) > 0
                            except Exception as e:
                                print(f"Feature extraction error: {e}")
                                has_features = False
                            
                            # FIXED: Improved duplicate detection
                            is_duplicate = False
                            best_match_idx = -1
                            best_match_score = float('inf')
                            
                            # FIXED: Simplified duplicate detection with stronger thresholds
                            for idx, (saved_gray, _, _) in enumerate(plate_data):
                                # Ensure plates are similar size
                                if abs(saved_gray.shape[0] - plate_gray.shape[0]) > (saved_gray.shape[0] * 0.3) or \
                                   abs(saved_gray.shape[1] - plate_gray.shape[1]) > (saved_gray.shape[1] * 0.3):
                                    continue
                                
                                # Resize for comparison
                                if saved_gray.shape != plate_gray.shape:
                                    compare_plate = cv2.resize(plate_gray, (saved_gray.shape[1], saved_gray.shape[0]))
                                else:
                                    compare_plate = plate_gray
                                
                                # Method 1: Simple pixel difference
                                diff = cv2.absdiff(saved_gray, compare_plate)
                                diff_score = np.mean(diff) / 255.0  # Normalize to 0-1
                                
                                # Method 2: Histogram comparison
                                hist1 = cv2.calcHist([compare_plate], [0], None, [64], [0, 256])
                                hist2 = cv2.calcHist([saved_gray], [0], None, [64], [0, 256])
                                cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                                cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                                hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                                
                                # Combined score (lower is more similar)
                                combined_score = diff_score - hist_score
                                
                                # Check for duplicate
                                if diff_score < 0.3 and hist_score > 0.7:  # Both methods must agree
                                    is_duplicate = True
                                    if combined_score < best_match_score:
                                        best_match_score = combined_score
                                        best_match_idx = idx
                            
                            # If not a duplicate or first plate, save it
                            if not is_duplicate or not plate_data:
                                # Create filename
                                plate_filename = os.path.join(PLATES_FOLDER, f"plate_{len(plate_data)}.jpg")
                                cv2.imwrite(plate_filename, plate_img)
                                
                                # Add to our collection of plates
                                plate_data.append((plate_gray, descriptors, plate_img))
                                saved_plates = [item[2] for item in plate_data]  # Keep original RGB images for display
                                
                                # Update last detection time
                                last_detection_time = time.time()
                                
                                # Use last saved plate for display
                                display_plate = plate_img
                            else:
                                # If it's a duplicate but significantly clearer, replace the old one
                                if best_match_idx >= 0:
                                    old_plate_gray = plate_data[best_match_idx][0]
                                    old_std = np.std(old_plate_gray)
                                    new_std = np.std(plate_gray)
                                    
                                    # If new image has better contrast or clarity
                                    if new_std > (old_std * 1.2):
                                        # Replace existing file
                                        plate_filename = os.path.join(PLATES_FOLDER, f"plate_{best_match_idx}.jpg")
                                        cv2.imwrite(plate_filename, plate_img)
                                        
                                        # Update data
                                        plate_data[best_match_idx] = (plate_gray, descriptors, plate_img)
                                        saved_plates[best_match_idx] = plate_img
                                        
                                        # Use for display
                                        display_plate = plate_img
                                        
                                        print(f"Replaced plate {best_match_idx} with better quality image")
                            
                            # Draw rectangle and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"License Plate: {conf:.2f}", 
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                # Display the extracted plate in the top-left corner
                if display_plate is not None and display_plate.size > 0:
                    try:
                        h, w = display_plate.shape[:2]
                        # Only resize if dimensions are reasonable
                        if h > 0 and w > 0 and h < 1000 and w < 1000:
                            plate_resized = cv2.resize(display_plate, (200, 80))
                            # Make sure the overlay region is valid
                            if frame.shape[0] >= 90 and frame.shape[1] >= 210:
                                frame[10:90, 10:210] = plate_resized
                    except Exception as e:
                        print(f"Error resizing plate: {e}")
            
            # FIXED: Removed the unique plates counter
            
            # Add text indicating stop button is available
            cv2.putText(frame, "Press 'Stop Detection' to end", 
                        (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Update the current frame for streaming
            if not frame_queue.full():
                frame_queue.put(frame)
            
            # Brief pause to reduce CPU usage
            time.sleep(0.01)
            
        cap.release()
        
        # Add completion message to the last frame
        if current_frame is not None:
            completion_frame = current_frame.copy()
            if stop_processing:
                message = "Detection Stopped by User"
            else:
                message = "Processing Complete"
                
            cv2.putText(completion_frame, message, 
                      (completion_frame.shape[1]//4, completion_frame.shape[0]//2),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frame_queue.put(completion_frame)
        
        # Mark processing as complete    
        processing_complete = True
            
    except Exception as e:
        print(f"Error in processing thread: {e}")
    finally:
        processing_video = False
        processing_complete = True
        stop_processing = False

def generate_frames():
    global current_frame, processing_complete
    
    while True:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get(timeout=1)
                current_frame = frame
            elif current_frame is not None:
                if processing_complete:
                    # If processing is complete, add a message to the current frame
                    frame = current_frame.copy()
                    if stop_processing:
                        message = "Detection Stopped by User"
                    else:
                        message = "Processing Complete"
                    
                    cv2.putText(frame, message, 
                              (frame.shape[1]//4, frame.shape[0]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    frame = current_frame
            else:
                # If no frames are available, send a blank frame
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                cv2.putText(frame, "Waiting for video processing...", 
                          (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
            # If processing is complete and we've shown the final frame, slow down updates
            if processing_complete:
                time.sleep(0.1)  # Reduce updates when processing is done
            else:
                time.sleep(0.03)  # About 30 FPS during active processing
            
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    global processing_video
    
    if processing_video:
        flash("A video is already being processed. Please wait.")
        return redirect(url_for('index'))
        
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
        
    # Check if the file is allowed
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        flash('Invalid file type. Please upload a video file (mp4, avi, mov, mkv).')
        return redirect(url_for('index'))
    
    # Clear previous plates
    for filename in os.listdir(PLATES_FOLDER):
        file_path = os.path.join(PLATES_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Save the uploaded file
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)
    
    # Start processing in a background thread
    threading.Thread(target=process_video_thread, args=(video_path,), daemon=True).start()
    
    return redirect(url_for('analysis'))

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_plates')
def view_plates():
    plates = []
    for filename in sorted(os.listdir(PLATES_FOLDER)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            plates.append(filename)
    return render_template('plates.html', plates=plates)

@app.route('/plate/<filename>')
def plate_file(filename):
    return send_from_directory(PLATES_FOLDER, filename)

# Endpoint to stop detection
@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global stop_processing
    stop_processing = True
    return jsonify({"status": "success", "message": "Detection stopping..."})

if __name__ == '__main__':
    app.run(debug=True, port=3000)