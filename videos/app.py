import os
from flask import Flask, render_template, request, redirect, url_for, flash, Response
import cv2
import numpy as np
from ultralytics import YOLO
import math
import threading
import time
from werkzeug.utils import secure_filename

# Import your existing classes
class VehicleTracker:
    def __init__(self):
        self.vehicles = {}
        self.next_id = 0
        self.violation_ids = set()

    def update(self, detections):
        updated_vehicles = {}
        
        for (x, y, w, h) in detections:
            # Calculate center of detection
            cx, cy = (x + w) // 2, (y + h) // 2
            
            # Find closest existing vehicle
            best_match = None
            min_dist = 50  # Max distance to match
            
            for vid, (prev_x, prev_y, prev_w, prev_h, prev_id, prev_dir) in self.vehicles.items():
                prev_cx, prev_cy = (prev_x + prev_w) // 2, (prev_y + prev_h) // 2
                dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_match = vid
            
            if best_match is not None:
                # Update existing vehicle
                old_x, old_y, old_w, old_h, old_id, old_dir = self.vehicles[best_match]
                
                # Calculate direction
                dx = cx - ((old_x + old_w) // 2)
                dy = cy - ((old_y + old_h) // 2)
                
                # Determine primary direction
                if abs(dx) > abs(dy):
                    new_dir = 'right' if dx > 0 else 'left'
                else:
                    new_dir = 'down' if dy > 0 else 'up'
                
                updated_vehicles[best_match] = (x, y, w, h, old_id, new_dir)
            else:
                # New vehicle
                updated_vehicles[self.next_id] = (x, y, w, h, self.next_id, 'unknown')
                self.next_id += 1
        
        self.vehicles = updated_vehicles
        return [(x, y, w, h, vid, dir) for (x, y, w, h, vid, dir) in self.vehicles.values()]

    def is_new_violation(self, obj_id):
        if obj_id not in self.violation_ids:
            self.violation_ids.add(obj_id)
            return True
        return False

class DynamicTrafficViolationDetector:
    def __init__(self, video_path, model_path="yolov8m.pt"):
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file at {video_path}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.coco = self.model.model.names
        self.target_labels = ["bicycle", "car", "motorcycle", "bus", "truck"]
        
        # Initialize tracker and violation counter
        self.tracker = VehicleTracker()
        self.violation_counter = 0
        
        # Lane and region variables
        self.traffic_light_region = None
        self.lane_regions = []
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # For Flask streaming
        self.output_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def detect_traffic_light_color(self, frame, traffic_light_region):
        """Improved color detection method"""
        # Create a mask for the traffic light region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [traffic_light_region], 255)
        
        # Extract the traffic light region
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Improved color ranges with multiple thresholds
        color_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'yellow': [
                (np.array([20, 100, 100]), np.array([35, 255, 255]))
            ],
            'green': [
                (np.array([40, 50, 50]), np.array([90, 255, 255]))
            ]
        }
        
        # Detect color percentages with improved accuracy
        total_pixels = cv2.countNonZero(mask)
        color_percentages = {}
        
        for color, ranges in color_ranges.items():
            color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for (lower, upper) in ranges:
                temp_mask = cv2.inRange(hsv, lower, upper)
                color_mask = cv2.bitwise_or(color_mask, temp_mask)
            
            color_pixels = cv2.countNonZero(color_mask)
            color_percentages[color] = (color_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # More robust color determination
        color_thresholds = {
            'red': 3.0,     # Lower threshold for red
            'yellow': 2.0,  # Lower threshold for yellow
            'green': 2.0    # Lower threshold for green
        }
        
        # Find dominant color above its threshold
        dominant_color = 'unknown'
        max_percentage = 0
        
        for color, percentage in color_percentages.items():
            if percentage > color_thresholds[color] and percentage > max_percentage:
                dominant_color = color
                max_percentage = percentage
        
        return dominant_color

    def define_lanes(self, frame):
        """Define lane regions based on frame width"""
        frame_width = frame.shape[1]
        lane_width = frame_width // 4
        
        self.lane_regions = [
            # Left lane
            np.array([
                [0, frame.shape[0]//2],
                [lane_width, frame.shape[0]//2],
                [lane_width, frame.shape[0]],
                [0, frame.shape[0]]
            ]),
            # Middle left lane
            np.array([
                [lane_width, frame.shape[0]//2],
                [lane_width*2, frame.shape[0]//2],
                [lane_width*2, frame.shape[0]],
                [lane_width, frame.shape[0]]
            ]),
            # Middle right lane
            np.array([
                [lane_width*2, frame.shape[0]//2],
                [lane_width*3, frame.shape[0]//2],
                [lane_width*3, frame.shape[0]],
                [lane_width*2, frame.shape[0]]
            ]),
            # Right lane
            np.array([
                [lane_width*3, frame.shape[0]//2],
                [frame_width, frame.shape[0]//2],
                [frame_width, frame.shape[0]],
                [lane_width*3, frame.shape[0]]
            ])
        ]

    def draw_text_with_background(self, frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                                   scale=0.8, text_color=(255,255,255), 
                                   background_color=(0,0,0), border_color=(0,0,255), 
                                   thickness=2, padding=5):
        """Draws text with a background for better visibility."""
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = position
        cv2.rectangle(frame, (x - padding, y - text_height - padding), 
                      (x + text_width + padding, y + baseline + padding), 
                      background_color, cv2.FILLED)
        cv2.rectangle(frame, (x - padding, y - text_height - padding), 
                      (x + text_width + padding, y + baseline + padding), 
                      border_color, thickness)
        cv2.putText(frame, text, (x, y), font, scale, text_color, 
                    thickness, lineType=cv2.LINE_AA)

    def detect_violations_thread(self):
        """Detection thread for Flask streaming"""
        self.running = True
        
        # Read first frame
        success, first_frame = self.cap.read()
        if not success:
            print("Error: Could not read first frame")
            self.running = False
            return
        
        # Resize and define lanes
        first_frame = cv2.resize(first_frame, (1100, 700))
        self.define_lanes(first_frame)
        
        # Define traffic light region (with more precise location)
        self.traffic_light_region = np.array([
            [50, 50],   # Top-left
            [100, 50],  # Top-right
            [100, 150], # Bottom-right
            [50, 150]   # Bottom-left
        ])
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        while self.running:
            success, frame = self.cap.read()
            if not success:
                print(f"Number of frames processed: {frame_count}")
                self.running = False
                break
            
            frame_count += 1
            frame = cv2.resize(frame, (1100, 700))
            
            # Draw lane regions
            for lane in self.lane_regions:
                cv2.polylines(frame, [lane], True, [0, 255, 0], 1)
            
            # Draw traffic light region
            cv2.polylines(frame, [self.traffic_light_region], True, [0, 0, 255], 1)
            
            # Detect traffic light color
            traffic_light_color = self.detect_traffic_light_color(frame, self.traffic_light_region)
            
            # Detect objects
            results = self.model.predict(frame, conf=0.75)
            
            detections = []
            for result in results:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    if self.coco[int(cls)] in self.target_labels:
                        x, y, w, h = map(int, box)
                        detections.append((x, y, w, h))
                        cv2.rectangle(frame, (x, y), (w, h), [0, 255, 0], 2)
            
            # Track and check violations
            tracked_objects = self.tracker.update(detections)
            
            for x, y, w, h, obj_id, direction in tracked_objects:
                # Determine which lane the vehicle is in
                vehicle_center = ((x + w) // 2, (y + h) // 2)
                current_lane = None
                
                for i, lane in enumerate(self.lane_regions):
                    if cv2.pointPolygonTest(lane, vehicle_center, False) >= 0:
                        current_lane = i
                        break
                
                # Violation conditions (improved)
                light_violation = traffic_light_color in ['red']
                
                # Check violation based on direction and lane
                if light_violation:
                    # Lane-specific directional rules
                    lane_directions = {
                        0: ['right', 'up'],   # Left lane
                        1: ['right', 'up'],   # Middle left lane
                        2: ['right', 'up'],   # Middle right lane
                        3: ['right', 'up']    # Right lane
                    }
                    
                    # Check if vehicle is in a lane and moving in a violating direction
                    if (current_lane is not None and 
                        direction in lane_directions.get(current_lane, [])):
                        
                        if self.tracker.is_new_violation(obj_id):
                            self.violation_counter += 1
                        
                        cv2.rectangle(frame, (x, y), (w, h), [0, 0, 255], 2)
            
            # Display information
            self.draw_text_with_background(frame, f"Violations: {self.violation_counter}", (10, 30))
            self.draw_text_with_background(frame, f"Light: {traffic_light_color}", (10, 60), 
                                           background_color=(0,0,0), 
                                           border_color=(0,255,0) if traffic_light_color == 'green' else (0,0,255))
            
            # Optional: Display vehicle directions
            for x, y, w, h, obj_id, direction in tracked_objects:
                cv2.putText(frame, direction, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Update the output frame for Flask
            with self.lock:
                self.output_frame = frame.copy()
            
            # Slow down processing to avoid excessive CPU usage
            time.sleep(0.03)
        
        self.cap.release()

    def start_detection(self):
        """Start the detection in a separate thread"""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.detect_violations_thread)
            self.thread.daemon = True
            self.thread.start()
            return True
        return False

    def stop_detection(self):
        """Stop the detection thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def get_frame(self):
        """Get the current processed frame"""
        if self.output_frame is None:
            return None
        
        with self.lock:
            return self.output_frame.copy()

# Flask application
app = Flask(__name__)
app.secret_key = "traffic_violation_detection_secret_key"

# Global detector instance
detector = None

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global detector
    
    # Stop any existing detection
    if detector:
        detector.stop_detection()
        detector = None
    
    # Check if the post request has the file part
    if 'file' not in request.files and 'video_path' not in request.form:
        flash('No file part or video path')
        return redirect(request.url)
    
    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                detector = DynamicTrafficViolationDetector(filepath)
                detector.start_detection()
                flash('Detection started')
                return redirect(url_for('video_feed'))
            except Exception as e:
                flash(f'Error: {str(e)}')
                return redirect(url_for('index'))
    
    elif 'video_path' in request.form and request.form['video_path']:
        video_path = request.form['video_path']
        
        if os.path.exists(video_path):
            try:
                detector = DynamicTrafficViolationDetector(video_path)
                detector.start_detection()
                flash('Detection started')
                return redirect(url_for('video_feed'))
            except Exception as e:
                flash(f'Error: {str(e)}')
                return redirect(url_for('index'))
        else:
            flash(f'Error: File not found at {video_path}')
            return redirect(url_for('index'))
    
    flash('No valid file or path provided')
    return redirect(url_for('index'))

def generate():
    global detector
    
    while detector and detector.running:
        frame = detector.get_frame()
        
        if frame is not None:
            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    global detector
    
    if detector and detector.running:
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return redirect(url_for('index'))

@app.route('/detection_status')
def detection_status():
    global detector
    
    if detector and detector.running:
        return render_template('detection.html', status='running')
    else:
        return render_template('detection.html', status='stopped')

@app.route('/stop_detection')
def stop_detection():
    global detector
    
    if detector:
        detector.stop_detection()
        detector = None
        flash('Detection stopped')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    import webbrowser
    
    # Define the URL (default Flask port is 5000)
    url = "http://127.0.0.1:5000"
    
    # Open the URL in a new browser window/tab
    webbrowser.open_new(url)
    
    # Then start the Flask app
    app.run(debug=True, threaded=True)