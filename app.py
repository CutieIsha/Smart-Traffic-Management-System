import os
import cv2
import numpy as np
import random
import csv
import time
from flask import Flask, render_template, request, Response, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from tracker import Tracker

app = Flask(__name__)
app.secret_key = 'traffic_violation_detection_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['VIOLATIONS_CSV'] = 'static/violations.csv'  # Path to CSV file

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Sample video options
SAMPLE_VIDEOS = {
    'intersection': 'static/samples/intersection.mp4',
    'highway': 'static/samples/highway.mp4',
}

class TrafficViolationDetector:
    def __init__(self, model_path="yolov8m.pt", confidence=0.75):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.coco_names = self.model.model.names
        self.target_labels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]
        
        # Initialize tracker and counters
        self.tracker = Tracker()
        self.reset()
        
        # Define brightness threshold for light detection
        self.brightness_threshold = 128
        
        # Initialize CSV for violations if it doesn't exist
        self.csv_file = app.config['VIOLATIONS_CSV']
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Violation Count', 'Video Name'])
    
    def reset(self):
        """Reset counters and tracking information for a new video"""
        self.violation_counter = 0
        self.vehicle_info = {}
        self.recorded_violations = {}  # Track which object IDs have been recorded
        self.tracker = Tracker()  # Reset the tracker as well
        
    def generate_maharashtra_plate(self):
        """Generate a random Maharashtra license plate"""
        # Format: MH-XX-XX-XXXX
        # MH is fixed for Maharashtra
        # First two digits: District code (01-99)
        # Second two characters: Series code (A-Z)
        # Last four digits: Random number
        
        district_code = random.randint(1, 99)
        series_code1 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        series_code2 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        number = random.randint(1, 9999)
        
        plate = f"MH-{district_code:02d}-{series_code1}{series_code2}-{number:04d}"
        return plate
    
    def record_violation(self, vehicle_type, obj_id):
        """Record violation details to CSV only if this object ID hasn't been recorded yet"""
        # Check if this object ID is already in our recorded violations
        # This way we only record each vehicle once
        if obj_id in self.recorded_violations:
            return self.recorded_violations[obj_id]
            
        license_plate = self.generate_maharashtra_plate()
        # Store this object ID as recorded
        self.recorded_violations[obj_id] = license_plate
        return license_plate
        
    def save_violation_count(self, video_name):
        """Save only the total violation count to CSV"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.violation_counter, video_name])
        
    def set_regions(self, red_light, green_light, roi):
        """Set the detection regions with forward-shifted ROI"""
        self.red_light = np.array(red_light)
        self.green_light = np.array(green_light)
        
        # Shift amount to move ROI forward
        shift_amount = 80
        
        # Shift all points upward (reduce y-coordinates)
        shifted_roi = []
        for point in roi:
            shifted_roi.append([point[0], point[1] - shift_amount])
        
        # Make ROI rectangular
        x_coords = [point[0] for point in shifted_roi]
        min_x, max_x = min(x_coords), max(x_coords)
        
        y_coords_top = [point[1] for point in shifted_roi[:2]]
        y_coords_bottom = [point[1] for point in shifted_roi[2:]]
        avg_top = int(sum(y_coords_top) / len(y_coords_top))
        avg_bottom = int(sum(y_coords_bottom) / len(y_coords_bottom))
        
        self.roi = np.array([
            [max_x, avg_top],
            [min_x, avg_top],
            [min_x, avg_bottom],
            [max_x, avg_bottom]
        ])

    def is_light_on(self, frame, polygon):
        """Check if traffic light is on"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [polygon], 255)
        roi = cv2.bitwise_and(gray, gray, mask=mask)
        return cv2.mean(roi, mask=mask)[0] > self.brightness_threshold

    def draw_text_with_background(self, frame, text, position, font, scale, text_color, bg_color, border_color, thickness=2, padding=5):
        """Draw text with background"""
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = position
        cv2.rectangle(frame, (x-padding, y-text_height-padding), 
                     (x+text_width+padding, y+baseline+padding), bg_color, cv2.FILLED)
        cv2.rectangle(frame, (x-padding, y-text_height-padding),
                     (x+text_width+padding, y+baseline+padding), border_color, thickness)
        cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

    def process_frame(self, frame):
        """Process each frame for violations"""
        # Draw regions
        cv2.polylines(frame, [self.red_light], True, [0,0,255], 1)
        cv2.polylines(frame, [self.green_light], True, [0,255,0], 1)
        cv2.polylines(frame, [self.roi], True, [255,0,0], 2)
        
        # Check light state
        red_on = self.is_light_on(frame, self.red_light)
        green_on = self.is_light_on(frame, self.green_light)
        light_state = red_on or not green_on
        
        # Detect objects
        results = self.model.predict(frame, conf=self.confidence)
        detections = []
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if self.coco_names[int(cls)] in self.target_labels:
                    x1, y1, x2, y2 = map(int, box)
                    vehicle_type = self.coco_names[int(cls)]
                    detections.append((x1, y1, x2, y2, vehicle_type))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), [0,255,0], 2)
        
        # Track objects
        tracked_objects = self.tracker.update([d[:4] for d in detections])
        
        # Map tracked objects back to their vehicle types
        tracked_with_type = []
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            # Find the original detection that matches this tracked object
            best_iou = 0
            best_type = "unknown"
            for det_x1, det_y1, det_x2, det_y2, veh_type in detections:
                # Calculate IoU between tracked box and detection box
                xA = max(x1, det_x1)
                yA = max(y1, det_y1)
                xB = min(x2, det_x2)
                yB = min(y2, det_y2)
                
                # Calculate intersection area
                interArea = max(0, xB - xA) * max(0, yB - yA)
                
                # Calculate union area
                boxAArea = (x2 - x1) * (y2 - y1)
                boxBArea = (det_x2 - det_x1) * (det_y2 - det_y1)
                unionArea = float(boxAArea + boxBArea - interArea)
                
                # Calculate IoU
                iou = 0 if unionArea == 0 else interArea / unionArea
                
                if iou > best_iou:
                    best_iou = iou
                    best_type = veh_type
            
            tracked_with_type.append((x1, y1, x2, y2, obj_id, best_type))
        
        # Process each object
        for obj in tracked_with_type:
            x1, y1, x2, y2, obj_id, vehicle_type = obj
            center = ((x1+x2)//2, (y1+y2)//2)
            
            if obj_id not in self.vehicle_info:
                self.vehicle_info[obj_id] = {
                    'positions': [],
                    'entered_roi': False,
                    'violated': False,
                    'counted': False,
                    'license_plate': None,
                    'vehicle_type': vehicle_type
                }
            
            self.vehicle_info[obj_id]['positions'].append(center)
            if len(self.vehicle_info[obj_id]['positions']) > 10:
                self.vehicle_info[obj_id]['positions'] = self.vehicle_info[obj_id]['positions'][-10:]
            
            in_roi = cv2.pointPolygonTest(self.roi, (float(center[0]), float(center[1])), False) >= 0
            
            if in_roi and not self.vehicle_info[obj_id]['entered_roi']:
                self.vehicle_info[obj_id]['entered_roi'] = True
            
            if (light_state and self.vehicle_info[obj_id]['entered_roi'] 
                and not self.vehicle_info[obj_id]['violated']):
                if len(self.vehicle_info[obj_id]['positions']) >= 5:
                    prev_pos = self.vehicle_info[obj_id]['positions'][0]
                    curr_pos = self.vehicle_info[obj_id]['positions'][-1]
                    distance = ((curr_pos[0]-prev_pos[0])**2 + (curr_pos[1]-prev_pos[1])**2)**0.5
                    
                    if distance > 10:
                        self.vehicle_info[obj_id]['violated'] = True
                        if not self.vehicle_info[obj_id]['counted']:
                            self.violation_counter += 1
                            self.vehicle_info[obj_id]['counted'] = True
                            
                            # Generate plate but don't record individual violations
                            license_plate = self.generate_maharashtra_plate()
                            self.vehicle_info[obj_id]['license_plate'] = license_plate
                            self.recorded_violations[obj_id] = license_plate
                            
                        cv2.rectangle(frame, (x1,y1), (x2,y2), [0,0,255], 2)
                        
                        # Display the license plate above the vehicle
                        if self.vehicle_info[obj_id]['license_plate']:
                            self.draw_text_with_background(
                                frame, 
                                f"Violation! {self.vehicle_info[obj_id]['license_plate']}", 
                                (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), (0,0,255), (0,0,0)
                            )
            
            # Remove vehicle info when it leaves the ROI completely
            if (not in_roi and 
                len(self.vehicle_info[obj_id]['positions']) > 5 and
                all(cv2.pointPolygonTest(self.roi, (float(p[0]), float(p[1])), False) < 0 
                for p in self.vehicle_info[obj_id]['positions'][-5:])):
                del self.vehicle_info[obj_id]
        
        # Display info
        light_status = "RED" if light_state else "GREEN"
        light_color = (0,0,255) if light_state else (0,255,0)
        
        self.draw_text_with_background(
            frame, f"Violations: {self.violation_counter}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), (0,0,0), (0,0,255))
        
        self.draw_text_with_background(
            frame, f"Light: {light_status}", (10,70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), (0,0,0), light_color)
        
        return frame

# Initialize detector
detector = TrafficViolationDetector()
# Default regions - these should be configurable in a real application
red_light = [[998,125], [998,155], [972,152], [970,127]]
green_light = [[971,200], [996,200], [1001,228], [971,230]]
roi = [[910,372], [388,365], [338,428], [917,441]]
detector.set_regions(red_light, green_light, roi)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_frames(video_path, resize_dim=(1100, 700)):
    # Reset detector counters for new video
    detector.reset()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + open('static/error.jpg', 'rb').read() + b'\r\n'
        return
    
    # Get total frame count and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Extract video name from path
    video_name = os.path.basename(video_path)
    
    # Create a progress indicator
    frame_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video - save final violation count to CSV
            detector.save_violation_count(video_name)
            
            # Show final frame with "Processing Complete" message
            final_frame = np.zeros((resize_dim[1], resize_dim[0], 3), dtype=np.uint8)
            cv2.putText(final_frame, "Processing Complete", (int(resize_dim[0]/4), int(resize_dim[1]/2)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            ret, buffer = cv2.imencode('.jpg', final_frame)
            final_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + final_bytes + b'\r\n')
            break
        
        frame_counter += 1
        frame = cv2.resize(frame, resize_dim)
        
        # Calculate progress percentage
        progress = int((frame_counter / total_frames) * 100)
        
        # Add progress bar
        cv2.rectangle(frame, (10, resize_dim[1]-30), (10 + int(progress * (resize_dim[0]-20)/100), resize_dim[1]-10), 
                     (0,255,0), -1)
        cv2.putText(frame, f"Progress: {progress}%", (10, resize_dim[1]-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        processed = detector.process_frame(frame)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', processed)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html', sample_videos=SAMPLE_VIDEOS)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['video_path'] = filepath
        return redirect(url_for('detect'))
    else:
        flash('Invalid file type')
        return redirect(url_for('index'))

@app.route('/sample/<sample_name>')
def use_sample(sample_name):
    if sample_name in SAMPLE_VIDEOS:
        session['video_path'] = SAMPLE_VIDEOS[sample_name]
        return redirect(url_for('detect'))
    else:
        flash('Sample not found')
        return redirect(url_for('index'))

@app.route('/detect')
def detect():
    if 'video_path' not in session:
        flash('No video selected')
        return redirect(url_for('index'))
    
    video_path = session['video_path']
    if not os.path.exists(video_path):
        flash('Video file not found')
        return redirect(url_for('index'))
    
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    if 'video_path' not in session:
        return Response()
    
    video_path = session['video_path']
    if not os.path.exists(video_path):
        return Response()
    
    return Response(generate_frames(video_path),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/violations')
def view_violations():
    violations = []
    try:
        with open(app.config['VIOLATIONS_CSV'], 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header row
            for row in reader:
                violations.append(row)
    except FileNotFoundError:
        flash('No violations recorded yet')
    
    return render_template('violations.html', violations=violations)

@app.route('/download_csv')
def download_csv():
    """Download the violations CSV file"""
    if not os.path.exists(app.config['VIOLATIONS_CSV']):
        flash('No violations data available for download')
        return redirect(url_for('view_violations'))
    
    # Generate a more descriptive filename with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    download_name = f"traffic_violations_{timestamp}.csv"
    
    return send_file(
        app.config['VIOLATIONS_CSV'],
        mimetype='text/csv',
        as_attachment=True,
        download_name=download_name
    )

if __name__ == '__main__':
    # Create samples directory if it doesn't exist
    os.makedirs('static/samples', exist_ok=True)
    
    # Create violations directory if it doesn't exist
    csv_dir = os.path.dirname(app.config['VIOLATIONS_CSV'])
    os.makedirs(csv_dir, exist_ok=True)
    
    app.run(debug=True, port=8080)