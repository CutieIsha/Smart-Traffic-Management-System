import cv2
import os
import numpy as np
import random
import csv
import datetime
from ultralytics import YOLO
from tracker import Tracker

class TrafficViolationDetector:
    def __init__(self, model_path="yolov8m.pt", confidence=0.75, csv_path="violations.csv"):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.coco_names = self.model.model.names
        self.target_labels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]
        
        # Initialize tracker and counters
        self.tracker = Tracker()
        self.violation_counter = 0
        self.vehicle_info = {}
        
        # Define brightness threshold for light detection
        self.brightness_threshold = 128
        
        # CSV file setup for violations
        self.csv_path = csv_path
        self.initialize_csv()
        
    def initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['ID', 'License Plate', 'Vehicle Type', 'Timestamp', 'Location'])
                
    def generate_maharashtra_license_plate(self):
        """Generate a random Maharashtra license plate number in the format MH-XX-XX-XXXX"""
        # Maharashtra state code
        state_code = "MH"
        
        # District code (01-50)
        district_code = f"{random.randint(1, 50):02d}"
        
        # Series code (A-Z)
        series_code = random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") + random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        
        # Registration number (0000-9999)
        reg_number = f"{random.randint(0, 9999):04d}"
        
        return f"{state_code}-{district_code}-{series_code}-{reg_number}"
        
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

    def record_violation(self, vehicle_type):
        """Record violation with random license plate to CSV"""
        license_plate = self.generate_maharashtra_license_plate()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location = "Junction XYZ" # This could be parameterized or detected
        
        with open(self.csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.violation_counter, license_plate, vehicle_type, timestamp, location])
            
        return license_plate

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
                cls_id = int(cls)
                if self.coco_names[cls_id] in self.target_labels:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append((x1, y1, x2, y2, cls_id))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), [0,255,0], 2)
        
        # Track objects
        tracked_objects = self.tracker.update([det[:4] for det in detections])
        
        # Map cls_id back to tracked objects
        tracked_with_class = []
        for tracked_obj in tracked_objects:
            x1, y1, x2, y2, obj_id = tracked_obj
            # Find matching detection to get class
            for det in detections:
                det_x1, det_y1, det_x2, det_y2, cls_id = det
                if abs(x1 - det_x1) < 10 and abs(y1 - det_y1) < 10:  # Simple matching based on position
                    tracked_with_class.append((x1, y1, x2, y2, obj_id, cls_id))
                    break
            else:
                # If no match found, use previous class if available
                if obj_id in self.vehicle_info and 'class_id' in self.vehicle_info[obj_id]:
                    tracked_with_class.append((x1, y1, x2, y2, obj_id, self.vehicle_info[obj_id]['class_id']))
                else:
                    # Default to car if no class info available
                    tracked_with_class.append((x1, y1, x2, y2, obj_id, 2))  # 2 is usually car in COCO
        
        # Process each object
        for obj in tracked_with_class:
            x1, y1, x2, y2, obj_id, cls_id = obj
            center = ((x1+x2)//2, (y1+y2)//2)
            vehicle_type = self.coco_names[cls_id]
            
            if obj_id not in self.vehicle_info:
                self.vehicle_info[obj_id] = {
                    'positions': [],
                    'entered_roi': False,
                    'violated': False,
                    'counted': False,
                    'class_id': cls_id,
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
                            # Generate and store license plate
                            license_plate = self.record_violation(vehicle_type)
                            self.vehicle_info[obj_id]['license_plate'] = license_plate
                            self.vehicle_info[obj_id]['counted'] = True
                        
                        # Display license plate on frame
                        cv2.rectangle(frame, (x1,y1), (x2,y2), [0,0,255], 2)
                        plate_text = f"Violation! {self.vehicle_info[obj_id]['license_plate']}"
                        cv2.putText(frame, plate_text, (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
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

    def process_video(self, video_path, output_path=None, resize_dim=(1100,700)):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return False
            
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 
                                cap.get(cv2.CAP_PROP_FPS), resize_dim)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Processed {frame_count} frames")
                break
                
            frame_count += 1
            frame = cv2.resize(frame, resize_dim)
            processed = self.process_frame(frame)
            
            if output_path:
                out.write(processed)
                
            cv2.imshow("Traffic Violation Detection", processed)
            if cv2.waitKey(1) == 27:
                break
                
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        return True

if __name__ == "__main__":
    detector = TrafficViolationDetector(csv_path="traffic_violations.csv")
    
    red_light = [[998,125], [998,155], [972,152], [970,127]]
    green_light = [[971,200], [996,200], [1001,228], [971,230]]
    roi = [[910,372], [388,365], [338,428], [917,441]]
    
    detector.set_regions(red_light, green_light, roi)
    
    video_path = r"C:\Users\hp\Downloads\Traffic-Signal-Violation-Detection-main\Traffic-Signal-Violation-Detection-main\videos\aziz2.mp4"
    detector.process_video(video_path)