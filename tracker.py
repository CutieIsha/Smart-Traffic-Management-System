import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class KalmanFilter:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = None
        self.age = 0
        self.missed = 0
        self.history = []

    def update(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.age += 1
        self.missed = 0
        self.history.append((x, y, w, h))

    def predict(self):
        self.missed += 1
        return (self.x, self.y, self.w, self.h)

class Tracker:
    def __init__(self, max_missed=10):  # Increased max_missed to avoid premature deletion
        self.trackers = []
        self.next_id = 1
        self.max_missed = max_missed
        self.violation_history = set()  # To store vehicle IDs that violated
    
    def update(self, detections):
        detections = np.array(detections)
        if len(detections) == 0:
            for tracker in self.trackers:
                tracker.predict()
            self.trackers = [t for t in self.trackers if t.missed < self.max_missed or t.age < 3]
            return []
        
        tracker_boxes = np.array([t.predict() for t in self.trackers])
        
        if len(tracker_boxes) > 0:
            iou_matrix = self.compute_iou(tracker_boxes, detections)
            row_idx, col_idx = linear_sum_assignment(-iou_matrix)
        else:
            row_idx, col_idx = np.array([]), np.array([])
        
        assigned_detections = set()
        for t_idx, d_idx in zip(row_idx, col_idx):
            if iou_matrix[t_idx, d_idx] > 0.3:  # IoU threshold
                self.trackers[t_idx].update(*detections[d_idx])
                assigned_detections.add(d_idx)
            
        for d_idx, det in enumerate(detections):
            if d_idx not in assigned_detections:
                new_tracker = KalmanFilter(*det)
                new_tracker.id = self.next_id
                self.next_id += 1
                self.trackers.append(new_tracker)
                
        self.trackers = [t for t in self.trackers if t.missed < self.max_missed or t.age < 3]

        return [(t.x, t.y, t.w, t.h, t.id) for t in self.trackers]
    
    def is_new_violation(self, obj_id):
        """Check if this object ID has already violated."""
        if obj_id not in self.violation_history:
            self.violation_history.add(obj_id)
            return True
        return False

    def compute_iou(self, boxesA, boxesB):
        iou_matrix = np.zeros((len(boxesA), len(boxesB)))
        for i, a in enumerate(boxesA):
            for j, b in enumerate(boxesB):
                iou_matrix[i, j] = self.iou(a, b)
        return iou_matrix
    
    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])  # Correct width calculation
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])  # Correct height calculation
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return max(0, min(1, iou))  # Ensure IoU is between 0 and 1
