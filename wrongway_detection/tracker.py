import numpy as np
from scipy.spatial import distance

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        new_center_points = {}

        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check if this object already exists
            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                if distance.euclidean((cx, cy), pt) < 50:  # 50-pixel threshold
                    new_center_points[obj_id] = (cx, cy)
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.id_count += 1
                new_center_points[self.id_count] = (cx, cy)

        self.center_points = new_center_points.copy()
        tracked_objects = [[x1, y1, x2, y2, obj_id] for obj_id, (cx, cy) in new_center_points.items()]
        return tracked_objects
