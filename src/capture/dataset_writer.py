import csv
import os
import cv2
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FrameRecord:
    filename: str
    timestamp: str
    latitude: str
    longitude: str
    distance_to_node_m: str
    node_index: str
    speed_mps: str
    heading_deg: str
    label: str


class DatasetWriter:
    def __init__(self, output_dir: str, resize_output: bool, out_w: int, out_h: int):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.csv_path = os.path.join(output_dir, "gps_log.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            "filename", "timestamp", "latitude", "longitude",
            "distance_to_node_m", "node_index", "speed_mps",
            "heading_deg", "label"
        ])

        self.resize_output = resize_output
        self.out_w = out_w
        self.out_h = out_h

    def make_filename(self, frame_count: int, latitude, longitude):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"frame_{frame_count}_lat-{latitude}_lon-{longitude}.png"
        return filename, ts

    def save_frame(self, frame, filename: str):
        path = os.path.join(self.images_dir, filename)
        if self.resize_output:
            frame = cv2.resize(frame, (self.out_w, self.out_h))
        cv2.imwrite(path, frame)
        return path

    def write_record(self, record: FrameRecord):
        self._writer.writerow([
            record.filename,
            record.timestamp,
            record.latitude,
            record.longitude,
            record.distance_to_node_m,
            record.node_index,
            record.speed_mps,
            record.heading_deg,
            record.label,
        ])

    def close(self):
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass

