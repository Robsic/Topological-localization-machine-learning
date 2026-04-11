import os
import csv
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
from geopy.distance import geodesic

from src.gps.gps_reader import GPSReader
from src.capture.video_capture import VideoCapture
from src.tracker.node_tracker import NodeTracker
from src.utils.file_manager import create_numbered_dir
from src.Navegation.route_predictor import RoutePredictor
from src.capture.preview_opencv import OpenCVPreview
from src.tracker.motion_labeler import MotionLabeler
from src.config.free_route_config import FreeRouteConfig   # ajuste se necessário


@dataclass
class FreeRouteResult:
    output_dir: str = ""
    frames_saved: int = 0
    reason: str = ""


class FreeRouteRunner:
    """
    Free-route runner for model validation.

    - Saves output inside validation/ (configurable base dir)
    - CSV contains ALL columns from CaptureRunner:
        filename, timestamp, latitude, longitude, distance_to_node_m, node_index,
        speed_mps, heading_deg, label
      plus:
        predicted_label, predicted_node_index

    - Uses the SAME filtering rules:
        * wait for GPS fix
        * MotionLabeler discard logic (speed anomalies, etc.)
        * min distance between saved frames (min_save_dist_m)
        * OpenCV preview with 'q' to stop (optional)
    """

    def __init__(self, config: FreeRouteConfig):
        self.cfg = config

    def run(self, nodes, should_stop, on_stop_callback=None) -> FreeRouteResult:
        result = FreeRouteResult()

        print("Operation mode: FREE ROUTE (VALIDATION)")
        print("Capture started. Press 'q' to stop (preview window).")

        tracker = NodeTracker(nodes)

        gps = self._create_gps()
        try:
            if not self._wait_for_gps_fix(gps, should_stop):
                result.reason = "gps_fix_not_acquired"
                return result

            camera = self._create_camera()
            preview = self._create_preview()

            output_dir = create_numbered_dir(
                base_dir=self.cfg.output_base_dir,
                prefix=self.cfg.output_prefix,
            )
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            csv_path = os.path.join(output_dir, self.cfg.csv_name)

            predictor = RoutePredictor(
                model_1_path=self.cfg.model_1_path,
                model_2_path=self.cfg.model_2_path,
                device=self.cfg.device,
            )

            labeler = self._create_labeler()

            result.output_dir = output_dir

            frames_saved = 0
            last_saved_coords = None

            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "filename",
                        "timestamp",
                        "latitude",
                        "longitude",
                        "distance_to_node_m",
                        "node_index",
                        "speed_mps",
                        "heading_deg",
                        "label",
                        "predicted_label",
                        "predicted_node_index",
                    ])

                    while True:
                        if should_stop():
                            print("Stop requested. Finishing.")
                            result.reason = "stop_requested"
                            break

                        frame = camera.read_frame()

                        coords = gps.get_coordinates()
                        current_time = time.time()

                        # Defaults
                        latitude, longitude = "not_found", "not_found"
                        if coords:
                            latitude, longitude = coords

                        # Node info (needed to build ground-truth columns)
                        node_index, distance = None, None
                        if coords:
                            node_index, distance = tracker.check_node(latitude, longitude)

                        # Free route always considers saving, but filters apply below.
                        save_frame = True

                        # Motion/label (same logic as CaptureRunner)
                        motion = labeler.update(
                            coords=coords,
                            distance_to_node_m=distance,
                            current_time=current_time
                        )

                        # Discard frame if motion labeler says so (speed anomaly etc.)
                        if motion.discard:
                            if preview is not None:
                                if not preview.show(frame, fps=self.cfg.fps):
                                    print("Stopped by user ('q').")
                                    result.reason = "user_pressed_q"
                                    break
                            continue

                        # Min distance filter (same as CaptureRunner)
                        if save_frame and coords:
                            if last_saved_coords is not None:
                                dist_from_last = geodesic((latitude, longitude), last_saved_coords).meters
                                if dist_from_last < self.cfg.min_save_dist_m:
                                    if preview is not None:
                                        if not preview.show(frame, fps=self.cfg.fps):
                                            print("Stopped by user ('q').")
                                            result.reason = "user_pressed_q"
                                            break
                                    continue
                        
                        # Resize frame for model input
                        frame_resized = cv2.resize(
                            frame,
                            (self.cfg.output_width, self.cfg.output_height),  # (W, H)
                            interpolation=cv2.INTER_LINEAR
                        )

                        # Predictions (models)
                        pred = predictor.predict(frame_resized)
                        predicted_label = pred.predicted_label
                        predicted_node_index = pred.predicted_node_index  # "NA" if straight

                        # Save image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"frame_{frames_saved}_lat-{latitude}_lon-{longitude}.png"
                        img_path = os.path.join(images_dir, filename)

                        if self.cfg.resize_output:
                            resized = cv2.resize(frame, (self.cfg.output_width, self.cfg.output_height))
                            cv2.imwrite(img_path, resized)
                        else:
                            cv2.imwrite(img_path, frame)

                        # Prepare CaptureRunner-like fields
                        dist_field = f"{distance:.2f}" if (distance is not None and isinstance(distance, (int, float))) else ""
                        node_field = str(node_index) if node_index is not None else ""

                        speed_field = f"{motion.speed_mps:.2f}" if motion.speed_mps is not None else ""
                        heading_field = f"{motion.heading_deg:.2f}" if motion.heading_deg is not None else ""
                        label_field = motion.label or ""

                        # Write CSV row (same + extras)
                        writer.writerow([
                            filename,
                            timestamp,
                            str(latitude),
                            str(longitude),
                            dist_field,
                            node_field,
                            speed_field,
                            heading_field,
                            label_field,
                            predicted_label,
                            predicted_node_index,
                        ])

                        frames_saved += 1
                        result.frames_saved = frames_saved

                        if coords:
                            last_saved_coords = (latitude, longitude)

                        print(
                            f"Saved: {filename} — true={label_field or 'unlabeled'}\n"
                            f"real={label_field}:{node_field}\n"
                            f"pred={predicted_label}:{predicted_node_index}\n"
                        )

                        # Preview (optional)
                        if preview is not None:
                            if not preview.show(frame, fps=self.cfg.fps):
                                print("Stopped by user ('q').")
                                result.reason = "user_pressed_q"
                                break

            finally:
                try:
                    camera.release()
                except Exception:
                    pass

                try:
                    if preview is not None:
                        preview.close()
                except Exception:
                    pass

                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            if on_stop_callback:
                try:
                    on_stop_callback()
                except Exception:
                    pass

            return result

        finally:
            try:
                gps.close()
            except Exception:
                pass

    def _create_gps(self) -> GPSReader:
        gps_port = os.getenv("GPS_PORT", self.cfg.gps_port)
        gps_baud = int(os.getenv("GPS_BAUD", str(self.cfg.gps_baud)))

        return GPSReader(
            port=gps_port,
            baud=gps_baud,
            timeout=self.cfg.gps_timeout,
            hdop_max=self.cfg.gps_hdop_max,
            window_size=self.cfg.gps_window_size,
        )

    def _wait_for_gps_fix(self, gps: GPSReader, should_stop) -> bool:
        gps_port = getattr(gps, "port", self.cfg.gps_port)
        gps_baud = getattr(gps, "baud", self.cfg.gps_baud)

        print(f"Waiting for GPS fix on {gps_port} @ {gps_baud} ...")
        t0 = time.time()

        while True:
            coords = gps.get_coordinates()
            if coords:
                print(f"GPS fix acquired: {coords}")
                return True

            if should_stop():
                print("Stop requested while waiting for GPS fix.")
                return False

            if time.time() - t0 > self.cfg.gps_fix_timeout_s:
                print(f"GPS fix timeout ({self.cfg.gps_fix_timeout_s:.0f}s).")
                return False

            time.sleep(0.1)

    def _create_camera(self) -> VideoCapture:
        video_device = os.getenv("VIDEO_DEVICE", self.cfg.video_device)

        return VideoCapture(
            device=video_device,
            fps=self.cfg.fps,
            width=self.cfg.width,
            height=self.cfg.height,
        )

    def _create_preview(self):
        if not self.cfg.enable_preview:
            return None
        return OpenCVPreview(width=self.cfg.preview_width, height=self.cfg.preview_height)

    def _create_labeler(self) -> MotionLabeler:
        return MotionLabeler(
            heading_window_size=self.cfg.heading_window_size,
            intersection_enter_m=self.cfg.intersection_enter_m,
            intersection_exit_m=self.cfg.intersection_exit_m,
            curve_deg_threshold=self.cfg.curve_deg_threshold,
            min_speed_mps=self.cfg.min_speed_mps,
            min_step_dist_m=self.cfg.min_step_dist_m,
            max_reasonable_mps=self.cfg.max_reasonable_mps,
        )

