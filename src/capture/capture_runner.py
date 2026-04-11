import os
import time
from dataclasses import dataclass
from geopy.distance import geodesic

from src.gps.gps_reader import GPSReader
from src.tracker.node_tracker import NodeTracker
from src.tracker.motion_labeler import MotionLabeler
from src.capture.video_capture import VideoCapture
from src.capture.dataset_writer import DatasetWriter, FrameRecord
from src.capture.preview_opencv import OpenCVPreview
from src.utils.file_manager import create_numbered_dir
from src.config.capture_config import CaptureConfig


@dataclass
class CaptureResult:
    output_dir: str
    frames_saved: int


class CaptureRunner:
    def __init__(self, config: CaptureConfig):
        self.config = config

    def run(self, nodes, mode: str, should_stop, on_stop_callback=None) -> CaptureResult:
        mode = mode.lower().strip()
        if mode not in ("all", "nodes_only"):
            raise ValueError("mode must be 'all' or 'nodes_only'")

        print(f"Operation mode: {mode.upper()}")
        print("Capture started. Press 'q' to stop (preview window).")

        tracker = NodeTracker(nodes)

        gps = self._create_gps()
        try:
            if not self._wait_for_gps_fix(gps, should_stop):
                return CaptureResult(output_dir="", frames_saved=0)

            camera = self._create_camera()
            preview = self._create_preview()
            output_dir = create_numbered_dir()

            writer = DatasetWriter(
                output_dir=output_dir,
                resize_output=self.config.resize_output,
                out_w=self.config.output_width,
                out_h=self.config.output_height,
            )

            labeler = self._create_labeler()

            frames_saved = 0
            last_saved_coords = None

            try:
                while True:
                    if should_stop():
                        print("Stop requested. Finishing.")
                        break

                    frame = camera.read_frame()

                    coords = gps.get_coordinates()
                    current_time = time.time()

                    # Default fields
                    latitude, longitude = "not_found", "not_found"
                    if coords:
                        latitude, longitude = coords

                    # Determine node distance/index if coords exist
                    node_index, distance = None, None
                    if coords:
                        node_index, distance = tracker.check_node(latitude, longitude)

                    # Decide whether to save
                    save_frame = False
                    if mode == "all":
                        save_frame = True
                    elif mode == "nodes_only":
                        if node_index is not None:
                            save_frame = True

                    # Motion label update
                    motion = labeler.update(coords, distance_to_node_m=distance, current_time=current_time)
                    if motion.discard:
                        # Preview even if discarded, so you can see what is happening
                        if preview is not None:
                            if not preview.show(frame, fps=self.config.fps):
                                print("Stopped by user ('q').")
                                break
                        continue

                    # Distance filter between saved frames (only if coords exist)
                    if save_frame and coords:
                        if last_saved_coords is not None:
                            dist_from_last = geodesic((latitude, longitude), last_saved_coords).meters
                            if dist_from_last < self.config.min_save_dist_m:
                                if preview is not None:
                                    if not preview.show(frame, fps=self.config.fps):
                                        print("Stopped by user ('q').")
                                        break
                                continue

                    # Save/log
                    if save_frame:
                        filename, timestamp = writer.make_filename(frames_saved, latitude, longitude)
                        writer.save_frame(frame, filename)

                        dist_field = f"{distance:.2f}" if (distance is not None and isinstance(distance, (int, float))) else ""
                        node_field = str(node_index) if node_index is not None else ""

                        speed_field = f"{motion.speed_mps:.2f}" if motion.speed_mps is not None else ""
                        heading_field = f"{motion.heading_deg:.2f}" if motion.heading_deg is not None else ""

                        record = FrameRecord(
                            filename=filename,
                            timestamp=timestamp,
                            latitude=str(latitude),
                            longitude=str(longitude),
                            distance_to_node_m=dist_field,
                            node_index=node_field,
                            speed_mps=speed_field,
                            heading_deg=heading_field,
                            label=motion.label or "",
                        )
                        writer.write_record(record)

                        frames_saved += 1
                        if coords:
                            last_saved_coords = (latitude, longitude)

                        print(f"Saved: {filename} — {motion.label or 'unlabeled'}")
                    else:
                        print(f"[{mode}] Frame skipped.")

                    # nodes_only completion
                    if mode == "nodes_only" and tracker.all_visited():
                        print("All nodes visited. Finishing.")
                        break

                    # Preview
                    if preview is not None:
                        if not preview.show(frame, fps=self.config.fps):
                            print("Stopped by user ('q').")
                            break

            finally:
                try:
                    writer.close()
                except Exception:
                    pass
                try:
                    camera.release()
                except Exception:
                    pass
                try:
                    if preview is not None:
                        preview.close()
                except Exception:
                    pass

            if on_stop_callback:
                try:
                    on_stop_callback()
                except Exception:
                    pass

            return CaptureResult(output_dir=output_dir, frames_saved=frames_saved)

        finally:
            try:
                gps.close()
            except Exception:
                pass

    def _create_gps(self) -> GPSReader:
        # Allow overriding via env vars
        gps_port = os.getenv("GPS_PORT", self.config.gps_port)
        gps_baud = int(os.getenv("GPS_BAUD", str(self.config.gps_baud)))

        return GPSReader(
            port=gps_port,
            baud=gps_baud,
            timeout=self.config.gps_timeout,
            hdop_max=self.config.gps_hdop_max,
            window_size=self.config.gps_window_size,
        )

    def _wait_for_gps_fix(self, gps: GPSReader, should_stop) -> bool:
        gps_port = getattr(gps, "port", "unknown")
        gps_baud = getattr(gps, "baud", "unknown")

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

            if time.time() - t0 > self.config.gps_fix_timeout_s:
                print(f"GPS fix timeout ({self.config.gps_fix_timeout_s:.0f}s).")
                return False

            time.sleep(0.1)

    def _create_camera(self) -> VideoCapture:
        return VideoCapture(
            device=self.config.video_device,
            fps=self.config.fps,
            width=self.config.width,
            height=self.config.height,
        )

    def _create_preview(self):
        if not self.config.enable_preview:
            return None
        return OpenCVPreview(width=self.config.preview_width, height=self.config.preview_height)

    def _create_labeler(self) -> MotionLabeler:
        return MotionLabeler(
            heading_window_size=self.config.heading_window_size,
            intersection_enter_m=self.config.intersection_enter_m,
            intersection_exit_m=self.config.intersection_exit_m,
            curve_deg_threshold=self.config.curve_deg_threshold,
            min_speed_mps=self.config.min_speed_mps,
            min_step_dist_m=self.config.min_step_dist_m,
            max_reasonable_mps=self.config.max_reasonable_mps,
        )

