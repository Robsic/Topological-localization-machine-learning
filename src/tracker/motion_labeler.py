from collections import deque
from dataclasses import dataclass
from geopy.distance import geodesic

from src.gps.gps_utils import compute_heading, circular_mean_deg, angular_diff_deg


@dataclass
class MotionUpdate:
    label: str
    speed_mps: float | None
    heading_deg: float | None
    discard: bool
    discard_reason: str | None


class MotionLabeler:
    def __init__(
        self,
        heading_window_size: int,
        intersection_enter_m: float,
        intersection_exit_m: float,
        curve_deg_threshold: float,
        min_speed_mps: float,
        min_step_dist_m: float,
        max_reasonable_mps: float,
    ):
        self.heading_window = deque(maxlen=heading_window_size)
        self.in_intersection_gate = False

        self.intersection_enter_m = intersection_enter_m
        self.intersection_exit_m = intersection_exit_m
        self.curve_deg_threshold = curve_deg_threshold
        self.min_speed_mps = min_speed_mps
        self.min_step_dist_m = min_step_dist_m
        self.max_reasonable_mps = max_reasonable_mps

        self.last_coords_motion = None
        self.last_time = None

    def update(self, coords, distance_to_node_m: float | None, current_time: float) -> MotionUpdate:
        if coords is None:
            return MotionUpdate(label="", speed_mps=None, heading_deg=None, discard=False, discard_reason=None)

        lat, lon = coords

        # First sample: cannot compute speed/heading yet
        if self.last_coords_motion is None or self.last_time is None:
            self.last_coords_motion = (lat, lon)
            self.last_time = current_time
            # label remains empty until we have motion context; intersection can still be computed if distance exists
            label = "intersection" if self._intersection_state(distance_to_node_m) else ""
            return MotionUpdate(label=label, speed_mps=None, heading_deg=None, discard=False, discard_reason=None)

        step_dist_m = geodesic((lat, lon), self.last_coords_motion).meters
        dt = current_time - self.last_time
        speed = (step_dist_m / dt) if dt > 0 else 0.0

        # Update motion reference early (even if discard)
        prev_coords = self.last_coords_motion
        self.last_coords_motion = (lat, lon)
        self.last_time = current_time

        if speed > self.max_reasonable_mps:
            return MotionUpdate(
                label="",
                speed_mps=speed,
                heading_deg=None,
                discard=True,
                discard_reason="unreasonable_speed",
            )

        inst_heading = compute_heading(prev_coords, (lat, lon))
        self.heading_window.append(inst_heading)
        sm_heading = circular_mean_deg(list(self.heading_window))

        in_intersection = self._intersection_state(distance_to_node_m)

        if in_intersection:
            return MotionUpdate(label="intersection", speed_mps=speed, heading_deg=sm_heading, discard=False, discard_reason=None)

        # Otherwise straight/curve
        is_curve = False
        if sm_heading is not None and len(self.heading_window) >= 2:
            prev_sm_heading = circular_mean_deg(list(self.heading_window)[:-1])
            if prev_sm_heading is not None:
                dpsi = angular_diff_deg(sm_heading, prev_sm_heading)
                if (
                    dpsi > self.curve_deg_threshold
                    and speed >= self.min_speed_mps
                    and step_dist_m >= self.min_step_dist_m
                ):
                    is_curve = True

        return MotionUpdate(
            label="curve" if is_curve else "straight",
            speed_mps=speed,
            heading_deg=sm_heading,
            discard=False,
            discard_reason=None,
        )

    def _intersection_state(self, distance_to_node_m: float | None) -> bool:
        if distance_to_node_m is None:
            return self.in_intersection_gate

        if self.in_intersection_gate:
            if distance_to_node_m > self.intersection_exit_m:
                self.in_intersection_gate = False
        else:
            if distance_to_node_m < self.intersection_enter_m:
                self.in_intersection_gate = True

        return self.in_intersection_gate

