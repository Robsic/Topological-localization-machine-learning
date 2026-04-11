from dataclasses import dataclass

@dataclass
class FreeRouteConfig:
    # Output
    output_base_dir: str = "validation"
    output_prefix: str = "route_images"
    csv_name: str = "gps_log.csv"

    # Camera
    video_device: str = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
    fps: int = 10
    width: int = 1280
    height: int = 720

    # Preview
    enable_preview: bool = True
    preview_width: int = 1280
    preview_height: int = 720

    # Image output
    resize_output: bool = True
    output_width: int = 255
    output_height: int = 143

    # Save filtering
    min_save_dist_m: float = 1.5

    # GPS
    gps_port: str = "/dev/ttyACM0"
    gps_baud: int = 9600
    gps_timeout: float = 0.3
    gps_hdop_max: float = 5.0
    gps_window_size: int = 5
    gps_fix_timeout_s: float = 60.0

    # Models
    model_1_path: str = "src/Models/mobilenet_best.pth"
    model_2_path: str = "src/Models/efficientnet_b0_best_16.pth"
    device: str = "cpu"

    # Labeling / motion
    heading_window_size: int = 5
    intersection_enter_m: float = 23.0
    intersection_exit_m: float = 23.0
    curve_deg_threshold: float = 30.0
    min_speed_mps: float = 1.0
    min_step_dist_m: float = 1.0
    max_reasonable_mps: float = 13.89  # ~50 km/h
