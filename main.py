import os

from src.capture.capture_runner import CaptureRunner
from src.config.capture_config import CaptureConfig

from src.Navegation.free_route_runner import FreeRouteRunner
from src.config.free_route_config import FreeRouteConfig


def start_capture(nodes, mode, should_stop, on_stop_callback=None):
    """
    Entry-point called by GUI.
    Keeps backward compatibility with interface.py.
    """

    config = CaptureConfig(
        # Camera
        video_device=os.getenv("VIDEO_DEVICE", "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"),
        fps=int(os.getenv("VIDEO_FPS", "10")),
        width=int(os.getenv("VIDEO_WIDTH", "1280")),
        height=int(os.getenv("VIDEO_HEIGHT", "720")),

        # Output
        resize_output=os.getenv("OUTPUT_RESIZE", "1") == "1",
        output_width=int(os.getenv("OUTPUT_WIDTH", "255")),
        output_height=int(os.getenv("OUTPUT_HEIGHT", "143")),

        # Preview
        enable_preview=os.getenv("PREVIEW_ENABLE", "1") == "1",
        preview_width=int(os.getenv("PREVIEW_WIDTH", "1280")),
        preview_height=int(os.getenv("PREVIEW_HEIGHT", "720")),

        # GPS
        gps_port=os.getenv("GPS_PORT", "/dev/ttyACM0"),
        gps_baud=int(os.getenv("GPS_BAUD", "9600")),
    )

    runner = CaptureRunner(config)
    result = runner.run(nodes=nodes, mode=mode, should_stop=should_stop, on_stop_callback=on_stop_callback)

    if result.output_dir:
        print(f"{result.frames_saved} images saved in '{result.output_dir}'")
    else:
        print("Capture finished without output directory (GPS fix not acquired or stop requested).")


def start_free_route(nodes, should_stop, on_stop_callback=None):
    config = FreeRouteConfig(
        # Camera
        video_device=os.getenv("VIDEO_DEVICE", "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"),
        fps=int(os.getenv("VIDEO_FPS", "10")),
        width=int(os.getenv("VIDEO_WIDTH", "1280")),
        height=int(os.getenv("VIDEO_HEIGHT", "720")),

        # Preview
        enable_preview=os.getenv("PREVIEW_ENABLE", "1") == "1",
        preview_width=int(os.getenv("PREVIEW_WIDTH", "1280")),
        preview_height=int(os.getenv("PREVIEW_HEIGHT", "720")),

        # GPS
        gps_port=os.getenv("GPS_PORT", "/dev/ttyACM0"),
        gps_baud=int(os.getenv("GPS_BAUD", "9600")),

        # Output (1)
        output_base_dir="validation",
        resize_output=os.getenv("OUTPUT_RESIZE", "1") == "1",
        output_width=int(os.getenv("OUTPUT_WIDTH", "255")),
        output_height=int(os.getenv("OUTPUT_HEIGHT", "143")),

        # Models (2) (3)
        model_1_path=os.getenv("MODEL_1_PATH", "src/Models/mobilenet_best.pth"),
        model_2_path=os.getenv("MODEL_2_PATH", "src/Models/efficientnet_b0_best_16.pth"),
    )

    runner = FreeRouteRunner(config)
    result = runner.run(nodes=nodes, should_stop=should_stop, on_stop_callback=on_stop_callback)

    if result.output_dir:
        print(f"{result.frames_saved} validation frames saved in '{result.output_dir}' (reason={result.reason})")
    else:
        print(f"Free route finished without output directory (reason={result.reason})")