import argparse

import cv2

from control.follow_controller import get_drone_action
from person_detector import PERSON_LABEL, detect_objects, load_model
from person_tracker import (
    get_direction,
    get_distance_status,
    get_height_ratio,
    get_horizontal_error,
    select_target,
)
from safety_manager import SafetyManager

WEIGHTS_PATH = "yolo11n.pt"
WINDOW_NAME = "DroneBuddy"

BOX_COLOR = (0, 255, 0)
TARGET_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and follow a person in a video feed.")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0), path to a recorded DJI clip, or a stream URL. Default: 0",
    )
    return parser.parse_args()


def open_source(source):
    ## A plain number means a local camera, anything else is a file path / stream URL
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def run(source):
    model = load_model(WEIGHTS_PATH)
    video = open_source(source)
    safety = SafetyManager()

    try:
        ## Loop to continuously read frames and perform YOLO detection
        while True:

            success, frame = video.read()  # Read a frame from the video source

            if not success:
                print("No more frames (video ended or source unavailable)")
                break

            frame_height, frame_width = frame.shape[:2]  # Get frame dimensions

            detections = detect_objects(model, frame)
            for label, box in detections:
                draw_box(frame, box, label, BOX_COLOR)

            ## Following only makes sense for one person - pick them out
            people = [box for label, box in detections if label == PERSON_LABEL]
            target = select_target(people)

            command = None
            if target is not None:
                horizontal_error = get_horizontal_error(target, frame_width)
                height_ratio = get_height_ratio(target, frame_height)
                command = get_drone_action(horizontal_error, height_ratio)

                draw_box(frame, target, "TARGET", TARGET_COLOR)
                draw_text(frame, f"Direction: {get_direction(horizontal_error)}", (30, 50))
                draw_text(frame, f"Distance: {get_distance_status(height_ratio)} ({height_ratio:.2f})", (30, 90))

            ## Every command goes through the safety manager before it could reach the drone
            safe_command = safety.filter(command, target_seen=target is not None)

            draw_text(frame, f"Command: {safe_command}", (30, 130), font_scale=0.8)
            status = safety.status()
            draw_text(frame, f"Safety: {status}", (30, 170), font_scale=0.8,
                      color=TEXT_COLOR if status == "OK" else ALERT_COLOR)

            ## Display the frame
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1)
            ## 'q' quits, SPACE toggles the emergency stop
            if key == ord("q"):
                break
            if key == ord(" "):
                if safety.emergency_stopped:
                    safety.reset_emergency_stop()
                else:
                    safety.trigger_emergency_stop()
    finally:
        ## Release video source and close OpenCV windows
        video.release()
        cv2.destroyAllWindows()


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = box

    ## Draw bounding box + class label around the object
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        2
    )

    draw_text(frame, label, (int(x1), int(y1) - 10), font_scale=0.6, color=color)


def draw_text(frame, text, position, font_scale=1.0, color=TEXT_COLOR):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2
    )


if __name__ == "__main__":
    run(parse_args().source)
