import argparse

import cv2

from audio_output import AudioOutput
from compute_actions import ComputeActions
from control.follow_controller import get_drone_action
from detection_module import PERSON_LABEL, detect_objects, load_model
from object_locator import ObjectLocator
from person_tracker import (
    get_direction,
    get_distance_status,
    get_height_ratio,
    get_horizontal_error,
    select_target,
)
from safety_manager import SafetyManager
from session_logger import SessionLogger
from tracking_module import TrackingModule

WEIGHTS_PATH = "yolo11n.pt"
WINDOW_NAME = "Visual Buddy"

BOX_COLOR = (0, 255, 0)
TARGET_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect, track and announce people and objects in a video feed.")
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

    tracking = TrackingModule()
    actions = ComputeActions()
    locator = ObjectLocator()
    audio = AudioOutput()
    logger = SessionLogger()
    safety = SafetyManager()

    target_id = None

    logger.log("session_start", source=source)
    audio.say("Visual Buddy ready")

    try:
        ## Loop to continuously read frames and perform detection + tracking
        while True:

            success, frame = video.read()  # Read a frame from the video source

            if not success:
                print("No more frames (video ended or source unavailable)")
                break

            frame_height, frame_width = frame.shape[:2]  # Get frame dimensions

            ## Detection Module: every object in the frame, with ByteTrack IDs
            detections = detect_objects(model, frame)
            people = [d for d in detections if d.label == PERSON_LABEL]

            ## Tracking Module -> Compute Actions -> Audio Output (people)
            events = tracking.update(people, frame_width, frame_height)
            for event in events:
                logger.log("activity", track_id=event.track_id, activity=event.activity,
                           horizontal_error=round(event.horizontal_error, 2),
                           height_ratio=round(event.height_ratio, 2))

            for alert in actions.compute(events):
                logger.log("person_alert", message=alert)
                audio.say(alert)

            ## Object Locator -> Audio Output (everything that isn't a person)
            for label, position, message in locator.locate(detections, frame_width, frame_height):
                logger.log("object_alert", label=label, position=position)
                audio.say(message)

            ## Follow target (for the drone): stay locked on the same person while visible
            target = select_target(people, target_id)
            target_id = target.track_id if target else None

            command = None
            if target is not None:
                horizontal_error = get_horizontal_error(target.box, frame_width)
                height_ratio = get_height_ratio(target.box, frame_height)
                command = get_drone_action(horizontal_error, height_ratio)

            ## Every command goes through the safety manager before it could reach the drone
            safe_command = safety.filter(command, target_seen=target is not None)

            draw_overlay(frame, detections, target, tracking, safe_command, safety.status())
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1)
            ## 'q' ends the session, SPACE toggles the emergency stop
            if key == ord("q"):
                break
            if key == ord(" "):
                if safety.emergency_stopped:
                    safety.reset_emergency_stop()
                    logger.log("emergency_stop_reset")
                else:
                    safety.trigger_emergency_stop()
                    logger.log("emergency_stop")
                    audio.say("Emergency stop")
    finally:
        ## Release video source, close windows, save the log and confirm to the user
        video.release()
        cv2.destroyAllWindows()

        log_path = logger.close()
        print(f"Session log saved to {log_path}")

        audio.say("Session ended")
        audio.close()


def draw_overlay(frame, detections, target, tracking, command, safety_status):
    frame_height, frame_width = frame.shape[:2]

    for detection in detections:
        label = detection.label
        if detection.track_id is not None:
            label += f" #{detection.track_id}"
            activity = tracking.activity_of(detection.track_id)
            if activity:
                label += f" {activity}"
        draw_box(frame, detection.box, label, BOX_COLOR)

    if target is not None:
        horizontal_error = get_horizontal_error(target.box, frame_width)
        height_ratio = get_height_ratio(target.box, frame_height)

        draw_box(frame, target.box, f"TARGET #{target.track_id}", TARGET_COLOR)
        draw_text(frame, f"Direction: {get_direction(horizontal_error)}", (30, 50))
        draw_text(frame, f"Distance: {get_distance_status(height_ratio)} ({height_ratio:.2f})", (30, 90))

    draw_text(frame, f"Command: {command}", (30, 130), font_scale=0.8)
    draw_text(frame, f"Safety: {safety_status}", (30, 170), font_scale=0.8,
              color=TEXT_COLOR if safety_status == "OK" else ALERT_COLOR)


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = box

    ## Draw bounding box + label around the object
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
