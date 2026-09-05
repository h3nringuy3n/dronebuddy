import cv2

from control.follow_controller import get_drone_action
from person_detector import detect_people, load_model
from person_tracker import get_direction, get_distance_status

WEIGHTS_PATH = "yolo11n.pt"
WINDOW_NAME = "YOLO Detection"


def run():
    model = load_model(WEIGHTS_PATH)
    camera = cv2.VideoCapture(0)

    try:
        ## Loop to continuously read frames from the camera and perform YOLO detection
        while True:

            success, frame = camera.read()  # Read a frame from the camera

            if not success:
                print("Could not read from camera")
                break

            frame_width = frame.shape[1]  # Get frame dimensions

            for box in detect_people(model, frame):
                direction = get_direction(box, frame_width)
                distance_status = get_distance_status(box)
                action = get_drone_action(direction, distance_status)

                draw_overlay(frame, box, direction, distance_status, action)
                log_detection(box, direction, distance_status, action)

            ## Display the frame
            cv2.imshow(WINDOW_NAME, frame)

            ## Exit when 'q' is pressed
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        ## Release camera and close OpenCV windows
        camera.release()
        cv2.destroyAllWindows()


def draw_overlay(frame, box, direction, distance_status, action):
    x1, y1, x2, y2 = box

    ## Draw bounding box around person
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2
    )

    ## Display direction
    draw_text(frame, f"Direction: {direction}", 50)

    ## Display distance
    draw_text(frame, f"Distance: {distance_status}", 90)

    ## Display drone action
    draw_text(frame, f"Action: {action}", 130, font_scale=0.8)


def draw_text(frame, text, y, font_scale=1.0):
    cv2.putText(
        frame,
        text,
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        2
    )


def log_detection(box, direction, distance_status, action):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    ## Print information to Terminal
    print("PERSON FOUND!")
    print("Box:", x1, y1, x2, y2)
    print("Center:", center_x, center_y)
    print("Direction:", direction)
    print("Distance Status:", distance_status)
    print("Drone Action:", action)


if __name__ == "__main__":
    run()
