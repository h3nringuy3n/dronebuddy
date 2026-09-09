import cv2

from control.follow_controller import get_drone_action
from person_detector import PERSON_LABEL, detect_objects, load_model
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

            for label, box in detect_objects(model, frame):
                draw_box(frame, box, label)

                ## Following only makes sense for a person - every other
                ## object just gets labeled and otherwise ignored.
                if label != PERSON_LABEL:
                    continue

                direction = get_direction(box, frame_width)
                distance_status = get_distance_status(box)
                action = get_drone_action(direction, distance_status)

                draw_person_status(frame, direction, distance_status, action)
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


def draw_box(frame, box, label):
    x1, y1, x2, y2 = box

    ## Draw bounding box + class label around every detected object
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2
    )

    draw_text(frame, label, (int(x1), int(y1) - 10), font_scale=0.6)


def draw_person_status(frame, direction, distance_status, action):
    ## Display direction
    draw_text(frame, f"Direction: {direction}", (30, 50))

    ## Display distance
    draw_text(frame, f"Distance: {distance_status}", (30, 90))

    ## Display drone action
    draw_text(frame, f"Action: {action}", (30, 130), font_scale=0.8)


def draw_text(frame, text, position, font_scale=1.0):
    cv2.putText(
        frame,
        text,
        position,
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
