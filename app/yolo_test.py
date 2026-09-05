from ultralytics import YOLO
import cv2

def get_drone_action(direction, distance_status):
    if distance_status == "TOO FAR":

        if direction == "LEFT":
            return "MOVE LEFT + FORWARD"

        elif direction =="RIGHT":
            return "MOVE RIGHT + FORWARD"

        else:
            return "MOVE FORWARD"

    elif distance_status == "TOO CLOSE":
        if direction == "LEFT":
            return "MOVE LEFT + BACKWARD"

        elif direction =="RIGHT":
            return "MOVE RIGHT + BACKWARD"

        else:
            return "MOVE BACKWARD"

    else:
        if direction == "LEFT":
            return "MOVE LEFT"

        elif direction =="RIGHT":
            return "MOVE RIGHT"

        else:
            return "HOVER"

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

## Loop to continuously read frames from the camera and perform YOLO detection
while True:

    success, frame = camera.read()  # Read a frame from the camera

    if not success:
        print("Could not read from camera")
        break

    height, width = frame.shape[:2]  # Get frame dimensions

    camera_center_x = width / 2
    camera_center_y = height / 2

    results = model(frame)  # Perform YOLO detection

    boxes = results[0].boxes  # Extract detected boxes

    ## Loop through the detected boxes
    for box in boxes:

        class_id = int(box.cls[0])

        ## Class ID 0 = person
        if class_id == 0:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            ## Calculate center of the person's bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            ## Determine horizontal position
            dead_zone = 100

            if center_x < camera_center_x - dead_zone:
                direction = "LEFT"

            elif center_x > camera_center_x + dead_zone:
                direction = "RIGHT"

            else:
                direction = "CENTER"

            ## Calculate bounding box size
            box_width = x2 - x1
            box_height = y2 - y1

            ## Estimate relative distance
            if box_height < 300:
                distance_status = "TOO FAR"

            elif box_height > 600:
                distance_status = "TOO CLOSE"

            else:
                distance_status = "GOOD DISTANCE"

            action = get_drone_action(direction, distance_status)

            ## Draw bounding box around person
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            ## Display direction
            cv2.putText(
                frame,
                f"Direction: {direction}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            ## Display distance
            cv2.putText(
                frame,
                f"Distance: {distance_status}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            ## Display drone action
            cv2.putText(
                frame,
                f"Action: {action}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )



            ## Print information to Terminal
            print("PERSON FOUND!")
            print("Box:", x1, y1, x2, y2)
            print("Center:", center_x, center_y)
            print("Direction:", direction)
            print("Distance Status:", distance_status)
            print("Drone Action:", action)
    ## Display the frame
    cv2.imshow("YOLO Detection", frame)

    ## Exit when 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

## Release camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
