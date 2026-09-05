from ultralytics import YOLO

PERSON_CLASS_ID = 0  ## Class ID 0 = person


def load_model(weights_path):
    return YOLO(weights_path)


def detect_people(model, frame):
    """Runs YOLO on a frame and returns a (x1, y1, x2, y2) box for each detected person."""
    results = model(frame)  # Perform YOLO detection

    boxes = results[0].boxes  # Extract detected boxes

    people = []

    ## Loop through the detected boxes
    for box in boxes:

        class_id = int(box.cls[0])

        if class_id == PERSON_CLASS_ID:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            people.append((x1, y1, x2, y2))

    return people
