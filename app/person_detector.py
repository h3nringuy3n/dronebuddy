from ultralytics import YOLO

PERSON_LABEL = "person"


def load_model(weights_path):
    return YOLO(weights_path)


def detect_objects(model, frame):
    """Runs YOLO on a frame and returns a (label, box) pair for every detected object,
    where box is (x1, y1, x2, y2). Not filtered to any one class - callers decide
    which labels matter to them (e.g. only "person" for following)."""
    results = model(frame, verbose=False)  # Perform YOLO detection

    boxes = results[0].boxes  # Extract detected boxes

    objects = []

    ## Loop through the detected boxes
    for box in boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        objects.append((label, (x1, y1, x2, y2)))

    return objects
