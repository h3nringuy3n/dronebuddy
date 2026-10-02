## All thresholds are fractions of the frame, so they work at any resolution
## (webcam 640x480, DJI Mini 3 stream 1280x720 / 1920x1080, recorded clips...)
DEAD_ZONE = 0.10          # +/- fraction of half-frame width counted as "CENTER"
TARGET_HEIGHT_RATIO = 0.55  # Box height / frame height we want to hold the person at
DISTANCE_TOLERANCE = 0.10   # +/- around the target ratio counted as "GOOD DISTANCE"


def select_target(person_boxes):
    """Picks the one person to follow: the largest box (usually the closest person).
    Returns None if there are no people in the frame."""
    if not person_boxes:
        return None

    return max(person_boxes, key=box_area)


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def get_horizontal_error(box, frame_width):
    """How far the person is from the frame center: -1.0 = far left edge,
    0.0 = centered, +1.0 = far right edge."""
    x1, y1, x2, y2 = box

    ## Calculate center of the person's bounding box
    center_x = (x1 + x2) / 2

    frame_center_x = frame_width / 2

    return (center_x - frame_center_x) / frame_center_x


def get_height_ratio(box, frame_height):
    """Box height as a fraction of the frame height - our stand-in for distance.
    Bigger ratio = person is closer."""
    x1, y1, x2, y2 = box

    return (y2 - y1) / frame_height


def get_direction(horizontal_error, dead_zone=DEAD_ZONE):
    ## Determine horizontal position
    if horizontal_error < -dead_zone:
        return "LEFT"
    elif horizontal_error > dead_zone:
        return "RIGHT"
    else:
        return "CENTER"


def get_distance_status(height_ratio, target=TARGET_HEIGHT_RATIO, tolerance=DISTANCE_TOLERANCE):
    ## Estimate relative distance
    if height_ratio < target - tolerance:
        return "TOO FAR"
    elif height_ratio > target + tolerance:
        return "TOO CLOSE"
    else:
        return "GOOD DISTANCE"
