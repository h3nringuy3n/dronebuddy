DEAD_ZONE_PX = 100
FAR_HEIGHT_PX = 300
CLOSE_HEIGHT_PX = 600


def get_direction(box, frame_width, dead_zone_px=DEAD_ZONE_PX):
    x1, y1, x2, y2 = box

    ## Calculate center of the person's bounding box
    center_x = (x1 + x2) / 2

    frame_center_x = frame_width / 2

    ## Determine horizontal position
    if center_x < frame_center_x - dead_zone_px:
        return "LEFT"
    elif center_x > frame_center_x + dead_zone_px:
        return "RIGHT"
    else:
        return "CENTER"


def get_distance_status(box, far_height_px=FAR_HEIGHT_PX, close_height_px=CLOSE_HEIGHT_PX):
    x1, y1, x2, y2 = box

    ## Calculate bounding box size
    height = y2 - y1

    ## Estimate relative distance
    if height < far_height_px:
        return "TOO FAR"
    elif height > close_height_px:
        return "TOO CLOSE"
    else:
        return "GOOD DISTANCE"
