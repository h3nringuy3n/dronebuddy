## Turns the tracked direction + distance into a single drone command
def get_drone_action(direction, distance_status):

    ## Person is too far away - move closer, correcting left/right on the way
    if distance_status == "TOO FAR":
        if direction == "LEFT":
            return "MOVE LEFT + FORWARD"
        elif direction == "RIGHT":
            return "MOVE RIGHT + FORWARD"
        else:
            return "MOVE FORWARD"

    ## Person is too close - back away, correcting left/right on the way
    elif distance_status == "TOO CLOSE":
        if direction == "LEFT":
            return "MOVE LEFT + BACKWARD"
        elif direction == "RIGHT":
            return "MOVE RIGHT + BACKWARD"
        else:
            return "MOVE BACKWARD"

    ## Distance is good - only correct left/right, or hover if already centered
    else:
        if direction == "LEFT":
            return "MOVE LEFT"
        elif direction == "RIGHT":
            return "MOVE RIGHT"
        else:
            return "HOVER"
