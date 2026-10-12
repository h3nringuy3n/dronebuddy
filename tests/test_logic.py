"""Unit tests for the decision logic. No camera, model or drone needed.

Run from the project root:  python -m unittest discover tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

from compute_actions import ComputeActions
from control.follow_controller import get_drone_action
from detection_module import Detection
from object_locator import ObjectLocator
from person_tracker import describe_position, select_target
from safety_manager import SafetyManager
from tracking_module import APPROACHING, LEFT_FRAME, NEW, STANDING_STILL, TrackingModule

FRAME_W, FRAME_H = 1280, 720
FPS = 10


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def person(track_id, center_x, height):
    """A person Detection centered at center_x, standing on the bottom of the frame."""
    width = height * 0.4
    return Detection("person", (center_x - width / 2, FRAME_H - height, center_x + width / 2, FRAME_H), track_id)


def run_frames(tracking, clock, frames):
    """Feeds a list of per-frame people lists through the tracker, returns all events."""
    events = []
    for people in frames:
        events += tracking.update(people, FRAME_W, FRAME_H)
        clock.now += 1 / FPS
    return events


class TrackingModuleTests(unittest.TestCase):
    def test_new_person_then_standing_still(self):
        clock = FakeClock()
        tracking = TrackingModule(clock)

        events = run_frames(tracking, clock, [[person(1, 640, 300)]] * 20)

        activities = [e.activity for e in events]
        self.assertEqual(activities[0], NEW)
        self.assertIn(STANDING_STILL, activities)

    def test_growing_box_is_approaching(self):
        clock = FakeClock()
        tracking = TrackingModule(clock)

        ## Box grows from 200px to 500px over 2 seconds
        frames = [[person(1, 640, 200 + i * 15)] for i in range(20)]
        events = run_frames(tracking, clock, frames)

        self.assertIn(APPROACHING, [e.activity for e in events])

    def test_person_disappearing_left_frame(self):
        clock = FakeClock()
        tracking = TrackingModule(clock)

        frames = [[person(1, 640, 300)]] * 5 + [[]] * 15
        events = run_frames(tracking, clock, frames)

        self.assertEqual(events[-1].activity, LEFT_FRAME)
        self.assertIsNone(tracking.activity_of(1))


class ComputeActionsTests(unittest.TestCase):
    def test_alert_text_and_cooldown(self):
        clock = FakeClock()
        tracking = TrackingModule(clock)
        actions = ComputeActions(cooldown_s=5, clock=clock)

        events = tracking.update([person(1, 100, 300)], FRAME_W, FRAME_H)
        self.assertEqual(actions.compute(events), ["Person on your left"])

        ## Same event again within the cooldown is not repeated
        self.assertEqual(actions.compute(events), [])

        clock.now = 6
        self.assertEqual(actions.compute(events), ["Person on your left"])

    def test_two_people_same_message_spoken_once(self):
        clock = FakeClock()
        tracking = TrackingModule(clock)
        actions = ComputeActions(clock=clock)

        events = tracking.update([person(1, 640, 300), person(2, 660, 300)], FRAME_W, FRAME_H)
        self.assertEqual(actions.compute(events), ["Person ahead"])


class ObjectLocatorTests(unittest.TestCase):
    def test_only_close_objects_announced_once(self):
        clock = FakeClock()
        locator = ObjectLocator(cooldown_s=10, clock=clock)

        near_chair = Detection("chair", (1000, 300, 1250, 700))  # 400px tall = 0.56 of frame
        far_cup = Detection("cup", (600, 300, 620, 330))
        someone = Detection("person", (0, 0, 640, 720), 1)

        result = locator.locate([near_chair, far_cup, someone], FRAME_W, FRAME_H)
        self.assertEqual(result, [("chair", "on your right", "chair on your right")])

        self.assertEqual(locator.locate([near_chair], FRAME_W, FRAME_H), [])


class TargetSelectionTests(unittest.TestCase):
    def test_stays_locked_on_same_person(self):
        small = person(7, 300, 200)
        big = person(8, 900, 500)

        self.assertEqual(select_target([small, big]).track_id, 8)
        self.assertEqual(select_target([small, big], locked_id=7).track_id, 7)
        self.assertIsNone(select_target([]))

    def test_describe_position(self):
        self.assertEqual(describe_position(-0.8), "on your left")
        self.assertEqual(describe_position(0.0), "ahead")
        self.assertEqual(describe_position(0.8), "on your right")


class FollowAndSafetyTests(unittest.TestCase):
    def test_follow_command_is_clamped_and_hovers_when_lost(self):
        clock = FakeClock()
        safety = SafetyManager(clock=clock)

        command = get_drone_action(horizontal_error=1.0, height_ratio=0.1)
        safe = safety.filter(command, target_seen=True)
        self.assertLessEqual(abs(safe.yaw), safety.max_yaw)
        self.assertGreater(safe.forward, 0)

        self.assertTrue(safety.filter(command, target_seen=False).is_hover())

        safety.trigger_emergency_stop()
        self.assertTrue(safety.filter(command, target_seen=True).is_hover())


if __name__ == "__main__":
    unittest.main()
