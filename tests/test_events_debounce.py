import time

import pytest

from core.events import EventManager, EventType


class DummyZone:
    def __init__(self, name: str):
        self.name = name

    def contains_point(self, point, frame_shape):
        # Contains if x < 0.5 (simple deterministic rule)
        return point[0] < 0.5


class DummyDet:
    def __init__(self, track_id: int, centroid=None, action=None):
        self.track_id = track_id
        self.centroid = centroid
        self.action = action


def _events_of_type(events, t):
    return [e for e in events if e.type == t]

def test_zone_enter_is_debounced():
    mgr = EventManager(
        stream_id="s",
        zone_enter_confirm_frames=2,
        zone_exit_confirm_frames=2,
        zone_missing_grace_seconds=0.0,
        track_lost_timeout=10.0,
    )
    zone = DummyZone("z")
    frame_shape = (100, 100)

    # Frame 1: inside -> should NOT enter yet
    ev1 = mgr.update([DummyDet(1, centroid=(0.1, 0.1))], [zone], frame_shape, timestamp=1.0)
    assert _events_of_type(ev1, EventType.ZONE_ENTER) == []

    # Frame 2: inside again -> enter should fire
    ev2 = mgr.update([DummyDet(1, centroid=(0.1, 0.1))], [zone], frame_shape, timestamp=1.1)
    assert len(_events_of_type(ev2, EventType.ZONE_ENTER)) == 1


def test_zone_exit_ignored_during_short_detection_dropout_then_exits():
    mgr = EventManager(
        stream_id="s",
        zone_enter_confirm_frames=1,
        zone_exit_confirm_frames=2,
        zone_missing_grace_seconds=0.35,
        track_lost_timeout=10.0,
    )
    zone = DummyZone("z")
    frame_shape = (100, 100)

    # Enter immediately
    ev1 = mgr.update([DummyDet(1, centroid=(0.1, 0.1))], [zone], frame_shape, timestamp=1.0)
    assert len(_events_of_type(ev1, EventType.ZONE_ENTER)) == 1

    # Dropout frame: no detections; within grace => no exit
    ev2 = mgr.update([], [zone], frame_shape, timestamp=1.1)
    assert _events_of_type(ev2, EventType.ZONE_EXIT) == []

    # Back inside: still no exit
    ev3 = mgr.update([DummyDet(1, centroid=(0.1, 0.1))], [zone], frame_shape, timestamp=1.2)
    assert _events_of_type(ev3, EventType.ZONE_EXIT) == []

    # Move outside for 1 frame: shouldn't exit yet
    ev4 = mgr.update([DummyDet(1, centroid=(0.9, 0.1))], [zone], frame_shape, timestamp=1.3)
    assert _events_of_type(ev4, EventType.ZONE_EXIT) == []

    # Outside second frame: exit should fire
    ev5 = mgr.update([DummyDet(1, centroid=(0.9, 0.1))], [zone], frame_shape, timestamp=1.4)
    assert len(_events_of_type(ev5, EventType.ZONE_EXIT)) == 1
