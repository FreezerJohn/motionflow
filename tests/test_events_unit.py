"""
Unit tests for core.events module.

Tests EventManager, event generation, and listeners.
"""

import pytest

from core.events import (
    EventListener,
    EventManager,
    EventType,
    LoggingListener,
    MotionEvent,
    ZoneState,
)


class MockZone:
    """Mock zone for testing."""

    def __init__(self, name: str, x_threshold: float = 0.5):
        self.name = name
        self.x_threshold = x_threshold
        self.points = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
        self.triggers = ["person"]

    def contains_point(self, point, frame_shape) -> bool:
        """Contains if x < x_threshold."""
        return point[0] < self.x_threshold * frame_shape[1]


class MockDetection:
    """Mock detection for testing."""

    def __init__(self, track_id: int, centroid: tuple, action: str = "unknown"):
        self.track_id = track_id
        self.centroid = centroid
        self.action = action
        self.velocity = None
        self.speed = 0.0


class RecordingListener(EventListener):
    """Listener that records all events for testing."""

    def __init__(self):
        self.events = []

    def on_event(self, event: MotionEvent) -> None:
        self.events.append(event)

    def clear(self):
        self.events = []

    def get_events_of_type(self, event_type: EventType):
        return [e for e in self.events if e.type == event_type]


class TestMotionEvent:
    """Tests for the MotionEvent class."""

    def test_event_creation(self):
        """MotionEvent can be created with required fields."""
        event = MotionEvent(
            type=EventType.ZONE_ENTER,
            timestamp=1234567890.0,
            stream_id="living_room",
            detection_id=1,
            zone_id="couch",
        )

        assert event.type == EventType.ZONE_ENTER
        assert event.stream_id == "living_room"
        assert event.detection_id == 1
        assert event.zone_id == "couch"

    def test_event_to_dict(self):
        """MotionEvent.to_dict() returns serializable dict."""
        event = MotionEvent(
            type=EventType.ZONE_ENTER,
            timestamp=1234567890.0,
            stream_id="living_room",
            detection_id=1,
            zone_id="couch",
            data={"occupancy": 2, "person": {"action": "sitting"}},
        )

        d = event.to_dict()

        assert d["event"] == "zone_enter"
        assert d["stream"] == "living_room"
        assert d["zone"] == "couch"
        assert d["person"]["id"] == 1
        assert d["occupancy"] == 2

    def test_event_to_dict_without_zone(self):
        """MotionEvent without zone_id works in to_dict()."""
        event = MotionEvent(
            type=EventType.TRACK_NEW,
            timestamp=123.0,
            stream_id="s1",
            detection_id=1,
        )

        d = event.to_dict()

        assert "zone" not in d


class TestZoneState:
    """Tests for the ZoneState class."""

    def test_zone_state_empty(self):
        """Empty ZoneState has occupancy 0."""
        state = ZoneState()
        assert state.occupancy == 0

    def test_zone_state_with_detections(self):
        """ZoneState counts detections correctly."""
        state = ZoneState()
        state.detection_ids.add(1)
        state.detection_ids.add(2)

        assert state.occupancy == 2


class TestEventManager:
    """Tests for the EventManager class."""

    @pytest.fixture
    def manager(self):
        return EventManager(
            stream_id="test_stream",
            track_lost_timeout=2.0,
            zone_enter_confirm_frames=1,
            zone_exit_confirm_frames=1,
            zone_missing_grace_seconds=0.0,
            track_new_confirm_frames=1,  # Immediate for most tests
        )

    @pytest.fixture
    def zone(self):
        return MockZone("test_zone", x_threshold=0.5)

    @pytest.fixture
    def frame_shape(self):
        return (480, 640)

    def test_manager_creation(self):
        """EventManager can be created with default settings."""
        mgr = EventManager(stream_id="test")
        assert mgr.stream_id == "test"

    def test_track_new_event(self, manager, zone, frame_shape):
        """New track generates TRACK_NEW event."""
        det = MockDetection(track_id=1, centroid=(100, 240))

        events = manager.update([det], [zone], frame_shape, timestamp=1.0)

        track_new = [e for e in events if e.type == EventType.TRACK_NEW]
        assert len(track_new) == 1
        assert track_new[0].detection_id == 1

    def test_zone_enter_event(self, manager, zone, frame_shape):
        """Entering a zone generates ZONE_ENTER event."""
        # Detection inside zone (x=100 < 320 = 0.5*640)
        det = MockDetection(track_id=1, centroid=(100, 240))

        events = manager.update([det], [zone], frame_shape, timestamp=1.0)

        enter_events = [e for e in events if e.type == EventType.ZONE_ENTER]
        assert len(enter_events) == 1
        assert enter_events[0].zone_id == "test_zone"

    def test_zone_exit_event(self, manager, zone, frame_shape):
        """Exiting a zone generates ZONE_EXIT event."""
        # First: enter zone
        det_inside = MockDetection(track_id=1, centroid=(100, 240))
        manager.update([det_inside], [zone], frame_shape, timestamp=1.0)

        # Then: exit zone (x=400 > 320)
        det_outside = MockDetection(track_id=1, centroid=(400, 240))
        events = manager.update([det_outside], [zone], frame_shape, timestamp=1.1)

        exit_events = [e for e in events if e.type == EventType.ZONE_EXIT]
        assert len(exit_events) == 1
        assert exit_events[0].zone_id == "test_zone"

    def test_no_duplicate_enter_events(self, manager, zone, frame_shape):
        """Staying in zone doesn't generate duplicate ZONE_ENTER."""
        det = MockDetection(track_id=1, centroid=(100, 240))

        # Frame 1: enter
        events1 = manager.update([det], [zone], frame_shape, timestamp=1.0)
        enter1 = [e for e in events1 if e.type == EventType.ZONE_ENTER]

        # Frame 2: still inside
        events2 = manager.update([det], [zone], frame_shape, timestamp=1.1)
        enter2 = [e for e in events2 if e.type == EventType.ZONE_ENTER]

        assert len(enter1) == 1
        assert len(enter2) == 0  # No duplicate

    def test_track_lost_event(self, frame_shape):
        """Lost track generates TRACK_LOST event after timeout."""
        manager = EventManager(
            stream_id="test",
            track_lost_timeout=0.5,
            zone_enter_confirm_frames=1,
            zone_exit_confirm_frames=1,
            track_new_confirm_frames=1,
        )
        zone = MockZone("z", x_threshold=0.5)

        # Frame 1: detection appears
        det = MockDetection(track_id=1, centroid=(100, 240))
        manager.update([det], [zone], frame_shape, timestamp=1.0)

        # Frame 2: detection disappears, but within timeout
        events2 = manager.update([], [zone], frame_shape, timestamp=1.2)
        lost2 = [e for e in events2 if e.type == EventType.TRACK_LOST]
        assert len(lost2) == 0

        # Frame 3: after timeout
        events3 = manager.update([], [zone], frame_shape, timestamp=2.0)
        lost3 = [e for e in events3 if e.type == EventType.TRACK_LOST]
        assert len(lost3) == 1
        assert lost3[0].detection_id == 1

    def test_listener_receives_events(self, manager, zone, frame_shape):
        """Registered listener receives dispatched events."""
        listener = RecordingListener()
        manager.add_listener(listener)

        det = MockDetection(track_id=1, centroid=(100, 240))
        manager.update([det], [zone], frame_shape, timestamp=1.0)

        # Listener should have received TRACK_NEW and ZONE_ENTER
        assert len(listener.events) >= 2
        types = {e.type for e in listener.events}
        assert EventType.TRACK_NEW in types
        assert EventType.ZONE_ENTER in types

    def test_remove_listener(self, manager, zone, frame_shape):
        """Removed listener stops receiving events."""
        listener = RecordingListener()
        manager.add_listener(listener)
        manager.remove_listener(listener)

        det = MockDetection(track_id=1, centroid=(100, 240))
        manager.update([det], [zone], frame_shape, timestamp=1.0)

        assert len(listener.events) == 0

    def test_action_change_event(self, manager, zone, frame_shape):
        """Action change generates ACTION_CHANGE event."""
        # Frame 1: standing
        det1 = MockDetection(track_id=1, centroid=(100, 240), action="standing")
        manager.update([det1], [zone], frame_shape, timestamp=1.0)

        # Frame 2: sitting
        det2 = MockDetection(track_id=1, centroid=(100, 240), action="sitting")
        events2 = manager.update([det2], [zone], frame_shape, timestamp=1.1)

        action_events = [e for e in events2 if e.type == EventType.ACTION_CHANGE]
        assert len(action_events) == 1
        assert action_events[0].data.get("person", {}).get("action") == "sitting"


class TestEventDebouncing:
    """Tests for zone enter/exit debouncing."""

    def test_zone_enter_requires_confirmation(self):
        """ZONE_ENTER requires multiple frames of confirmation."""
        manager = EventManager(
            stream_id="test",
            zone_enter_confirm_frames=3,
            zone_exit_confirm_frames=1,
            zone_missing_grace_seconds=0.0,
            track_new_confirm_frames=1,
        )
        zone = MockZone("z")
        frame_shape = (480, 640)

        det = MockDetection(track_id=1, centroid=(100, 240))

        # Frame 1 & 2: inside but not confirmed yet
        events1 = manager.update([det], [zone], frame_shape, timestamp=1.0)
        events2 = manager.update([det], [zone], frame_shape, timestamp=1.1)

        enter1 = [e for e in events1 if e.type == EventType.ZONE_ENTER]
        enter2 = [e for e in events2 if e.type == EventType.ZONE_ENTER]

        assert len(enter1) == 0
        assert len(enter2) == 0

        # Frame 3: confirmed
        events3 = manager.update([det], [zone], frame_shape, timestamp=1.2)
        enter3 = [e for e in events3 if e.type == EventType.ZONE_ENTER]
        assert len(enter3) == 1

    def test_zone_exit_requires_confirmation(self):
        """ZONE_EXIT requires multiple frames of confirmation."""
        manager = EventManager(
            stream_id="test",
            zone_enter_confirm_frames=1,
            zone_exit_confirm_frames=3,
            zone_missing_grace_seconds=0.0,
            track_new_confirm_frames=1,
        )
        zone = MockZone("z")
        frame_shape = (480, 640)

        # First: enter zone
        det_inside = MockDetection(track_id=1, centroid=(100, 240))
        manager.update([det_inside], [zone], frame_shape, timestamp=1.0)

        # Then: outside but not confirmed yet
        det_outside = MockDetection(track_id=1, centroid=(400, 240))

        events1 = manager.update([det_outside], [zone], frame_shape, timestamp=1.1)
        events2 = manager.update([det_outside], [zone], frame_shape, timestamp=1.2)

        exit1 = [e for e in events1 if e.type == EventType.ZONE_EXIT]
        exit2 = [e for e in events2 if e.type == EventType.ZONE_EXIT]

        assert len(exit1) == 0
        assert len(exit2) == 0

        # Frame 3: confirmed exit
        events3 = manager.update([det_outside], [zone], frame_shape, timestamp=1.3)
        exit3 = [e for e in events3 if e.type == EventType.ZONE_EXIT]
        assert len(exit3) == 1

    def test_missing_grace_prevents_exit(self):
        """Missing detection within grace period doesn't trigger exit."""
        manager = EventManager(
            stream_id="test",
            zone_enter_confirm_frames=1,
            zone_exit_confirm_frames=1,
            zone_missing_grace_seconds=0.5,
            track_new_confirm_frames=1,
        )
        zone = MockZone("z")
        frame_shape = (480, 640)

        # Enter zone
        det = MockDetection(track_id=1, centroid=(100, 240))
        manager.update([det], [zone], frame_shape, timestamp=1.0)

        # Detection disappears briefly (within grace)
        events2 = manager.update([], [zone], frame_shape, timestamp=1.2)
        exit2 = [e for e in events2 if e.type == EventType.ZONE_EXIT]

        assert len(exit2) == 0  # Grace prevents exit

        # Detection reappears
        events3 = manager.update([det], [zone], frame_shape, timestamp=1.3)
        exit3 = [e for e in events3 if e.type == EventType.ZONE_EXIT]

        assert len(exit3) == 0  # Still no exit

    def test_track_new_requires_confirmation(self):
        """TRACK_NEW requires multiple consecutive frames of detection."""
        manager = EventManager(
            stream_id="test",
            zone_enter_confirm_frames=1,
            zone_exit_confirm_frames=1,
            zone_missing_grace_seconds=0.0,
            track_new_confirm_frames=3,  # Require 3 frames
        )
        zone = MockZone("z")
        frame_shape = (480, 640)

        det = MockDetection(track_id=1, centroid=(100, 240))

        # Frame 1 & 2: detected but not confirmed yet
        events1 = manager.update([det], [zone], frame_shape, timestamp=1.0)
        events2 = manager.update([det], [zone], frame_shape, timestamp=1.1)

        track_new1 = [e for e in events1 if e.type == EventType.TRACK_NEW]
        track_new2 = [e for e in events2 if e.type == EventType.TRACK_NEW]

        assert len(track_new1) == 0
        assert len(track_new2) == 0

        # Frame 3: confirmed
        events3 = manager.update([det], [zone], frame_shape, timestamp=1.2)
        track_new3 = [e for e in events3 if e.type == EventType.TRACK_NEW]
        assert len(track_new3) == 1
        assert track_new3[0].detection_id == 1

    def test_track_new_resets_on_disappear(self):
        """Pending track resets if detection disappears before confirmation."""
        manager = EventManager(
            stream_id="test",
            zone_enter_confirm_frames=1,
            zone_exit_confirm_frames=1,
            zone_missing_grace_seconds=0.0,
            track_new_confirm_frames=3,
        )
        zone = MockZone("z")
        frame_shape = (480, 640)

        det = MockDetection(track_id=1, centroid=(100, 240))

        # Frame 1 & 2: detected
        manager.update([det], [zone], frame_shape, timestamp=1.0)
        manager.update([det], [zone], frame_shape, timestamp=1.1)

        # Frame 3: detection disappears (false positive filtered out)
        events3 = manager.update([], [zone], frame_shape, timestamp=1.2)
        track_new3 = [e for e in events3 if e.type == EventType.TRACK_NEW]
        assert len(track_new3) == 0

        # Frame 4-6: detection reappears, needs 3 more frames
        events4 = manager.update([det], [zone], frame_shape, timestamp=1.3)
        events5 = manager.update([det], [zone], frame_shape, timestamp=1.4)
        events6 = manager.update([det], [zone], frame_shape, timestamp=1.5)

        track_new4 = [e for e in events4 if e.type == EventType.TRACK_NEW]
        track_new5 = [e for e in events5 if e.type == EventType.TRACK_NEW]
        track_new6 = [e for e in events6 if e.type == EventType.TRACK_NEW]

        assert len(track_new4) == 0
        assert len(track_new5) == 0
        assert len(track_new6) == 1  # Now confirmed


class TestLoggingListener:
    """Tests for the LoggingListener."""

    def test_logging_listener_creates(self):
        """LoggingListener can be instantiated."""
        listener = LoggingListener()
        assert listener is not None

    def test_logging_listener_handles_event(self, caplog):
        """LoggingListener logs events without errors."""
        import logging

        listener = LoggingListener(level=logging.INFO)
        event = MotionEvent(
            type=EventType.ZONE_ENTER,
            timestamp=123.0,
            stream_id="test",
            detection_id=1,
            zone_id="zone1",
        )

        # Should not raise
        listener.on_event(event)
