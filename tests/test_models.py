"""
Unit tests for core.models module.

Tests Zone, Door, Detection, and DetectionManager classes.
"""

import numpy as np
import pytest

from core.models import Detection, DetectionManager, Door, Zone, calculate_centroid


class TestCalculateCentroid:
    """Tests for the calculate_centroid function."""

    def test_centroid_from_hips(self):
        """Centroid should prefer mid-hip when both hips visible."""
        kpts = np.zeros((17, 3), dtype=np.float32)
        # Set hips with high confidence
        kpts[11] = [100, 200, 0.9]  # left hip
        kpts[12] = [120, 200, 0.9]  # right hip
        # Set shoulders too
        kpts[5] = [100, 100, 0.9]
        kpts[6] = [120, 100, 0.9]

        centroid = calculate_centroid(kpts)

        # Should be midpoint of hips
        assert centroid[0] == pytest.approx(110.0)
        assert centroid[1] == pytest.approx(200.0)

    def test_centroid_from_shoulders_when_hips_low_confidence(self):
        """Falls back to shoulders when hips have low confidence."""
        kpts = np.zeros((17, 3), dtype=np.float32)
        # Set hips with LOW confidence
        kpts[11] = [100, 200, 0.1]  # left hip
        kpts[12] = [120, 200, 0.1]  # right hip
        # Set shoulders with high confidence
        kpts[5] = [100, 100, 0.9]
        kpts[6] = [120, 100, 0.9]

        centroid = calculate_centroid(kpts)

        # Should be midpoint of shoulders
        assert centroid[0] == pytest.approx(110.0)
        assert centroid[1] == pytest.approx(100.0)

    def test_centroid_fallback_to_visible_keypoints(self):
        """Falls back to average of visible keypoints."""
        kpts = np.zeros((17, 3), dtype=np.float32)
        # Only nose visible
        kpts[0] = [100, 50, 0.9]  # nose

        centroid = calculate_centroid(kpts)

        # Should be the nose position
        assert centroid[0] == pytest.approx(100.0)
        assert centroid[1] == pytest.approx(50.0)


class TestZone:
    """Tests for the Zone class."""

    def test_zone_creation(self):
        """Zone can be created with name and points."""
        zone = Zone(
            name="test_zone",
            points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )

        assert zone.name == "test_zone"
        assert len(zone.points) == 4
        assert zone.triggers == ["person"]

    def test_get_pixel_points(self):
        """Normalized points are correctly converted to pixel coordinates."""
        zone = Zone(
            name="test",
            points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )

        frame_shape = (480, 640)  # height, width
        pixel_pts = zone.get_pixel_points(frame_shape)

        assert pixel_pts.shape == (4, 2)
        assert pixel_pts[0].tolist() == [0, 0]
        assert pixel_pts[1].tolist() == [640, 0]
        assert pixel_pts[2].tolist() == [640, 480]
        assert pixel_pts[3].tolist() == [0, 480]

    def test_get_pixel_points_caches_result(self):
        """Pixel points are cached for the same frame shape."""
        zone = Zone(
            name="test",
            points=[(0.5, 0.5), (0.6, 0.5), (0.6, 0.6), (0.5, 0.6)],
        )

        frame_shape = (480, 640)
        pts1 = zone.get_pixel_points(frame_shape)
        pts2 = zone.get_pixel_points(frame_shape)

        assert pts1 is pts2  # Same object (cached)

    def test_contains_point_inside(self):
        """Point inside zone returns True."""
        zone = Zone(
            name="test",
            points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
        )

        frame_shape = (100, 100)
        # Center point should be inside
        assert zone.contains_point((50, 50), frame_shape) is True

    def test_contains_point_outside(self):
        """Point outside zone returns False."""
        zone = Zone(
            name="test",
            points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
        )

        frame_shape = (100, 100)
        # Point outside should be outside
        assert zone.contains_point((10, 10), frame_shape) is False

    def test_contains_point_with_radius(self):
        """Radius extends containment check."""
        zone = Zone(
            name="test",
            points=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)],
        )

        frame_shape = (100, 100)
        # Point just outside (at x=25, zone starts at x=30)
        assert zone.contains_point((25, 50), frame_shape, radius=0) is False
        # With radius=10, should overlap
        assert zone.contains_point((25, 50), frame_shape, radius=10) is True

    def test_empty_zone_contains_nothing(self):
        """Empty zone never contains points."""
        zone = Zone(name="empty", points=[])

        assert zone.contains_point((50, 50), (100, 100)) is False


class TestDoor:
    """Tests for the Door class."""

    def test_door_creation(self):
        """Door can be created with required parameters."""
        door = Door(
            name="entrance",
            points=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
            normal_angle=0.0,
        )

        assert door.name == "entrance"
        assert len(door.points) == 4
        assert door.normal_angle == 0.0
        assert door.direction_tolerance == 60.0

    def test_get_normal_vector_right(self):
        """Normal at 0° points right."""
        door = Door(name="test", points=[], normal_angle=0.0)

        normal = door.get_normal_vector()

        assert normal[0] == pytest.approx(1.0)  # x = 1
        assert normal[1] == pytest.approx(0.0)  # y = 0

    def test_get_normal_vector_down(self):
        """Normal at 90° points down."""
        door = Door(name="test", points=[], normal_angle=90.0)

        normal = door.get_normal_vector()

        assert normal[0] == pytest.approx(0.0)
        assert normal[1] == pytest.approx(1.0)

    def test_get_normal_vector_left(self):
        """Normal at 180° points left."""
        door = Door(name="test", points=[], normal_angle=180.0)

        normal = door.get_normal_vector()

        assert normal[0] == pytest.approx(-1.0)
        assert normal[1] == pytest.approx(0.0, abs=1e-6)

    def test_contains_point(self):
        """Point containment works for doors."""
        door = Door(
            name="test",
            points=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
        )

        frame_shape = (100, 100)
        assert door.contains_point((50, 50), frame_shape) is True
        assert door.contains_point((10, 10), frame_shape) is False

    def test_check_crossing_enter(self):
        """Crossing in normal direction returns 'enter'."""
        door = Door(
            name="test",
            points=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
            normal_angle=0.0,  # Enter = moving right
            direction_tolerance=60.0,
        )

        frame_shape = (100, 100)
        # Position inside door, moving right
        result = door.check_crossing(
            current_pos=(50, 50),
            velocity=np.array([100.0, 0.0]),  # Moving right
            frame_shape=frame_shape,
        )

        assert result == "enter"

    def test_check_crossing_exit(self):
        """Crossing opposite to normal direction returns 'exit'."""
        door = Door(
            name="test",
            points=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
            normal_angle=0.0,  # Enter = right, Exit = left
            direction_tolerance=60.0,
        )

        frame_shape = (100, 100)
        # Position inside door, moving left
        result = door.check_crossing(
            current_pos=(50, 50),
            velocity=np.array([-100.0, 0.0]),  # Moving left
            frame_shape=frame_shape,
        )

        assert result == "exit"

    def test_check_crossing_perpendicular_returns_none(self):
        """Crossing perpendicular to normal returns None."""
        door = Door(
            name="test",
            points=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
            normal_angle=0.0,  # Enter = right
            direction_tolerance=45.0,  # Narrow tolerance
        )

        frame_shape = (100, 100)
        # Moving up (perpendicular to horizontal normal)
        result = door.check_crossing(
            current_pos=(50, 50),
            velocity=np.array([0.0, -100.0]),  # Moving up
            frame_shape=frame_shape,
        )

        assert result is None

    def test_check_crossing_too_slow(self):
        """Slow movement returns None."""
        door = Door(
            name="test",
            points=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
        )

        frame_shape = (100, 100)
        result = door.check_crossing(
            current_pos=(50, 50),
            velocity=np.array([1.0, 0.0]),  # Very slow
            frame_shape=frame_shape,
            min_speed=5.0,
        )

        assert result is None

    def test_check_crossing_outside_door(self):
        """Point outside door returns None."""
        door = Door(
            name="test",
            points=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
        )

        frame_shape = (100, 100)
        result = door.check_crossing(
            current_pos=(10, 10),  # Outside
            velocity=np.array([100.0, 0.0]),
            frame_shape=frame_shape,
        )

        assert result is None


class TestDetection:
    """Tests for the Detection class."""

    def test_detection_creation(self, standing_pose):
        """Detection can be created with track ID and keypoints."""
        det = Detection(
            track_id=1,
            keypoints=standing_pose,
            bbox=np.array([100, 100, 200, 300]),
            confidence=0.95,
        )

        assert det.track_id == 1
        assert det.confidence == 0.95
        assert det.action is None

    def test_detection_centroid_calculated(self, standing_pose):
        """Centroid is calculated on creation."""
        det = Detection(
            track_id=1,
            keypoints=standing_pose,
            bbox=np.array([0, 0, 200, 300]),
            confidence=0.9,
        )

        assert det.centroid is not None
        assert len(det.centroid) == 2


class TestDetectionManager:
    """Tests for the DetectionManager class."""

    def test_manager_creation(self):
        """DetectionManager can be created with max_age."""
        manager = DetectionManager(max_age=2.0)
        assert manager.max_age == 2.0

    def test_update_creates_detection(self, standing_pose):
        """update() creates Detection from raw detection data."""
        manager = DetectionManager(max_age=2.0)

        raw_detections = [
            {
                "id": 1,  # Note: the API uses 'id', not 'track_id'
                "keypoints": standing_pose,
                "bbox": np.array([100, 100, 200, 300]),
                "score": 0.9,  # Note: the API uses 'score', not 'conf'
            }
        ]

        detections = manager.update(raw_detections, timestamp=0.0)

        assert len(detections) == 1
        assert detections[0].track_id == 1

    def test_manager_tracks_detection_lifecycle(self, standing_pose):
        """Detections are tracked across frames and eventually expire."""
        manager = DetectionManager(max_age=0.5)

        # Frame 1: Detection appears
        raw1 = [{"id": 1, "keypoints": standing_pose, "bbox": np.array([0, 0, 100, 200]), "score": 0.9}]
        dets1 = manager.update(raw1, timestamp=0.0)
        assert len(dets1) == 1

        # Frame 2: Detection still present
        dets2 = manager.update(raw1, timestamp=0.1)
        assert len(dets2) == 1

        # Frame 3: No new detections, but existing ones persist within max_age
        dets3 = manager.update([], timestamp=0.2)
        # The update returns all currently tracked detections
        assert len(dets3) == 1  # Detection still in manager

        # Check if track is still known (within max_age)
        assert manager.get_detection(1) is not None

        # Frame 4: After max_age, track should be pruned
        manager.update([], timestamp=1.0)
        assert manager.get_detection(1) is None
