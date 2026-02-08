"""
Pytest configuration and shared fixtures for MotionFlow tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Pose/Keypoint Fixtures
# =============================================================================


@pytest.fixture
def standing_pose() -> np.ndarray:
    """Create a typical standing pose with straight legs."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Head
    kpts[0] = [100, 20, 0.95]   # nose
    kpts[1] = [95, 15, 0.9]    # left eye
    kpts[2] = [105, 15, 0.9]   # right eye
    kpts[3] = [85, 20, 0.8]    # left ear
    kpts[4] = [115, 20, 0.8]   # right ear
    # Shoulders
    kpts[5] = [80, 60, 0.95]   # left shoulder
    kpts[6] = [120, 60, 0.95]  # right shoulder
    # Elbows (arms down)
    kpts[7] = [70, 110, 0.9]   # left elbow
    kpts[8] = [130, 110, 0.9]  # right elbow
    # Wrists (arms down)
    kpts[9] = [70, 160, 0.9]   # left wrist
    kpts[10] = [130, 160, 0.9]  # right wrist
    # Hips
    kpts[11] = [85, 150, 0.95]  # left hip
    kpts[12] = [115, 150, 0.95]  # right hip
    # Knees - straight for standing
    kpts[13] = [85, 220, 0.9]   # left knee
    kpts[14] = [115, 220, 0.9]  # right knee
    # Ankles
    kpts[15] = [85, 290, 0.9]   # left ankle
    kpts[16] = [115, 290, 0.9]  # right ankle
    return kpts


@pytest.fixture
def sitting_pose(standing_pose: np.ndarray) -> np.ndarray:
    """Create a sitting pose with bent knees."""
    kpts = standing_pose.copy()
    # Move knees forward (bent) to create ~90 degree angle
    kpts[13] = [55, 170, 0.9]   # left knee forward and up
    kpts[14] = [145, 170, 0.9]  # right knee forward and up
    # Ankles below knees, closer to body
    kpts[15] = [55, 240, 0.9]   # left ankle
    kpts[16] = [145, 240, 0.9]  # right ankle
    return kpts


@pytest.fixture
def lying_pose() -> np.ndarray:
    """Create a lying down pose (horizontal body orientation)."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    y_base = 100
    # Spread horizontally - body is lying from left to right
    kpts[0] = [50, y_base, 0.95]     # nose (head on left)
    kpts[1] = [45, y_base - 5, 0.9]    # left eye
    kpts[2] = [45, y_base + 5, 0.9]    # right eye
    kpts[5] = [100, y_base - 15, 0.95]  # left shoulder
    kpts[6] = [100, y_base + 15, 0.95]  # right shoulder
    kpts[7] = [120, y_base - 20, 0.9]  # left elbow
    kpts[8] = [120, y_base + 20, 0.9]  # right elbow
    kpts[9] = [140, y_base - 20, 0.9]  # left wrist
    kpts[10] = [140, y_base + 20, 0.9]  # right wrist
    kpts[11] = [180, y_base - 15, 0.95]  # left hip
    kpts[12] = [180, y_base + 15, 0.95]  # right hip
    kpts[13] = [240, y_base - 15, 0.9]  # left knee
    kpts[14] = [240, y_base + 15, 0.9]  # right knee
    kpts[15] = [300, y_base - 15, 0.9]  # left ankle
    kpts[16] = [300, y_base + 15, 0.9]  # right ankle
    return kpts


@pytest.fixture
def phone_calling_pose(standing_pose: np.ndarray) -> np.ndarray:
    """Create pose with one hand raised to ear (phone call)."""
    kpts = standing_pose.copy()
    # Raise left arm to head level
    kpts[7] = [75, 40, 0.9]    # left elbow up
    kpts[9] = [95, 18, 0.9]    # left wrist near head/ear
    return kpts


@pytest.fixture
def phone_texting_pose(standing_pose: np.ndarray) -> np.ndarray:
    """Create pose with both hands together at stomach level (texting)."""
    kpts = standing_pose.copy()
    # Both hands together in front at stomach/chest level
    kpts[7] = [85, 100, 0.9]   # left elbow bent
    kpts[8] = [115, 100, 0.9]  # right elbow bent
    kpts[9] = [95, 120, 0.9]   # left wrist close to center
    kpts[10] = [105, 120, 0.9]  # right wrist close to center
    return kpts


@pytest.fixture
def arms_raised_pose(standing_pose: np.ndarray) -> np.ndarray:
    """Create pose with both arms raised above head."""
    kpts = standing_pose.copy()
    # Raise both arms high
    kpts[7] = [60, 30, 0.9]    # left elbow up
    kpts[8] = [140, 30, 0.9]   # right elbow up
    kpts[9] = [50, 5, 0.9]     # left wrist high above head
    kpts[10] = [150, 5, 0.9]   # right wrist high above head
    return kpts


# =============================================================================
# Zone Fixtures
# =============================================================================


class MockZone:
    """Mock zone for testing without importing core.models."""

    def __init__(self, name: str, contains: bool = True):
        self.name = name
        self._contains = contains
        self.points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.triggers = ["person"]

    def contains_point(self, point, frame_shape) -> bool:
        return self._contains


class MockZoneLeftHalf:
    """Zone that contains points with x < 0.5."""

    def __init__(self, name: str):
        self.name = name
        self.points = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
        self.triggers = ["person"]

    def contains_point(self, point, frame_shape) -> bool:
        return point[0] < 0.5


@pytest.fixture
def mock_zone_full():
    """Zone that contains everything."""
    return MockZone("full_zone", contains=True)


@pytest.fixture
def mock_zone_empty():
    """Zone that contains nothing."""
    return MockZone("empty_zone", contains=False)


@pytest.fixture
def mock_zone_left_half():
    """Zone that contains the left half of the frame."""
    return MockZoneLeftHalf("left_zone")


# =============================================================================
# Detection Fixtures
# =============================================================================


class MockDetection:
    """Mock detection for testing."""

    def __init__(
        self,
        track_id: int,
        centroid: tuple = (0.5, 0.5),
        action: str = "unknown",
        keypoints: np.ndarray | None = None,
    ):
        self.track_id = track_id
        self.centroid = centroid
        self.action = action
        self.keypoints = keypoints if keypoints is not None else np.zeros((17, 3))
        self.velocity = None
        self.speed = 0.0


@pytest.fixture
def mock_detection():
    """Create a mock detection."""
    return MockDetection(track_id=1, centroid=(0.25, 0.5), action="standing")


# =============================================================================
# Video/Frame Fixtures
# =============================================================================


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample video frame (640x480 RGB)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_shape() -> tuple:
    """Standard frame shape for tests."""
    return (480, 640)


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Return the project root path."""
    return PROJECT_ROOT


@pytest.fixture
def test_footage_dir(project_root: Path) -> Path:
    """Return the test footage directory."""
    return project_root / "test_footage"


@pytest.fixture
def test_config_dir(project_root: Path) -> Path:
    """Return the config directory."""
    return project_root / "config"
