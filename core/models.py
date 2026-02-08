"""
Domain models for MotionFlow.

These classes represent the core entities in the system and hold their state.
The Visualizer and other components receive these objects and operate on them
without needing to maintain their own state.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# COCO keypoint names for reference
COCO_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]


def calculate_centroid(kpts: np.ndarray) -> np.ndarray:
    """
    Calculate robust centroid from keypoints.
    Prioritizes Hips and Shoulders for stability to avoid jumping when keypoints are missing.
    """
    # Indices in COCO_KEYPOINTS:
    # Hips: 11 (Left), 12 (Right)
    # Shoulders: 5 (Left), 6 (Right)

    # 1. Try Mid-Hip (Most stable Center of Gravity)
    hips = kpts[[11, 12]]
    if np.all(hips[:, 2] > 0.3):
        return np.mean(hips[:, :2], axis=0)

    # 2. Try Mid-Shoulder
    shoulders = kpts[[5, 6]]
    if np.all(shoulders[:, 2] > 0.3):
        return np.mean(shoulders[:, :2], axis=0)

    # 3. Fallback to average of Head, Torso and Upper Legs
    target_parts = {
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder",
        "Left Hip", "Right Hip",
        "Left Knee", "Right Knee"
    }
    core_indices = [i for i, name in enumerate(COCO_KEYPOINTS) if name in target_parts]
    core_kpts = kpts[core_indices]

    visible = core_kpts[:, 2] > 0.3
    if np.any(visible):
        return np.mean(core_kpts[visible, :2], axis=0)

    # 4. Fallback to all visible keypoints
    visible_all = kpts[:, 2] > 0.3
    if np.any(visible_all):
        return np.mean(kpts[visible_all, :2], axis=0)

    # 5. Fallback to mean of all keypoints
    return np.mean(kpts[:, :2], axis=0)


@dataclass
class Zone:
    """
    Represents a polygon zone of interest within a video stream.

    This is the runtime representation of a zone, with methods for
    point-in-polygon testing. It wraps the config Zone data.
    """
    name: str
    points: list[tuple[float, float]]  # Normalized (0.0-1.0) coordinates
    triggers: list[str] = field(default_factory=lambda: ["person"])
    color: tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)

    # Cached pixel coordinates (set when frame dimensions are known)
    _pixel_points: np.ndarray | None = field(default=None, repr=False)
    _frame_shape: tuple[int, int] | None = field(default=None, repr=False)

    def get_pixel_points(self, frame_shape: tuple[int, int]) -> np.ndarray:
        """
        Convert normalized points to pixel coordinates for the given frame shape.
        Caches the result for performance.
        """
        height, width = frame_shape[:2]

        # Use cached version if frame shape hasn't changed
        if self._pixel_points is not None and self._frame_shape == (height, width):
            return self._pixel_points

        self._pixel_points = np.array(
            [[int(p[0] * width), int(p[1] * height)] for p in self.points],
            dtype=np.int32
        )
        self._frame_shape = (height, width)
        return self._pixel_points

    def contains_point(self, point: tuple[float, float], frame_shape: tuple[int, int], radius: float = 0) -> bool:
        """
        Check if a point (in pixel coordinates) is inside this zone.

        Args:
            point: (x, y) in pixel coordinates
            frame_shape: (height, width) of the frame
            radius: Optional radius around the point to consider overlap

        Returns:
            True if point is inside (or within radius of) the zone
        """
        if not self.points:
            return False

        poly_pts = self.get_pixel_points(frame_shape)
        dist = cv2.pointPolygonTest(poly_pts, point, True)

        # dist >= 0 means inside or on edge
        # dist >= -radius means within radius of the boundary
        return dist >= -radius

    @classmethod
    def from_config(cls, config_zone) -> 'Zone':
        """Create a Zone from a config Zone pydantic model."""
        # Convert Point objects to tuples if needed
        points = []
        for p in config_zone.points:
            if hasattr(p, 'x') and hasattr(p, 'y'):
                # It's a Point object
                points.append((p.x, p.y))
            else:
                # It's already a tuple/list
                points.append((p[0], p[1]))

        return cls(
            name=config_zone.name,
            points=points,
            triggers=config_zone.triggers
        )


@dataclass
class Door:
    """
    Represents a door/threshold for detecting enter/exit direction.

    A door is defined by 4 points forming a polygon (the door area) and a
    user-defined normal angle that indicates the "enter" direction.

    Movement in the direction of the normal is "entering",
    movement opposite (180°) is "exiting".
    """
    name: str
    points: list[tuple[float, float]]  # Normalized (0.0-1.0) coordinates, 4 points
    triggers: list[str] = field(default_factory=lambda: ["person"])
    normal_angle: float = 0.0  # Degrees: 0=right, 90=down, 180=left, 270=up
    direction_tolerance: float = 60.0  # Degrees: tolerance for direction matching
    color: tuple[int, int, int] = (255, 128, 0)  # Orange (BGR)

    # Cached values
    _pixel_points: np.ndarray | None = field(default=None, repr=False)
    _frame_shape: tuple[int, int] | None = field(default=None, repr=False)

    def get_pixel_points(self, frame_shape: tuple[int, int]) -> np.ndarray:
        """Convert normalized points to pixel coordinates."""
        height, width = frame_shape[:2]

        if self._pixel_points is not None and self._frame_shape == (height, width):
            return self._pixel_points

        self._pixel_points = np.array(
            [[int(p[0] * width), int(p[1] * height)] for p in self.points],
            dtype=np.int32
        )
        self._frame_shape = (height, width)
        return self._pixel_points

    def get_normal_vector(self) -> np.ndarray:
        """
        Get the unit normal vector based on user-defined angle.

        Returns:
            Unit normal vector as np.ndarray([nx, ny])
            Direction: 0°=right, 90°=down, 180°=left, 270°=up
        """
        angle_rad = np.radians(self.normal_angle)
        return np.array([np.cos(angle_rad), np.sin(angle_rad)])

    def get_center(self, frame_shape: tuple[int, int]) -> np.ndarray:
        """Get the center point of the door polygon in pixel coordinates."""
        pts = self.get_pixel_points(frame_shape)
        return np.mean(pts, axis=0)

    def contains_point(self, point: tuple[float, float], frame_shape: tuple[int, int]) -> bool:
        """Check if a point is inside the door polygon."""
        if len(self.points) < 3:
            return False

        poly_pts = self.get_pixel_points(frame_shape)
        dist = cv2.pointPolygonTest(poly_pts, point, False)
        return dist >= 0

    def check_crossing(
        self,
        current_pos: tuple[float, float],
        velocity: np.ndarray,
        frame_shape: tuple[int, int],
        min_speed: float = 5.0
    ) -> str | None:
        """
        Check if a detection is crossing the door and determine direction.

        Uses the user-defined normal_angle and direction_tolerance to determine
        if the person is entering or exiting.

        Args:
            current_pos: Current centroid position (x, y) in pixels
            velocity: Velocity vector (vx, vy) in pixels/second
            frame_shape: Frame dimensions (height, width)
            min_speed: Minimum speed to consider as movement (pixels/second)

        Returns:
            'enter' if moving within tolerance of normal direction
            'exit' if moving within tolerance of opposite direction (180°)
            None if not crossing, not moving fast enough, or outside tolerance
        """
        # Check if point is inside door polygon
        if not self.contains_point(current_pos, frame_shape):
            return None

        # Check if moving fast enough
        speed = np.linalg.norm(velocity)
        if speed < min_speed:
            return None

        # Get the normal vector (user-defined enter direction)
        normal = self.get_normal_vector()

        # Normalize velocity to get direction
        velocity_unit = velocity / speed

        # Calculate angle between velocity and normal
        # dot = cos(angle)
        dot = np.dot(velocity_unit, normal)

        # Clamp to valid range for acos
        dot = np.clip(dot, -1.0, 1.0)
        angle_diff = np.degrees(np.arccos(dot))  # 0° = same direction, 180° = opposite

        # Check if within tolerance of "enter" direction (normal)
        if angle_diff <= self.direction_tolerance:
            return 'enter'

        # Check if within tolerance of "exit" direction (opposite of normal = 180°)
        if angle_diff >= (180.0 - self.direction_tolerance):
            return 'exit'

        return None

    @classmethod
    def from_config(cls, config_door) -> 'Door':
        """Create a Door from a config Door pydantic model."""
        points = []
        for p in config_door.points:
            if hasattr(p, 'x') and hasattr(p, 'y'):
                points.append((p.x, p.y))
            else:
                points.append((p[0], p[1]))

        return cls(
            name=config_door.name,
            points=points,
            triggers=getattr(config_door, 'triggers', ['person']),
            normal_angle=getattr(config_door, 'normal_angle', 0.0),
            direction_tolerance=getattr(config_door, 'direction_tolerance', 60.0)
        )


# COCO keypoint indices for ground point calculation
KEYPOINT_LEFT_ANKLE = 15
KEYPOINT_RIGHT_ANKLE = 16
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12


@dataclass
class GroundPointInfo:
    """Information about the calculated ground point for visualization."""
    point: tuple[float, float]  # (x, y) in pixels
    source: str  # 'ankles', 'left_ankle', 'right_ankle', 'hips_estimated', 'bbox_bottom', 'predicted'
    is_estimated: bool  # True if using fallback (hips, bbox, or predicted)


@dataclass
class TrackTripwireState:
    """
    Minimal state for tripwire crossing detection.

    Tracks: which side of line, last position/velocity for prediction.
    """
    # Which side of the tripwire: +1 = positive side (normal direction), -1 = negative side, 0 = unknown
    side: int = 0
    # Last known good data for prediction when person vanishes
    last_ground_point: tuple[float, float] | None = None
    last_velocity: tuple[float, float] | None = None  # pixels/second
    last_update_time: float = 0.0
    # Track if this is first time we see this track near this door
    first_seen_time: float | None = None
    # Did we already fire an "appear" event for this track?
    appear_event_fired: bool = False
    # Did we already fire a crossing event? (prevents double-firing on vanish)
    crossing_event_fired: bool = False


@dataclass
class TripwireDetector:
    """
    Robust tripwire detector using signed distance + velocity direction.

    Detection logic:
    1. Track which SIDE of the tripwire each person is on (+1 or -1)
    2. When side changes OR person vanishes near tripwire:
       - Use velocity direction to determine enter/exit
       - Velocity pointing WITH door normal = "enter"
       - Velocity pointing AGAINST door normal = "exit"

    This is simpler and more robust because:
    - Side change is a clear geometric event
    - Velocity direction is a strong signal for intent
    - Handles vanishing persons naturally
    """
    door: Door

    # Configurable parameters
    hip_y_offset_ratio: float = 0.4  # Estimate feet below hips
    min_keypoint_confidence: float = 0.3

    # Minimum distance from line to consider "on a side" (prevents jitter)
    side_threshold: float = 15.0  # pixels

    # Prediction timeout for vanishing persons
    prediction_timeout: float = 0.5  # seconds

    # Minimum speed to consider velocity valid
    min_velocity: float = 20.0  # pixels/second

    # Direction tolerance (degrees) - how close velocity must align with normal
    direction_tolerance: float = 75.0  # degrees from normal (generous for robustness)

    # State per track
    _track_states: dict[int, TrackTripwireState] = field(default_factory=dict)

    # Ground points for visualization
    _last_ground_points: dict[int, GroundPointInfo] = field(default_factory=dict)

    # Cached line equation
    _line_coeffs: tuple[float, float, float] | None = field(default=None, repr=False)
    _line_frame_shape: tuple[int, int] | None = field(default=None, repr=False)

    def get_ground_points_for_visualization(self) -> dict[int, GroundPointInfo]:
        """Get ground points for visualization."""
        return self._last_ground_points.copy()

    def get_distance_to_tripwire(self, point: tuple[float, float], frame_shape: tuple[int, int]) -> float:
        """
        Get perpendicular distance from point to tripwire line segment.

        Returns the actual distance (not signed), considering the line segment bounds.
        Used for choosing the closest door when multiple doors are nearby.
        """
        p1, p2 = self.get_threshold_line(frame_shape)

        line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            # Degenerate line - return distance to point
            return np.sqrt((point[0] - p1[0])**2 + (point[1] - p1[1])**2)

        line_unit = line_vec / line_len
        point_vec = np.array([point[0] - p1[0], point[1] - p1[1]])
        proj_length = np.dot(point_vec, line_unit)

        # Clamp projection to line segment
        proj_length = max(0, min(line_len, proj_length))

        # Closest point on line segment
        closest = np.array([p1[0], p1[1]]) + line_unit * proj_length

        # Distance from point to closest point on segment
        return np.sqrt((point[0] - closest[0])**2 + (point[1] - closest[1])**2)

    def get_threshold_line(self, frame_shape: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get tripwire line (bottom edge of door polygon)."""
        pts = self.door.get_pixel_points(frame_shape)

        if len(pts) < 4:
            if len(pts) >= 2:
                return (tuple(pts[0]), tuple(pts[1]))
            return ((0, 0), (1, 0))

        # Bottom edge = two points with highest Y
        sorted_by_y = sorted(range(len(pts)), key=lambda i: pts[i][1], reverse=True)
        idx1, idx2 = sorted_by_y[0], sorted_by_y[1]
        p1, p2 = pts[idx1], pts[idx2]

        # Consistent left-to-right ordering
        if p1[0] > p2[0]:
            p1, p2 = p2, p1

        return (tuple(p1), tuple(p2))

    def get_line_equation(self, frame_shape: tuple[int, int]) -> tuple[float, float, float]:
        """Get normalized line equation Ax + By + C = 0."""
        height, width = frame_shape[:2]

        if self._line_coeffs is not None and self._line_frame_shape == (height, width):
            return self._line_coeffs

        p1, p2 = self.get_threshold_line(frame_shape)
        x1, y1 = p1
        x2, y2 = p2

        A = y2 - y1
        B = x1 - x2
        C = (x2 - x1) * y1 - (y2 - y1) * x1

        norm = np.sqrt(A * A + B * B)
        if norm > 1e-6:
            A, B, C = A / norm, B / norm, C / norm

        self._line_coeffs = (A, B, C)
        self._line_frame_shape = (height, width)
        return self._line_coeffs

    def signed_distance(self, point: tuple[float, float], frame_shape: tuple[int, int]) -> float:
        """
        Signed distance from point to tripwire.
        Positive = side where door normal points, Negative = opposite side.
        """
        A, B, C = self.get_line_equation(frame_shape)
        dist = A * point[0] + B * point[1] + C

        # Align sign with door normal direction
        line_normal = np.array([A, B])
        door_normal = self.door.get_normal_vector()
        if np.dot(line_normal, door_normal) < 0:
            dist = -dist

        return dist

    def is_near_door(self, point: tuple[float, float], frame_shape: tuple[int, int]) -> bool:
        """
        Check if point is near this door's tripwire.

        Requirements:
        1. Point must project onto the tripwire segment (perpendicular projection
           falls between the endpoints, not to either side)
        2. Perpendicular distance must be < 0.75 × tripwire length
        """
        p1, p2 = self.get_threshold_line(frame_shape)

        line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            return False

        line_unit = line_vec / line_len
        point_vec = np.array([point[0] - p1[0], point[1] - p1[1]])
        proj_length = np.dot(point_vec, line_unit)

        # Must project onto line segment (no extension to either side)
        if proj_length < 0 or proj_length > line_len:
            return False

        # Perpendicular distance must be < 1.0 × tripwire length
        perp_dist = abs(self.signed_distance(point, frame_shape))
        max_dist = 1.0 * line_len
        return perp_dist <= max_dist

    def get_ground_point(self, keypoints: np.ndarray, bbox: np.ndarray) -> GroundPointInfo | None:
        """Calculate ground point from keypoints (ankles preferred, then hips+offset)."""
        min_conf = self.min_keypoint_confidence

        # Try ankles
        left_ankle = keypoints[KEYPOINT_LEFT_ANKLE]
        right_ankle = keypoints[KEYPOINT_RIGHT_ANKLE]

        if left_ankle[2] >= min_conf and right_ankle[2] >= min_conf:
            x = (left_ankle[0] + right_ankle[0]) / 2
            y = (left_ankle[1] + right_ankle[1]) / 2
            return GroundPointInfo(point=(x, y), source='ankles', is_estimated=False)
        elif left_ankle[2] >= min_conf:
            return GroundPointInfo(point=(left_ankle[0], left_ankle[1]), source='left_ankle', is_estimated=False)
        elif right_ankle[2] >= min_conf:
            return GroundPointInfo(point=(right_ankle[0], right_ankle[1]), source='right_ankle', is_estimated=False)

        # Fallback to hips
        left_hip = keypoints[KEYPOINT_LEFT_HIP]
        right_hip = keypoints[KEYPOINT_RIGHT_HIP]

        if left_hip[2] >= min_conf or right_hip[2] >= min_conf:
            if left_hip[2] >= min_conf and right_hip[2] >= min_conf:
                hip_x = (left_hip[0] + right_hip[0]) / 2
                hip_y = (left_hip[1] + right_hip[1]) / 2
            elif left_hip[2] >= min_conf:
                hip_x, hip_y = left_hip[0], left_hip[1]
            else:
                hip_x, hip_y = right_hip[0], right_hip[1]

            bbox_height = bbox[3] - bbox[1]
            y_offset = bbox_height * self.hip_y_offset_ratio
            return GroundPointInfo(point=(hip_x, hip_y + y_offset), source='hips_estimated', is_estimated=True)

        # Last fallback: bbox bottom center
        x = (bbox[0] + bbox[2]) / 2
        y = bbox[3]
        return GroundPointInfo(point=(x, y), source='bbox_bottom', is_estimated=True)

    def _get_direction_from_velocity(self, velocity: tuple[float, float]) -> str | None:
        """
        Determine enter/exit direction from velocity vector.

        Returns:
            'enter' if velocity aligns with door normal (within tolerance)
            'exit' if velocity aligns opposite to door normal
            None if velocity too slow or perpendicular
        """
        vel = np.array(velocity)
        speed = np.linalg.norm(vel)

        if speed < self.min_velocity:
            return None

        vel_unit = vel / speed
        door_normal = self.door.get_normal_vector()

        # dot = cos(angle between velocity and normal)
        dot = np.dot(vel_unit, door_normal)
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(dot)))

        # Check if within tolerance of normal or opposite
        if angle <= self.direction_tolerance:
            # Aligned with or opposite to normal
            if dot > 0:
                return 'enter'  # Moving with normal
            else:
                return 'exit'   # Moving against normal

        return None  # Moving perpendicular-ish

    def check_crossing(
        self,
        track_id: int,
        keypoints: np.ndarray,
        bbox: np.ndarray,
        frame_shape: tuple[int, int],
        velocity: np.ndarray | None = None,
        timestamp: float | None = None,
        detection_present: bool = True
    ) -> str | None:
        """
        Check if person crossed tripwire and determine direction.

        Logic:
        1. If detection present: update side, check for side change
        2. If detection vanished: use velocity prediction
        3. Direction determined by velocity alignment with door normal

        Args:
            track_id: Track ID
            keypoints: Pose keypoints (17, 3)
            bbox: Bounding box (4,)
            frame_shape: (height, width)
            velocity: Velocity vector (vx, vy) pixels/second
            timestamp: Current time
            detection_present: False if track disappeared (use prediction)

        Returns:
            'enter', 'exit', or None
        """
        if timestamp is None:
            timestamp = time.time()

        # Get or create state
        is_new_to_this_door = track_id not in self._track_states
        if is_new_to_this_door:
            self._track_states[track_id] = TrackTripwireState()
        state = self._track_states[track_id]

        ground_point: tuple[float, float] | None = None
        current_velocity: tuple[float, float] | None = None

        if detection_present:
            # Get ground point from keypoints
            ground_info = self.get_ground_point(keypoints, bbox)
            if ground_info is None:
                return None

            ground_point = ground_info.point

            # Only track if near this door
            if not self.is_near_door(ground_point, frame_shape):
                return None

            # Store for visualization and prediction
            self._last_ground_points[track_id] = ground_info
            state.last_ground_point = ground_point
            if velocity is not None:
                state.last_velocity = (float(velocity[0]), float(velocity[1]))
                current_velocity = state.last_velocity
            state.last_update_time = timestamp

            # Track first seen time for appearing detection
            if state.first_seen_time is None:
                state.first_seen_time = timestamp

        else:
            # Detection vanished - check if they were heading into/out of this door

            if (state.last_ground_point is None or
                timestamp - state.last_update_time > self.prediction_timeout):
                return None

            # Must be near the door to trigger on vanish
            if not self.is_near_door(state.last_ground_point, frame_shape):
                return None

            # Skip if we already fired a crossing event for this track
            # (prevents double-firing: once on side change, once on vanish)
            if state.crossing_event_fired:
                return None

            # Must have velocity that matches door direction
            if state.last_velocity is None:
                return None

            current_velocity = state.last_velocity
            direction = self._get_direction_from_velocity(current_velocity)

            if direction is None:
                # Velocity doesn't match door direction - just walking past
                return None

            # Velocity matches! They were heading into/out of the door
            # Mark this track as handled so we don't trigger again
            state.last_velocity = None  # Prevent re-triggering
            state.crossing_event_fired = True

            # Store predicted point for visualization
            dt = timestamp - state.last_update_time
            predicted_x = state.last_ground_point[0] + current_velocity[0] * dt
            predicted_y = state.last_ground_point[1] + current_velocity[1] * dt
            self._last_ground_points[track_id] = GroundPointInfo(
                point=(predicted_x, predicted_y), source='predicted', is_estimated=True
            )

            return direction

        # Calculate current side (+1, -1, or 0 if too close to line)
        dist = self.signed_distance(ground_point, frame_shape)
        if dist > self.side_threshold:
            current_side = 1
        elif dist < -self.side_threshold:
            current_side = -1
        else:
            current_side = 0  # On/near the line

        prev_side = state.side
        crossing_detected = None

        # Case 1: Geometric side change (person physically crossed the line)
        # Skip if we already fired any crossing event for this track
        if (not state.crossing_event_fired and
            prev_side != 0 and current_side != 0 and prev_side != current_side):
            # Side changed! Use velocity to determine direction
            if current_velocity:
                crossing_detected = self._get_direction_from_velocity(current_velocity)
                if crossing_detected:
                    state.crossing_event_fired = True

        # Case 2: Track appeared near door with matching velocity (exiting from door)
        # Only fire once per track, and only if we have velocity established
        if (crossing_detected is None and
            not state.crossing_event_fired and
            not state.appear_event_fired and
            state.first_seen_time is not None and
            timestamp - state.first_seen_time < 0.5 and  # Within 0.5s of first appearing
            current_velocity is not None):

            direction = self._get_direction_from_velocity(current_velocity)
            if direction == 'exit':
                # Person appeared near door moving away from it - they exited!
                crossing_detected = 'exit'
                state.appear_event_fired = True
                state.crossing_event_fired = True

        # Update state
        if current_side != 0:
            state.side = current_side

        return crossing_detected

    def cleanup_stale_tracks(self, active_track_ids: set) -> None:
        """Remove state for tracks that no longer exist."""
        stale = [tid for tid in self._track_states if tid not in active_track_ids]
        for tid in stale:
            del self._track_states[tid]
            if tid in self._last_ground_points:
                del self._last_ground_points[tid]

    @classmethod
    def from_door(cls, door: Door) -> 'TripwireDetector':
        """Create detector from Door config."""
        return cls(door=door)


@dataclass
class Detection:
    """
    Represents a detected and tracked person in a video stream.

    This is the core domain model that holds all state for a tracked entity:
    - Pose keypoints from YOLO
    - Tracking ID from ByteTrack
    - Centroid for zone detection
    - Track history for visualization
    - Velocity vector (direction and speed)
    - Current zones the person is in
    - Future: detected action from action recognition model
    """
    track_id: int
    keypoints: np.ndarray  # Shape: (17, 3) - x, y, confidence
    bbox: np.ndarray  # Shape: (4,) - x1, y1, x2, y2
    confidence: float

    # Computed properties
    centroid: np.ndarray | None = None  # (x, y) in pixel coordinates

    # Velocity vector (pixels per second)
    # velocity[0] = dx/dt (positive = moving right)
    # velocity[1] = dy/dt (positive = moving down)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # Speed in pixels per second (magnitude of velocity)
    speed: float = 0.0

    # State that persists across frames
    track_history: list[tuple[float, float]] = field(default_factory=list)
    _timestamp_history: list[float] = field(default_factory=list, repr=False)  # For velocity calculation
    zones: list[str] = field(default_factory=list)

    # Future: action recognition result
    action: str | None = None
    action_confidence: float = 0.0

    # Timestamp for cleanup
    last_seen: float = 0.0

    # Maximum history length (roughly 2 seconds at 15fps)
    MAX_HISTORY_LENGTH: int = field(default=30, repr=False)

    # Exponential moving average alpha for velocity smoothing
    # Higher = more responsive but noisier, lower = smoother but laggier
    VELOCITY_ALPHA: float = field(default=0.2, repr=False)

    def __post_init__(self):
        """Calculate centroid if not provided."""
        if self.centroid is None:
            self.centroid = calculate_centroid(self.keypoints)

    def update(self, keypoints: np.ndarray, bbox: np.ndarray, confidence: float, timestamp: float):
        """
        Update detection with new frame data.

        Args:
            keypoints: New keypoints from YOLO
            bbox: New bounding box
            confidence: Detection confidence
            timestamp: Current time for history management
        """
        self.keypoints = keypoints
        self.bbox = bbox
        self.confidence = confidence

        # Recalculate centroid
        self.centroid = calculate_centroid(keypoints)

        # Update track history and velocity
        if self.centroid is not None:
            cx, cy = self.centroid
            self.track_history.append((float(cx), float(cy)))
            self._timestamp_history.append(timestamp)

            # Trim history to max length
            if len(self.track_history) > self.MAX_HISTORY_LENGTH:
                self.track_history.pop(0)
                self._timestamp_history.pop(0)

            # Calculate velocity using exponential moving average
            self._update_velocity()

        self.last_seen = timestamp

        # Clear zones - will be recalculated by zone checker
        self.zones = []

    def _update_velocity(self):
        """
        Calculate velocity using exponential moving average of recent displacements.

        Uses the last few positions to compute instantaneous velocities,
        then applies EMA smoothing to reduce noise while remaining responsive.
        """
        if len(self.track_history) < 2 or len(self._timestamp_history) < 2:
            return

        # Calculate instantaneous velocity from last two positions
        p1 = np.array(self.track_history[-2])
        p2 = np.array(self.track_history[-1])
        t1 = self._timestamp_history[-2]
        t2 = self._timestamp_history[-1]

        dt = t2 - t1
        if dt <= 0:
            return

        # Displacement in pixels
        displacement = p2 - p1

        # Instantaneous velocity (pixels per second)
        instant_velocity = displacement / dt

        # Apply exponential moving average for smoothing
        # v_new = alpha * v_instant + (1 - alpha) * v_old
        self.velocity = self.VELOCITY_ALPHA * instant_velocity + (1 - self.VELOCITY_ALPHA) * self.velocity

        # Update speed (magnitude of velocity)
        self.speed = float(np.linalg.norm(self.velocity))

    def get_direction_angle(self) -> float:
        """
        Get the direction of movement as an angle in degrees.

        Returns:
            Angle in degrees (0 = right, 90 = down, 180 = left, 270 = up)
            Returns 0 if not moving.
        """
        if self.speed < 1.0:  # Threshold for "stationary"
            return 0.0
        return float(np.degrees(np.arctan2(self.velocity[1], self.velocity[0])))

    def get_direction_name(self) -> str:
        """
        Get a human-readable direction name.

        Returns:
            One of: 'stationary', 'right', 'down-right', 'down', 'down-left',
                    'left', 'up-left', 'up', 'up-right'
        """
        if self.speed < 10.0:  # pixels/sec threshold for "stationary"
            return 'stationary'

        angle = self.get_direction_angle()

        # Normalize to 0-360
        if angle < 0:
            angle += 360

        # Map to 8 cardinal/ordinal directions
        directions = ['right', 'down-right', 'down', 'down-left',
                      'left', 'up-left', 'up', 'up-right']
        idx = int((angle + 22.5) / 45) % 8
        return directions[idx]

    def check_zones(self, zones: list[Zone], frame_shape: tuple[int, int], radius: float = 0) -> list[dict[str, Any]]:
        """
        Check which zones this detection is currently in.

        Args:
            zones: List of Zone objects to check
            frame_shape: (height, width) of the frame
            radius: Optional radius around centroid for overlap detection

        Returns:
            List of zone event dictionaries
        """
        events = []
        self.zones = []  # Reset zones

        if self.centroid is None:
            return events

        point = (float(self.centroid[0]), float(self.centroid[1]))

        for zone in zones:
            if zone.contains_point(point, frame_shape, radius):
                if zone.name not in self.zones:
                    self.zones.append(zone.name)
                    events.append({
                        'zone': zone.name,
                        'detection_id': self.track_id,
                        'triggers': zone.triggers
                    })

        return events


class DetectionManager:
    """
    Manages the lifecycle of Detection objects for a single stream.

    Responsibilities:
    - Create new Detection objects for new track IDs
    - Update existing Detection objects with new frame data
    - Clean up stale Detection objects that haven't been seen recently
    """

    def __init__(self, max_age: float = 2.0):
        """
        Args:
            max_age: Maximum seconds since last seen before removing a detection
        """
        self.detections: dict[int, Detection] = {}
        self.max_age = max_age

    def update(self, raw_detections: list[dict], timestamp: float) -> list[Detection]:
        """
        Update detections with new frame data from the inference model.

        Args:
            raw_detections: List of detection dicts from PoseEstimator.predict()
                           Each contains: 'id', 'keypoints', 'bbox', 'score'
            timestamp: Current time

        Returns:
            List of current Detection objects
        """
        seen_ids = set()

        for raw in raw_detections:
            track_id = raw.get('id', -1)

            # Skip untracked detections
            if track_id == -1:
                continue

            seen_ids.add(track_id)

            if track_id in self.detections:
                # Update existing detection
                self.detections[track_id].update(
                    keypoints=raw['keypoints'],
                    bbox=raw['bbox'],
                    confidence=raw['score'],
                    timestamp=timestamp
                )
            else:
                # Create new detection
                self.detections[track_id] = Detection(
                    track_id=track_id,
                    keypoints=raw['keypoints'],
                    bbox=raw['bbox'],
                    confidence=raw['score'],
                    last_seen=timestamp
                )
                # Initialize track history with current centroid and timestamp
                det = self.detections[track_id]
                if det.centroid is not None:
                    det.track_history.append((float(det.centroid[0]), float(det.centroid[1])))
                    det._timestamp_history.append(timestamp)

        # Clean up stale detections
        to_remove = [
            tid for tid, det in self.detections.items()
            if timestamp - det.last_seen > self.max_age
        ]
        for tid in to_remove:
            del self.detections[tid]

        # Return list of current detections
        return list(self.detections.values())

    def get_detection(self, track_id: int) -> Detection | None:
        """Get a detection by its track ID."""
        return self.detections.get(track_id)

    def clear(self):
        """Clear all detections."""
        self.detections.clear()
