"""
Rule-Based Action Recognition Module.

This module implements CPU-side rule-based pose classification using the COCO 17-joint
skeleton. It serves as a simple but functional action recognition stage that can be
replaced by a YOLO11-Classify model in the future.

Supported Actions:
- standing: Upright posture with straight legs
- sitting: Bent knees, hips lower relative to body
- lying_down: Horizontal body orientation
- walking: Standing with leg movement (uses velocity)
- reading: Arms raised holding imaginary book
- phone_texting: Both hands close together at chest/stomach level
- phone_calling: One hand raised to ear
- arms_raised: Both arms raised above shoulders
- unknown: Default when no rule matches confidently

The classifier uses geometric relationships (angles, ratios, distances) rather than
raw pixel values to be robust across different person sizes and camera perspectives.

Reference: COCO Keypoint indices:
    0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
    5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow,
    9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip,
    13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# COCO keypoint indices
class Keypoint(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Minimum confidence threshold for a keypoint to be considered valid
MIN_CONFIDENCE = 0.3

# Action labels
ACTION_STANDING = "standing"
ACTION_SITTING = "sitting"
ACTION_LYING_DOWN = "lying_down"
ACTION_WALKING = "walking"
ACTION_READING = "reading"
ACTION_PHONE_TEXTING = "phone_texting"
ACTION_PHONE_CALLING = "phone_calling"
ACTION_ARMS_RAISED = "arms_raised"
ACTION_UNKNOWN = "unknown"


@dataclass
class ActionResult:
    """Result of action classification."""
    action: str
    confidence: float
    details: dict[str, float]  # Debug info about the classification


class PoseGeometry:
    """
    Utility class for computing geometric properties from pose keypoints.

    All methods handle missing keypoints gracefully by returning None
    when required keypoints are not available.
    """

    def __init__(self, keypoints: np.ndarray, min_conf: float = MIN_CONFIDENCE):
        """
        Args:
            keypoints: (17, 3) array with x, y, confidence for each keypoint
            min_conf: Minimum confidence to consider a keypoint valid
        """
        self.kpts = keypoints
        self.min_conf = min_conf

    def is_valid(self, idx: int) -> bool:
        """Check if a keypoint has sufficient confidence."""
        return self.kpts[idx, 2] >= self.min_conf

    def get_point(self, idx: int) -> np.ndarray | None:
        """Get (x, y) of a keypoint if valid, else None."""
        if self.is_valid(idx):
            return self.kpts[idx, :2]
        return None

    def get_midpoint(self, idx1: int, idx2: int) -> np.ndarray | None:
        """Get midpoint between two keypoints if both valid."""
        p1 = self.get_point(idx1)
        p2 = self.get_point(idx2)
        if p1 is not None and p2 is not None:
            return (p1 + p2) / 2
        return None

    def get_distance(self, idx1: int, idx2: int) -> float | None:
        """Get Euclidean distance between two keypoints if both valid."""
        p1 = self.get_point(idx1)
        p2 = self.get_point(idx2)
        if p1 is not None and p2 is not None:
            return float(np.linalg.norm(p1 - p2))
        return None

    def get_angle(self, idx1: int, idx2: int, idx3: int) -> float | None:
        """
        Calculate angle at idx2 formed by idx1-idx2-idx3.

        Returns angle in degrees (0-180).
        """
        p1 = self.get_point(idx1)
        p2 = self.get_point(idx2)
        p3 = self.get_point(idx3)

        if p1 is None or p2 is None or p3 is None:
            return None

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def get_vertical_angle(self, idx1: int, idx2: int) -> float | None:
        """
        Get angle of vector from idx1 to idx2 relative to vertical (down).

        Returns angle in degrees where:
        - 0° = perfectly vertical (idx2 directly below idx1)
        - 90° = horizontal
        - 180° = inverted vertical
        """
        p1 = self.get_point(idx1)
        p2 = self.get_point(idx2)

        if p1 is None or p2 is None:
            return None

        vec = p2 - p1
        vertical = np.array([0, 1])  # Down in image coordinates

        cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def get_skeleton_bbox(self) -> tuple[float, float, float, float] | None:
        """
        Get bounding box of visible keypoints.

        Returns (min_x, min_y, max_x, max_y) or None if no valid keypoints.
        """
        valid_mask = self.kpts[:, 2] >= self.min_conf
        if not np.any(valid_mask):
            return None

        valid_pts = self.kpts[valid_mask, :2]
        min_x, min_y = valid_pts.min(axis=0)
        max_x, max_y = valid_pts.max(axis=0)
        return (float(min_x), float(min_y), float(max_x), float(max_y))

    def get_bbox_aspect_ratio(self) -> float | None:
        """
        Get width/height ratio of skeleton bounding box.

        > 1.0 means wider than tall (lying down)
        < 1.0 means taller than wide (standing/sitting)
        """
        bbox = self.get_skeleton_bbox()
        if bbox is None:
            return None

        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y

        if height < 1.0:
            return None

        return width / height

    def get_shoulder_width(self) -> float | None:
        """Get distance between shoulders (reference for normalization)."""
        return self.get_distance(Keypoint.LEFT_SHOULDER.value, Keypoint.RIGHT_SHOULDER.value)

    def get_torso_height(self) -> float | None:
        """Get average distance from shoulders to hips."""
        left = self.get_distance(Keypoint.LEFT_SHOULDER.value, Keypoint.LEFT_HIP.value)
        right = self.get_distance(Keypoint.RIGHT_SHOULDER.value, Keypoint.RIGHT_HIP.value)

        if left is not None and right is not None:
            return (left + right) / 2
        return left or right


class RuleBasedActionClassifier:
    """
    Classifies human actions based on pose keypoint geometry.

    Uses a hierarchy of rules:
    1. Check body orientation (lying down vs upright)
    2. Check posture (standing vs sitting)
    3. Check arm positions (reading, phone use, arms raised)
    4. Consider velocity for walking detection
    """

    def __init__(self):
        """Initialize the classifier with default thresholds."""
        # Thresholds (can be tuned)
        self.lying_aspect_ratio_threshold = 1.0  # Width/height > 1.0 = lying
        self.lying_spine_angle_threshold = 60.0  # Spine angle from vertical > 60° = lying

        self.sitting_knee_angle_threshold = 140.0  # Knee angle < 140° = bent = sitting
        self.sitting_hip_ratio_threshold = 0.7  # Hip height / total height < 0.7 = sitting

        self.walking_speed_threshold = 30.0  # pixels/sec to be considered walking

        self.reading_elbow_angle_range = (60, 140)  # Elbows bent for reading
        self.reading_wrist_height_ratio = (0.3, 0.8)  # Wrists between 30-80% up the torso

        self.phone_wrist_proximity_threshold = 0.5  # Wrists within 50% of shoulder width
        self.phone_call_head_proximity_threshold = 0.4  # Wrist within 40% of shoulder width from head

        self.arms_raised_threshold = 0.2  # Wrists above shoulder level by this ratio of torso

    def classify(self, keypoints: np.ndarray, velocity: np.ndarray | None = None,
                 speed: float = 0.0) -> ActionResult:
        """
        Classify the action from pose keypoints.

        Args:
            keypoints: (17, 3) array of COCO keypoints
            velocity: Optional (2,) velocity vector in pixels/sec
            speed: Speed in pixels/sec (magnitude of velocity)

        Returns:
            ActionResult with action label and confidence
        """
        geom = PoseGeometry(keypoints)
        details = {}

        # Check if we have enough keypoints
        valid_count = np.sum(keypoints[:, 2] >= MIN_CONFIDENCE)
        if valid_count < 5:
            return ActionResult(ACTION_UNKNOWN, 0.0, {"valid_keypoints": valid_count})

        details["valid_keypoints"] = float(valid_count)

        # 1. Check for lying down (highest priority - distinct body orientation)
        lying_result = self._check_lying_down(geom, details)
        if lying_result is not None:
            return lying_result

        # 2. Check for arms raised (before other arm-based actions)
        arms_raised_result = self._check_arms_raised(geom, details)
        if arms_raised_result is not None:
            return arms_raised_result

        # 3. Check for phone calling (one hand to head)
        phone_call_result = self._check_phone_calling(geom, details)
        if phone_call_result is not None:
            return phone_call_result

        # 4. Check for phone texting (both hands together)
        phone_text_result = self._check_phone_texting(geom, details)
        if phone_text_result is not None:
            return phone_text_result

        # 5. Check for reading posture
        reading_result = self._check_reading(geom, details)
        if reading_result is not None:
            return reading_result

        # 6. Check standing vs sitting
        posture_result = self._check_posture(geom, details)
        if posture_result is not None:
            # If standing and moving, classify as walking
            if posture_result.action == ACTION_STANDING and speed > self.walking_speed_threshold:
                details["speed"] = speed
                return ActionResult(ACTION_WALKING, 0.8, details)
            return posture_result

        # Default: unknown
        return ActionResult(ACTION_UNKNOWN, 0.3, details)

    def _check_lying_down(self, geom: PoseGeometry, details: dict) -> ActionResult | None:
        """Check if person is lying down based on body orientation."""

        # Method 1: Bounding box aspect ratio
        aspect_ratio = geom.get_bbox_aspect_ratio()
        if aspect_ratio is not None:
            details["aspect_ratio"] = aspect_ratio
            if aspect_ratio > self.lying_aspect_ratio_threshold:
                return ActionResult(ACTION_LYING_DOWN, min(0.9, 0.5 + aspect_ratio * 0.3), details)

        # Method 2: Spine angle from vertical
        # Use midpoint of shoulders to midpoint of hips as spine
        mid_shoulder = geom.get_midpoint(Keypoint.LEFT_SHOULDER.value, Keypoint.RIGHT_SHOULDER.value)
        mid_hip = geom.get_midpoint(Keypoint.LEFT_HIP.value, Keypoint.RIGHT_HIP.value)

        if mid_shoulder is not None and mid_hip is not None:
            spine_vec = mid_hip - mid_shoulder
            vertical = np.array([0, 1])
            cos_angle = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            spine_angle = float(np.degrees(np.arccos(cos_angle)))
            details["spine_angle"] = spine_angle

            if spine_angle > self.lying_spine_angle_threshold:
                confidence = min(0.9, 0.5 + (spine_angle - 60) / 60)
                return ActionResult(ACTION_LYING_DOWN, confidence, details)

        return None

    def _check_posture(self, geom: PoseGeometry, details: dict) -> ActionResult | None:
        """Determine if person is standing or sitting."""

        # Method 1: Knee angles
        left_knee_angle = geom.get_angle(
            Keypoint.LEFT_HIP.value,
            Keypoint.LEFT_KNEE.value,
            Keypoint.LEFT_ANKLE.value
        )
        right_knee_angle = geom.get_angle(
            Keypoint.RIGHT_HIP.value,
            Keypoint.RIGHT_KNEE.value,
            Keypoint.RIGHT_ANKLE.value
        )

        knee_angles = [a for a in [left_knee_angle, right_knee_angle] if a is not None]

        if knee_angles:
            avg_knee_angle = np.mean(knee_angles)
            details["avg_knee_angle"] = avg_knee_angle

            if avg_knee_angle < self.sitting_knee_angle_threshold:
                confidence = min(0.85, 0.5 + (self.sitting_knee_angle_threshold - avg_knee_angle) / 80)
                return ActionResult(ACTION_SITTING, confidence, details)

        # Method 2: Hip height relative to ankle height
        bbox = geom.get_skeleton_bbox()
        mid_hip = geom.get_midpoint(Keypoint.LEFT_HIP.value, Keypoint.RIGHT_HIP.value)
        left_ankle = geom.get_point(Keypoint.LEFT_ANKLE.value)
        right_ankle = geom.get_point(Keypoint.RIGHT_ANKLE.value)
        mid_shoulder = geom.get_midpoint(Keypoint.LEFT_SHOULDER.value, Keypoint.RIGHT_SHOULDER.value)

        if bbox is not None and mid_hip is not None and mid_shoulder is not None:
            _min_x, min_y, _max_x, max_y = bbox
            total_height = max_y - min_y

            if total_height > 10:
                # Relative hip position (0 = top, 1 = bottom)
                hip_relative = (mid_hip[1] - min_y) / total_height
                details["hip_relative_position"] = hip_relative

                # Also check if legs are visible and their extent
                ankle_y = None
                if left_ankle is not None and right_ankle is not None:
                    ankle_y = (left_ankle[1] + right_ankle[1]) / 2
                elif left_ankle is not None:
                    ankle_y = left_ankle[1]
                elif right_ankle is not None:
                    ankle_y = right_ankle[1]

                if ankle_y is not None:
                    hip_ankle_ratio = (mid_hip[1] - min_y) / (ankle_y - min_y + 1e-6)
                    details["hip_ankle_ratio"] = hip_ankle_ratio

                    # In sitting, hips are lower in the skeleton (closer to ankles)
                    if hip_ankle_ratio > self.sitting_hip_ratio_threshold:
                        return ActionResult(ACTION_SITTING, 0.7, details)

        # Default to standing if upright
        return ActionResult(ACTION_STANDING, 0.6, details)

    def _check_arms_raised(self, geom: PoseGeometry, details: dict) -> ActionResult | None:
        """Check if both arms are raised above shoulders."""
        left_wrist = geom.get_point(Keypoint.LEFT_WRIST.value)
        right_wrist = geom.get_point(Keypoint.RIGHT_WRIST.value)
        left_shoulder = geom.get_point(Keypoint.LEFT_SHOULDER.value)
        right_shoulder = geom.get_point(Keypoint.RIGHT_SHOULDER.value)

        torso_height = geom.get_torso_height()

        if torso_height is None or torso_height < 10:
            return None

        threshold = torso_height * self.arms_raised_threshold

        left_raised = False
        right_raised = False

        if left_wrist is not None and left_shoulder is not None:
            # In image coords, y increases downward, so raised = wrist.y < shoulder.y
            left_raised = left_wrist[1] < left_shoulder[1] - threshold

        if right_wrist is not None and right_shoulder is not None:
            right_raised = right_wrist[1] < right_shoulder[1] - threshold

        details["left_arm_raised"] = left_raised
        details["right_arm_raised"] = right_raised

        if left_raised and right_raised:
            return ActionResult(ACTION_ARMS_RAISED, 0.85, details)

        return None

    def _check_phone_calling(self, geom: PoseGeometry, details: dict) -> ActionResult | None:
        """Check if one hand is raised to the head (phone call posture)."""
        left_wrist = geom.get_point(Keypoint.LEFT_WRIST.value)
        right_wrist = geom.get_point(Keypoint.RIGHT_WRIST.value)
        nose = geom.get_point(Keypoint.NOSE.value)
        left_ear = geom.get_point(Keypoint.LEFT_EAR.value)
        right_ear = geom.get_point(Keypoint.RIGHT_EAR.value)

        shoulder_width = geom.get_shoulder_width()
        if shoulder_width is None or shoulder_width < 10:
            return None

        threshold = shoulder_width * self.phone_call_head_proximity_threshold

        # Check if either wrist is near the head (ear or nose level)
        def near_head(wrist):
            if wrist is None:
                return False

            distances = []
            if nose is not None:
                distances.append(np.linalg.norm(wrist - nose))
            if left_ear is not None:
                distances.append(np.linalg.norm(wrist - left_ear))
            if right_ear is not None:
                distances.append(np.linalg.norm(wrist - right_ear))

            if distances:
                min_dist = min(distances)
                return min_dist < threshold
            return False

        left_near = near_head(left_wrist)
        right_near = near_head(right_wrist)

        details["left_wrist_near_head"] = left_near
        details["right_wrist_near_head"] = right_near

        # Only one hand should be at head for phone call
        if (left_near and not right_near) or (right_near and not left_near):
            return ActionResult(ACTION_PHONE_CALLING, 0.75, details)

        return None

    def _check_phone_texting(self, geom: PoseGeometry, details: dict) -> ActionResult | None:
        """Check if both hands are together in front of body (texting posture)."""
        left_wrist = geom.get_point(Keypoint.LEFT_WRIST.value)
        right_wrist = geom.get_point(Keypoint.RIGHT_WRIST.value)

        if left_wrist is None or right_wrist is None:
            return None

        shoulder_width = geom.get_shoulder_width()
        if shoulder_width is None or shoulder_width < 10:
            return None

        # Distance between wrists
        wrist_dist = float(np.linalg.norm(left_wrist - right_wrist))
        wrist_dist_normalized = wrist_dist / shoulder_width
        details["wrist_distance_normalized"] = wrist_dist_normalized

        # Check if wrists are close together
        if wrist_dist_normalized > self.phone_wrist_proximity_threshold:
            return None

        # Check if wrists are at torso level (not too high, not too low)
        mid_shoulder = geom.get_midpoint(Keypoint.LEFT_SHOULDER.value, Keypoint.RIGHT_SHOULDER.value)
        mid_hip = geom.get_midpoint(Keypoint.LEFT_HIP.value, Keypoint.RIGHT_HIP.value)

        if mid_shoulder is not None and mid_hip is not None:
            torso_top = mid_shoulder[1]
            torso_bottom = mid_hip[1]
            torso_height = torso_bottom - torso_top

            if torso_height > 10:
                avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
                wrist_relative = (avg_wrist_y - torso_top) / torso_height
                details["wrist_relative_height"] = wrist_relative

                # Wrists should be somewhere along the torso
                if 0.2 < wrist_relative < 1.5:
                    return ActionResult(ACTION_PHONE_TEXTING, 0.7, details)

        return None

    def _check_reading(self, geom: PoseGeometry, details: dict) -> ActionResult | None:
        """Check for reading posture (arms raised holding book)."""
        left_wrist = geom.get_point(Keypoint.LEFT_WRIST.value)
        right_wrist = geom.get_point(Keypoint.RIGHT_WRIST.value)

        if left_wrist is None or right_wrist is None:
            return None

        # Check elbow angles (bent for holding book)
        left_elbow_angle = geom.get_angle(
            Keypoint.LEFT_SHOULDER.value,
            Keypoint.LEFT_ELBOW.value,
            Keypoint.LEFT_WRIST.value
        )
        right_elbow_angle = geom.get_angle(
            Keypoint.RIGHT_SHOULDER.value,
            Keypoint.RIGHT_ELBOW.value,
            Keypoint.RIGHT_WRIST.value
        )

        elbow_angles = [a for a in [left_elbow_angle, right_elbow_angle] if a is not None]
        if not elbow_angles:
            return None

        avg_elbow_angle = np.mean(elbow_angles)
        details["avg_elbow_angle"] = avg_elbow_angle

        min_angle, max_angle = self.reading_elbow_angle_range
        if not (min_angle <= avg_elbow_angle <= max_angle):
            return None

        # Check wrist height (should be raised but not too high)
        mid_shoulder = geom.get_midpoint(Keypoint.LEFT_SHOULDER.value, Keypoint.RIGHT_SHOULDER.value)
        mid_hip = geom.get_midpoint(Keypoint.LEFT_HIP.value, Keypoint.RIGHT_HIP.value)

        if mid_shoulder is None or mid_hip is None:
            return None

        torso_height = mid_hip[1] - mid_shoulder[1]
        if torso_height < 10:
            return None

        avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
        # Relative position: 0 = shoulder level, 1 = hip level
        wrist_relative = (avg_wrist_y - mid_shoulder[1]) / torso_height
        details["reading_wrist_relative"] = wrist_relative

        min_h, max_h = self.reading_wrist_height_ratio
        if min_h <= wrist_relative <= max_h:
            # Also check wrists are spaced apart (holding something)
            shoulder_width = geom.get_shoulder_width()
            if shoulder_width is not None and shoulder_width > 10:
                wrist_spacing = float(np.linalg.norm(left_wrist - right_wrist))
                wrist_spacing_normalized = wrist_spacing / shoulder_width
                details["reading_wrist_spacing"] = wrist_spacing_normalized

                # Wrists should be moderately apart (30-100% of shoulder width)
                if 0.3 <= wrist_spacing_normalized <= 1.0:
                    return ActionResult(ACTION_READING, 0.65, details)

        return None


# =============================================================================
# Action Temporal Filter
# =============================================================================

@dataclass
class ActionState:
    """Tracks action state for a single detection with temporal filtering."""
    # Current confirmed (stable) action
    confirmed_action: str = ACTION_UNKNOWN
    confirmed_confidence: float = 0.0

    # Pending action being validated
    pending_action: str = ACTION_UNKNOWN
    pending_start_time: float = 0.0
    pending_frame_count: int = 0

    # History for debugging/analysis
    raw_action_history: deque = field(default_factory=lambda: deque(maxlen=60))


class ActionFilter:
    """
    Temporal filter for action recognition.

    Prevents flickering by requiring an action to be consistently detected
    for a minimum number of frames or duration before it's confirmed.

    This is similar to zone enter/exit debouncing but for actions.

    Example:
        filter = ActionFilter(confirm_frames=30)  # ~1 second at 30fps

        # In processing loop:
        raw_action, confidence = classify_action(keypoints)
        stable_action = filter.update(track_id, raw_action, confidence, timestamp)
    """

    def __init__(
        self,
        confirm_frames: int = 10,
        confirm_seconds: float = .5,
        use_frames: bool = True
    ):
        """
        Initialize the action filter.

        Args:
            confirm_frames: Number of consecutive frames required to confirm action
            confirm_seconds: Duration in seconds required to confirm action
            use_frames: If True, use frame count; if False, use time duration
        """
        self.confirm_frames = confirm_frames
        self.confirm_seconds = confirm_seconds
        self.use_frames = use_frames

        # State per track ID
        self._states: dict[int, ActionState] = {}

    def get_state(self, track_id: int) -> ActionState:
        """Get or create state for a track."""
        if track_id not in self._states:
            self._states[track_id] = ActionState()
        return self._states[track_id]

    def update(
        self,
        track_id: int,
        raw_action: str,
        confidence: float,
        timestamp: float
    ) -> tuple[str, float]:
        """
        Update action state with a new raw classification.

        Args:
            track_id: Detection track ID
            raw_action: Raw action classification from the classifier
            confidence: Confidence of the raw classification
            timestamp: Current timestamp

        Returns:
            Tuple of (confirmed_action, confirmed_confidence)
        """
        state = self.get_state(track_id)

        # Record raw action in history
        state.raw_action_history.append((timestamp, raw_action, confidence))

        # Check if raw action matches the pending action
        if raw_action == state.pending_action:
            # Same action continues - increment counter
            state.pending_frame_count += 1

            # Check if we've reached confirmation threshold
            confirmed = False
            if self.use_frames:
                confirmed = state.pending_frame_count >= self.confirm_frames
            else:
                elapsed = timestamp - state.pending_start_time
                confirmed = elapsed >= self.confirm_seconds

            if confirmed:
                # Action is now confirmed
                state.confirmed_action = state.pending_action
                state.confirmed_confidence = confidence
        else:
            # Action changed - reset pending state
            state.pending_action = raw_action
            state.pending_start_time = timestamp
            state.pending_frame_count = 1

        return state.confirmed_action, state.confirmed_confidence

    def get_confirmed_action(self, track_id: int) -> tuple[str, float]:
        """Get the current confirmed action for a track."""
        if track_id not in self._states:
            return ACTION_UNKNOWN, 0.0
        state = self._states[track_id]
        return state.confirmed_action, state.confirmed_confidence

    def get_pending_action(self, track_id: int) -> tuple[str, int]:
        """Get the pending action and its frame count for debugging."""
        if track_id not in self._states:
            return ACTION_UNKNOWN, 0
        state = self._states[track_id]
        return state.pending_action, state.pending_frame_count

    def remove_track(self, track_id: int) -> None:
        """Remove state for a track that's no longer active."""
        self._states.pop(track_id, None)

    def cleanup_stale_tracks(self, active_track_ids: set) -> None:
        """Remove state for tracks that are no longer active."""
        stale = [tid for tid in self._states if tid not in active_track_ids]
        for tid in stale:
            del self._states[tid]

    def clear(self) -> None:
        """Clear all state."""
        self._states.clear()


# Global classifier instance
_classifier: RuleBasedActionClassifier | None = None


def get_classifier() -> RuleBasedActionClassifier:
    """Get the singleton action classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = RuleBasedActionClassifier()
    return _classifier


def classify_action(keypoints: np.ndarray, velocity: np.ndarray | None = None,
                    speed: float = 0.0) -> tuple[str, float]:
    """
    Classify action from pose keypoints.

    Convenience function that uses the global classifier.

    Args:
        keypoints: (17, 3) COCO keypoints array
        velocity: Optional velocity vector (dx, dy) in pixels/sec
        speed: Speed in pixels/sec

    Returns:
        Tuple of (action_label, confidence)
    """
    classifier = get_classifier()
    result = classifier.classify(keypoints, velocity, speed)
    return result.action, result.confidence


def classify_detection(detection) -> None:
    """
    Classify action for a Detection object and update it in-place.

    Args:
        detection: Detection object with keypoints, velocity, and speed
    """
    action, confidence = classify_action(
        detection.keypoints,
        detection.velocity,
        detection.speed
    )
    detection.action = action
    detection.action_confidence = confidence


def classify_detections(detections: list) -> None:
    """
    Classify actions for a list of Detection objects.

    Updates each detection's action and action_confidence in-place.

    Args:
        detections: List of Detection objects
    """
    for det in detections:
        classify_detection(det)


def classify_detections_filtered(
    detections: list,
    action_filter: ActionFilter,
    timestamp: float
) -> None:
    """
    Classify actions for a list of Detection objects with temporal filtering.

    This applies the raw classification and then uses the ActionFilter to
    stabilize the action over time, requiring consistent detection before
    confirming an action change.

    Updates each detection's action and action_confidence in-place with the
    stable/confirmed action.

    Args:
        detections: List of Detection objects
        action_filter: ActionFilter instance for temporal smoothing
        timestamp: Current timestamp
    """
    active_ids = set()

    for det in detections:
        # Get raw classification
        raw_action, raw_confidence = classify_action(
            det.keypoints,
            det.velocity,
            det.speed
        )

        # Apply temporal filter
        stable_action, stable_confidence = action_filter.update(
            det.track_id,
            raw_action,
            raw_confidence,
            timestamp
        )

        # Update detection with stable action
        det.action = stable_action
        det.action_confidence = stable_confidence

        active_ids.add(det.track_id)

    # Clean up stale tracks from filter
    action_filter.cleanup_stale_tracks(active_ids)
