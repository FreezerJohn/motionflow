"""
Unit tests for core.actions module.

Tests the rule-based action recognition system.
"""

import numpy as np
import pytest

from core.actions import (
    ACTION_ARMS_RAISED,
    ACTION_LYING_DOWN,
    ACTION_PHONE_CALLING,
    ACTION_PHONE_TEXTING,
    ACTION_SITTING,
    ACTION_STANDING,
    ACTION_UNKNOWN,
    ACTION_WALKING,
    ActionFilter,
    PoseGeometry,
    RuleBasedActionClassifier,
    classify_action,
)


class TestPoseGeometry:
    """Tests for the PoseGeometry helper class."""

    def test_is_valid_with_high_confidence(self, standing_pose):
        """Keypoint with high confidence is valid."""
        geom = PoseGeometry(standing_pose, min_conf=0.3)
        assert geom.is_valid(0)  # nose has conf 0.95

    def test_is_valid_with_low_confidence(self):
        """Keypoint with low confidence is invalid."""
        kpts = np.zeros((17, 3))
        kpts[0] = [100, 100, 0.1]  # Low confidence

        geom = PoseGeometry(kpts, min_conf=0.3)
        assert not geom.is_valid(0)

    def test_get_point_returns_xy(self, standing_pose):
        """get_point returns (x, y) for valid keypoint."""
        geom = PoseGeometry(standing_pose)
        point = geom.get_point(0)  # nose

        assert point is not None
        assert len(point) == 2
        assert point[0] == pytest.approx(100)  # x
        assert point[1] == pytest.approx(20)   # y

    def test_get_point_returns_none_for_invalid(self):
        """get_point returns None for low-confidence keypoint."""
        kpts = np.zeros((17, 3))
        kpts[0] = [100, 100, 0.1]

        geom = PoseGeometry(kpts, min_conf=0.3)
        assert geom.get_point(0) is None

    def test_get_midpoint(self, standing_pose):
        """get_midpoint calculates center between two points."""
        geom = PoseGeometry(standing_pose)
        # Midpoint between left and right shoulder
        mid = geom.get_midpoint(5, 6)

        assert mid is not None
        # Left shoulder at 80, right at 120, mid should be 100
        assert mid[0] == pytest.approx(100)

    def test_get_distance(self, standing_pose):
        """get_distance calculates Euclidean distance."""
        geom = PoseGeometry(standing_pose)
        # Distance between left and right shoulder (horizontal)
        dist = geom.get_distance(5, 6)

        assert dist is not None
        # Left at 80, right at 120 -> distance = 40
        assert dist == pytest.approx(40.0)

    def test_get_angle(self, standing_pose):
        """get_angle calculates angle at middle point."""
        geom = PoseGeometry(standing_pose)
        # Angle at left knee (hip-knee-ankle should be ~180° for standing)
        angle = geom.get_angle(11, 13, 15)  # left hip, knee, ankle

        assert angle is not None
        assert angle > 150  # Should be nearly straight

    def test_get_skeleton_bbox(self, standing_pose):
        """get_skeleton_bbox returns bounding box of visible keypoints."""
        geom = PoseGeometry(standing_pose)
        bbox = geom.get_skeleton_bbox()

        assert bbox is not None
        min_x, min_y, max_x, max_y = bbox
        assert min_x < max_x
        assert min_y < max_y

    def test_get_bbox_aspect_ratio_standing(self, standing_pose):
        """Standing pose has aspect ratio < 1 (taller than wide)."""
        geom = PoseGeometry(standing_pose)
        ratio = geom.get_bbox_aspect_ratio()

        assert ratio is not None
        assert ratio < 1.0  # Taller than wide

    def test_get_bbox_aspect_ratio_lying(self, lying_pose):
        """Lying pose has aspect ratio > 1 (wider than tall)."""
        geom = PoseGeometry(lying_pose)
        ratio = geom.get_bbox_aspect_ratio()

        assert ratio is not None
        assert ratio > 1.0  # Wider than tall


class TestRuleBasedActionClassifier:
    """Tests for the RuleBasedActionClassifier."""

    @pytest.fixture
    def classifier(self):
        return RuleBasedActionClassifier()

    def test_classify_standing(self, classifier, standing_pose):
        """Standing pose is classified as 'standing'."""
        result = classifier.classify(standing_pose)
        assert result.action == ACTION_STANDING

    def test_classify_sitting(self, classifier, sitting_pose):
        """Sitting pose is classified as 'sitting'."""
        result = classifier.classify(sitting_pose)
        assert result.action == ACTION_SITTING

    def test_classify_lying(self, classifier, lying_pose):
        """Lying pose is classified as 'lying_down'."""
        result = classifier.classify(lying_pose)
        assert result.action == ACTION_LYING_DOWN

    def test_classify_phone_calling(self, classifier, phone_calling_pose):
        """Phone calling pose is classified as 'phone_calling'."""
        result = classifier.classify(phone_calling_pose)
        assert result.action == ACTION_PHONE_CALLING

    def test_classify_phone_texting(self, classifier, phone_texting_pose):
        """Phone texting pose is classified as 'phone_texting'."""
        result = classifier.classify(phone_texting_pose)
        assert result.action == ACTION_PHONE_TEXTING

    def test_classify_arms_raised(self, classifier, arms_raised_pose):
        """Arms raised pose is classified as 'arms_raised'."""
        result = classifier.classify(arms_raised_pose)
        assert result.action == ACTION_ARMS_RAISED

    def test_classify_walking(self, classifier, standing_pose):
        """Standing with velocity is classified as 'walking'."""
        velocity = np.array([50.0, 0.0])
        speed = np.linalg.norm(velocity)

        result = classifier.classify(standing_pose, velocity=velocity, speed=speed)
        assert result.action == ACTION_WALKING

    def test_classify_insufficient_keypoints(self, classifier):
        """Few visible keypoints returns 'unknown'."""
        kpts = np.zeros((17, 3))
        kpts[0] = [100, 100, 0.9]  # Only nose visible

        result = classifier.classify(kpts)
        assert result.action == ACTION_UNKNOWN


class TestClassifyAction:
    """Tests for the convenience classify_action function."""

    def test_classify_action_returns_tuple(self, standing_pose):
        """classify_action returns (action, confidence) tuple."""
        action, conf = classify_action(standing_pose)

        assert isinstance(action, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_classify_action_with_velocity(self, standing_pose):
        """classify_action handles velocity parameter."""
        action, _conf = classify_action(
            standing_pose,
            velocity=np.array([100.0, 0.0]),
            speed=100.0
        )

        assert action == ACTION_WALKING


class TestActionFilter:
    """Tests for the ActionFilter temporal smoothing."""

    def test_filter_requires_confirmation(self):
        """Action must be seen for confirm_frames to be accepted."""
        filt = ActionFilter(confirm_frames=3, use_frames=True)
        track_id = 1
        t = 0.0

        # First 2 frames: not confirmed yet
        for _ in range(2):
            action, _ = filt.update(track_id, "standing", 0.8, t)
            t += 0.033

        assert action == ACTION_UNKNOWN

        # Third frame: should confirm
        action, _ = filt.update(track_id, "standing", 0.8, t)
        assert action == ACTION_STANDING

    def test_filter_maintains_action_during_flicker(self):
        """Confirmed action is maintained during brief changes."""
        filt = ActionFilter(confirm_frames=3, use_frames=True)
        track_id = 1
        t = 0.0

        # Confirm standing (3 frames)
        for _ in range(3):
            filt.update(track_id, "standing", 0.8, t)
            t += 0.033

        # Single frame of sitting (flicker)
        action, _ = filt.update(track_id, "sitting", 0.7, t)
        t += 0.033

        # Should still be standing
        assert action == ACTION_STANDING

    def test_filter_transitions_after_confirmation(self):
        """Action changes after new action is confirmed."""
        filt = ActionFilter(confirm_frames=3, use_frames=True)
        track_id = 1
        t = 0.0

        # Confirm standing
        for _ in range(3):
            filt.update(track_id, "standing", 0.8, t)
            t += 0.033

        # Now consistently sitting for 3 frames
        for _ in range(3):
            action, _ = filt.update(track_id, "sitting", 0.7, t)
            t += 0.033

        assert action == ACTION_SITTING

    def test_filter_time_based_confirmation(self):
        """Time-based confirmation works correctly."""
        filt = ActionFilter(confirm_seconds=0.2, use_frames=False)
        track_id = 1

        # First update at t=0
        action1, _ = filt.update(track_id, "standing", 0.8, 0.0)
        assert action1 == ACTION_UNKNOWN

        # Update at t=0.1 (not enough time)
        action2, _ = filt.update(track_id, "standing", 0.8, 0.1)
        assert action2 == ACTION_UNKNOWN

        # Update at t=0.25 (enough time)
        action3, _ = filt.update(track_id, "standing", 0.8, 0.25)
        assert action3 == ACTION_STANDING

    def test_filter_independent_tracks(self):
        """Each track has independent filtering."""
        filt = ActionFilter(confirm_frames=2, use_frames=True)
        t = 0.0

        # Track 1: standing
        filt.update(1, "standing", 0.8, t)
        filt.update(1, "standing", 0.8, t + 0.033)

        # Track 2: sitting (not confirmed yet)
        action2, _ = filt.update(2, "sitting", 0.7, t)

        # Track 1 should be confirmed
        action1, _ = filt.update(1, "standing", 0.8, t + 0.066)

        assert action1 == ACTION_STANDING
        assert action2 == ACTION_UNKNOWN
