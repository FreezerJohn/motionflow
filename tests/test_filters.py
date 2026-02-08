"""
Unit tests for core.filters module.

Tests the PoseFilter (One Euro Filter) for keypoint smoothing.
"""

import numpy as np
import pytest

from core.filters import PoseFilter


class TestPoseFilter:
    """Tests for the PoseFilter class."""

    def test_filter_creation(self):
        """PoseFilter can be created with default parameters."""
        filt = PoseFilter()
        assert filt is not None
        assert filt.freq == 30.0

    def test_filter_custom_parameters(self):
        """PoseFilter accepts custom parameters."""
        filt = PoseFilter(
            min_cutoff=0.5,
            beta=0.1,
            d_cutoff=0.5,
            freq=60.0,
        )

        assert filt.min_cutoff == 0.5
        assert filt.beta == 0.1
        assert filt.freq == 60.0

    def test_first_call_returns_input(self):
        """First filter call returns the input unchanged."""
        filt = PoseFilter()

        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[0] = [100, 100, 0.9]

        result = filt.filter(kpts, timestamp=0.0)

        # First call initializes, should return input
        assert result[0, 0] == pytest.approx(100)
        assert result[0, 1] == pytest.approx(100)

    def test_filter_smooths_jitter(self):
        """Filter smooths out small jittery movements."""
        filt = PoseFilter(min_cutoff=1.0, beta=0.0)

        # Create keypoints with small jitter
        base_kpts = np.zeros((17, 3), dtype=np.float32)
        base_kpts[0] = [100, 100, 0.9]

        # Initialize
        filt.filter(base_kpts.copy(), timestamp=0.0)

        # Add jitter and filter
        jittery = base_kpts.copy()
        jittery[0, 0] = 105  # Jump from 100 to 105

        result = filt.filter(jittery, timestamp=0.033)

        # Filtered value should be between original and jittery
        # (smoothed, not jumping immediately to new value)
        assert 100 < result[0, 0] < 105

    def test_filter_tracks_real_movement(self):
        """Filter follows consistent movement direction."""
        filt = PoseFilter(min_cutoff=0.1, beta=0.5)  # Higher beta = more responsive

        # Start at 100
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[0] = [100, 100, 0.9]

        t = 0.0
        filt.filter(kpts.copy(), timestamp=t)

        # Move consistently in one direction
        for i in range(10):
            t += 0.033
            kpts[0, 0] = 100 + (i + 1) * 10  # 110, 120, 130, ...
            result = filt.filter(kpts.copy(), timestamp=t)

        # After consistent movement, filtered value should be close to actual
        # (within 20% of actual position)
        actual = kpts[0, 0]  # Should be 200
        assert result[0, 0] > actual * 0.8

    def test_filter_preserves_confidence(self):
        """Filter does not modify confidence values."""
        filt = PoseFilter()

        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[0] = [100, 100, 0.95]
        kpts[1] = [90, 90, 0.3]

        filt.filter(kpts.copy(), timestamp=0.0)
        result = filt.filter(kpts.copy(), timestamp=0.033)

        # Confidence should be unchanged
        assert result[0, 2] == pytest.approx(0.95)
        assert result[1, 2] == pytest.approx(0.3)

    def test_filter_all_keypoints(self):
        """Filter processes all 17 keypoints."""
        filt = PoseFilter()

        kpts = np.random.rand(17, 3).astype(np.float32)
        kpts[:, :2] *= 100  # Scale x, y to reasonable range

        result = filt.filter(kpts.copy(), timestamp=0.0)

        assert result.shape == (17, 3)

    def test_filter_uses_current_time_if_no_timestamp(self):
        """Filter uses time.time() if no timestamp provided."""
        import time

        filt = PoseFilter()

        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[0] = [100, 100, 0.9]

        before = time.time()
        filt.filter(kpts.copy())  # No timestamp
        after = time.time()

        # Should have recorded a timestamp
        assert filt.last_timestamp is not None
        assert before <= filt.last_timestamp <= after
