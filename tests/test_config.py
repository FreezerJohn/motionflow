"""
Unit tests for config.schema module.

Tests configuration loading and validation.
"""

from pathlib import Path

import pytest

from config.schema import (
    AppConfig,
    Door,
    Point,
    StreamConfig,
    Zone,
    load_config,
)


class TestPoint:
    """Tests for the Point model."""

    def test_point_creation(self):
        """Point can be created with x and y."""
        p = Point(x=0.5, y=0.5)
        assert p.x == 0.5
        assert p.y == 0.5

    def test_point_validation_range(self):
        """Point values must be between 0.0 and 1.0."""
        # Valid
        Point(x=0.0, y=0.0)
        Point(x=1.0, y=1.0)

        # Invalid - should raise
        with pytest.raises(ValueError):
            Point(x=-0.1, y=0.5)

        with pytest.raises(ValueError):
            Point(x=0.5, y=1.1)


class TestZone:
    """Tests for the Zone config model."""

    def test_zone_creation(self):
        """Zone can be created with name and points."""
        z = Zone(
            name="test",
            points=[Point(x=0.0, y=0.0), Point(x=1.0, y=0.0)],
        )

        assert z.name == "test"
        assert len(z.points) == 2

    def test_zone_default_triggers(self):
        """Zone has default triggers."""
        z = Zone(name="test", points=[])
        assert z.triggers == ["person"]


class TestDoor:
    """Tests for the Door config model."""

    def test_door_creation(self):
        """Door can be created with required fields."""
        d = Door(
            name="entrance",
            points=[
                Point(x=0.4, y=0.4),
                Point(x=0.6, y=0.4),
                Point(x=0.6, y=0.6),
                Point(x=0.4, y=0.6),
            ],
        )

        assert d.name == "entrance"
        assert len(d.points) == 4

    def test_door_default_normal_angle(self):
        """Door has default normal_angle of 0."""
        d = Door(name="test", points=[])
        assert d.normal_angle == 0.0

    def test_door_normal_angle_validation(self):
        """Door normal_angle must be 0-360."""
        # Valid
        Door(name="test", points=[], normal_angle=0.0)
        Door(name="test", points=[], normal_angle=359.9)

        # Invalid
        with pytest.raises(ValueError):
            Door(name="test", points=[], normal_angle=-1.0)

        with pytest.raises(ValueError):
            Door(name="test", points=[], normal_angle=360.0)

    def test_door_direction_tolerance_range(self):
        """Door direction_tolerance must be 0-90."""
        # Valid
        Door(name="test", points=[], direction_tolerance=0.0)
        Door(name="test", points=[], direction_tolerance=90.0)

        # Invalid
        with pytest.raises(ValueError):
            Door(name="test", points=[], direction_tolerance=-1.0)

        with pytest.raises(ValueError):
            Door(name="test", points=[], direction_tolerance=91.0)


class TestStreamConfig:
    """Tests for the StreamConfig model."""

    def test_stream_config_creation(self):
        """StreamConfig can be created with required fields."""
        sc = StreamConfig(
            name="living_room",
            rtsp_url="rtsp://192.168.1.100:8554/stream",
            mqtt_topic_suffix="/living_room",
        )

        assert sc.name == "living_room"
        assert sc.enabled is True  # Default

    def test_stream_config_zones(self):
        """StreamConfig can have zones."""
        sc = StreamConfig(
            name="test",
            rtsp_url="rtsp://test",
            mqtt_topic_suffix="/test",
            zones=[Zone(name="z1", points=[])],
        )

        assert len(sc.zones) == 1
        assert sc.zones[0].name == "z1"


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_from_file(self, test_config_dir):
        """Config can be loaded from YAML file."""
        config_path = test_config_dir / "settings.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_config(str(config_path))

        assert isinstance(config, AppConfig)

    def test_load_config_missing_file_returns_default(self):
        """Loading non-existent file returns default config."""
        # The current implementation returns a default config for missing files
        config = load_config("/nonexistent/path/config.yaml")

        # Should return a valid default AppConfig
        assert isinstance(config, AppConfig)

    def test_load_config_invalid_yaml(self, tmp_path):
        """Loading invalid YAML raises error."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # Could be yaml.YAMLError or validation error
            load_config(str(bad_file))
