"""
End-to-end system tests for MotionFlow.

These tests run the full inference pipeline on recorded video files and
verify that the expected events (zone enter/exit, door crossings, etc.)
are generated within the expected time windows.

Requirements:
- Axelera Voyager SDK installed (runs on target device only)
- Test footage in test_footage/ directory
- Test config in config/settings_test.yaml with expected_events defined

Run with:
    pytest tests/test_system.py -m system -v

These tests are automatically skipped when the SDK is not available,
so running `pytest` on a dev machine is safe.

Architecture note:
    The inference pipeline runs as a **subprocess** (tests/run_inference.py)
    rather than in-process, because the SDK's GStreamer event loop and signal
    handlers are incompatible with pytest's runtime.  The subprocess writes
    events as JSON lines to a temp file which the tests then load and verify.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from config.schema import AppConfig, load_config
from core.events import EventType

logger = logging.getLogger(__name__)

# Skip entire module if SDK not available
try:
    from axelera.app import create_inference_stream  # noqa: F401

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

pytestmark = [
    pytest.mark.system,
    pytest.mark.skipif(not HAS_SDK, reason="Axelera SDK not available"),
]

PROJECT_ROOT = Path(__file__).parent.parent
TEST_CONFIG = PROJECT_ROOT / "config" / "settings_test.yaml"
TEST_FOOTAGE = PROJECT_ROOT / "test_footage"
RUN_INFERENCE_SCRIPT = Path(__file__).parent / "run_inference.py"

# Timeout for inference subprocess (seconds).  The flur3 video is ~140s
# but inference is faster than real-time on the target hardware.
INFERENCE_TIMEOUT = 300


@dataclass
class CollectedEvent:
    """Lightweight event loaded from the JSONL output."""

    type: EventType
    video_time: float
    stream: str
    zone: str | None
    detection_id: int | None
    raw: dict

    @classmethod
    def from_json(cls, d: dict) -> "CollectedEvent":
        # detection_id is nested in person.id in MotionEvent.to_dict()
        person = d.get("person", {})
        return cls(
            type=EventType[d["event"].upper()],
            video_time=d.get("video_time", 0.0),
            stream=d.get("stream", ""),
            zone=d.get("zone"),
            detection_id=person.get("id") if person else None,
            raw=d,
        )


def _load_events(path: Path) -> list[CollectedEvent]:
    """Load events from a JSONL file written by run_inference.py."""
    events: list[CollectedEvent] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(CollectedEvent.from_json(json.loads(line)))
    return events


def _event_type_from_string(s: str) -> EventType:
    """Convert expected_events string like 'door_exit' to EventType."""
    mapping = {
        "track_new": EventType.TRACK_NEW,
        "track_lost": EventType.TRACK_LOST,
        "zone_enter": EventType.ZONE_ENTER,
        "zone_exit": EventType.ZONE_EXIT,
        "door_enter": EventType.DOOR_ENTER,
        "door_exit": EventType.DOOR_EXIT,
        "action_change": EventType.ACTION_CHANGE,
    }
    if s not in mapping:
        raise ValueError(f"Unknown event type '{s}', expected one of {list(mapping.keys())}")
    return mapping[s]


def _match_expected_events(
    actual_events: list[CollectedEvent],
    expected_events: list,
) -> tuple[list[str], list[str]]:
    """
    Match actual events against expected events.

    Returns (matches, failures) where each is a list of human-readable strings.

    Expected events (from settings_test.yaml) have:
    - event_type: str
    - timestamp_start / timestamp_end: float (seconds into the video)
    - zone_name: optional str
    - track_id: optional int
    """
    matches = []
    failures = []

    for exp in expected_events:
        exp_type = _event_type_from_string(exp.event_type)
        t_start = exp.timestamp_start
        t_end = exp.timestamp_end
        zone_name = exp.zone_name
        track_id = exp.track_id

        # Find matching actual events within the time window
        candidates = []
        for evt in actual_events:
            if evt.type != exp_type:
                continue
            if evt.video_time < t_start or evt.video_time > t_end:
                continue
            if zone_name and evt.zone != zone_name:
                continue
            if track_id is not None and evt.detection_id != track_id:
                continue
            candidates.append(evt)

        desc = (
            f"{exp.event_type} zone={zone_name or 'any'} "
            f"t=[{t_start:.1f}-{t_end:.1f}s]"
        )

        if candidates:
            best = candidates[0]
            matches.append(f"OK: {desc} -> matched at t={best.video_time:.2f}s")
        else:
            failures.append(f"MISSING: {desc}")

    return matches, failures


def _events_summary(events: list[CollectedEvent]) -> str:
    """Return a human-readable summary of collected events."""
    lines = []
    for e in sorted(events, key=lambda x: x.video_time):
        zone = e.zone or ""
        lines.append(
            f"  t={e.video_time:7.2f}s  {e.type.name.lower():<16} zone={zone}"
        )
    return "\n".join(lines)


def _run_inference(config_path: str, output_path: str) -> None:
    """
    Run the inference pipeline as a subprocess.

    Raises on non-zero exit or timeout.
    """
    cmd = [
        sys.executable,
        str(RUN_INFERENCE_SCRIPT),
        "--config", config_path,
        "--output", output_path,
    ]
    logger.info(f"Running inference subprocess: {' '.join(cmd)}")

    # Ensure the project root is on PYTHONPATH so 'config' and 'core' resolve
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = project_root_str + (os.pathsep + pythonpath if pythonpath else "")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=INFERENCE_TIMEOUT,
        capture_output=True,
        text=True,
    )

    # Log subprocess output for debugging
    if result.stdout:
        for line in result.stdout.splitlines()[-20:]:
            logger.info(f"[subprocess stdout] {line}")
    if result.stderr:
        for line in result.stderr.splitlines()[-30:]:
            logger.info(f"[subprocess stderr] {line}")

    if result.returncode != 0:
        pytest.fail(
            f"Inference subprocess failed (exit code {result.returncode}).\n"
            f"stderr (last 30 lines):\n"
            + "\n".join(result.stderr.splitlines()[-30:])
        )


@pytest.fixture(scope="module")
def test_config() -> AppConfig:
    """Load the test configuration."""
    if not TEST_CONFIG.exists():
        pytest.skip(f"Test config not found: {TEST_CONFIG}")
    return load_config(str(TEST_CONFIG))


class TestSystemFlur3:
    """
    End-to-end test using the flur3.mp4 hallway recording.

    This recording has a person walking through a hallway with two doors
    ('bad' and 'schlafzimmer'). Expected door crossing events are defined
    in config/settings_test.yaml.
    """

    @pytest.fixture(scope="class")
    def run_result(self, test_config):
        """
        Run the full inference pipeline on flur3.mp4 and collect events.

        This fixture is class-scoped so the expensive inference run
        happens only once, and individual test methods verify different
        aspects of the results.

        The pipeline runs as a **subprocess** to avoid GStreamer/signal
        conflicts with pytest.
        """
        # Find the flur3 stream config
        stream_cfg = None
        for s in test_config.streams:
            if s.name == "flur3" and s.enabled:
                stream_cfg = s
                break

        if stream_cfg is None:
            pytest.skip("flur3 stream not configured or not enabled")

        video_path = TEST_FOOTAGE / "flur3.mp4"
        if not video_path.exists():
            pytest.skip(f"Test footage not found: {video_path}")

        # Run inference in a subprocess, writing events to a temp file
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            output_path = tf.name

        _run_inference(str(TEST_CONFIG), output_path)

        # Load and parse collected events
        events = _load_events(Path(output_path))
        logger.info(f"System test: {len(events)} events collected")
        logger.info(f"Event summary:\n{_events_summary(events)}")

        # Clean up temp file
        Path(output_path).unlink(missing_ok=True)

        return {
            "events": events,
            "stream_config": stream_cfg,
        }

    def test_events_collected(self, run_result):
        """Verify that the pipeline produced events at all."""
        events = run_result["events"]
        assert len(events) > 0, "No events were generated"

    def test_track_events_generated(self, run_result):
        """Verify that track new/lost events were generated."""
        events = run_result["events"]
        track_new = [e for e in events if e.type == EventType.TRACK_NEW]
        track_lost = [e for e in events if e.type == EventType.TRACK_LOST]

        assert len(track_new) > 0, "No TRACK_NEW events"
        assert len(track_lost) > 0, "No TRACK_LOST events"

    def test_door_events_match_expected(self, run_result):
        """
        Verify that door crossing events match the expected events
        defined in settings_test.yaml.
        """
        events = run_result["events"]
        stream_cfg = run_result["stream_config"]

        expected = stream_cfg.expected_events
        if not expected:
            pytest.skip("No expected_events defined for flur3")

        matches, failures = _match_expected_events(events, expected)

        # Log all matches for debugging
        for m in matches:
            logger.info(m)
        for f in failures:
            logger.warning(f)

        # Also log all actual door events for reference
        door_events = [
            e for e in events
            if e.type in (EventType.DOOR_ENTER, EventType.DOOR_EXIT)
        ]
        logger.info(f"Actual door events ({len(door_events)}):")
        for e in door_events:
            logger.info(f"  t={e.video_time:.2f}s {e.type.name.lower()} zone={e.zone}")

        assert len(failures) == 0, (
            f"{len(failures)} expected events not matched:\n"
            + "\n".join(failures)
            + f"\n\nAll actual events:\n{_events_summary(events)}"
        )

    def test_no_unexpected_door_events(self, run_result):
        """
        Verify there are no spurious door events outside expected windows.

        This is a softer check -- we log unexpected events but allow some
        tolerance since detection can vary slightly between runs.
        """
        events = run_result["events"]
        stream_cfg = run_result["stream_config"]

        door_events = [
            e for e in events
            if e.type in (EventType.DOOR_ENTER, EventType.DOOR_EXIT)
        ]
        expected = stream_cfg.expected_events

        unexpected = []
        for evt in door_events:
            matched = False
            for exp in expected:
                exp_type = _event_type_from_string(exp.event_type)
                if (
                    evt.type == exp_type
                    and (exp.zone_name is None or evt.zone == exp.zone_name)
                    and exp.timestamp_start <= evt.video_time <= exp.timestamp_end
                ):
                    matched = True
                    break
            if not matched:
                unexpected.append(
                    f"  t={evt.video_time:.2f}s {evt.type.name.lower()} zone={evt.zone}"
                )

        if unexpected:
            logger.warning(
                f"{len(unexpected)} unexpected door events:\n"
                + "\n".join(unexpected)
            )

        # Soft assertion: allow up to 2 unexpected events (detection noise)
        assert len(unexpected) <= 2, (
            f"Too many unexpected door events ({len(unexpected)}):\n"
            + "\n".join(unexpected)
        )
