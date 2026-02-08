#!/usr/bin/env python3
"""
Helper script for system tests: runs the inference pipeline and writes
all events as JSON lines to a file.

This runs as a separate process to avoid GStreamer/signal-handler conflicts
with pytest.  The system tests invoke this via subprocess and then read
the resulting JSON event log.

Usage:
    python tests/run_inference.py --config config/settings_test.yaml --output /tmp/events.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time

# Ensure the project root is on sys.path so 'config' and 'core' imports work
# when this script is invoked directly (not via python -m).
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Minimal logging so SDK output goes to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run inference and dump events as JSON lines")
    parser.add_argument("-c", "--config", required=True, help="Path to config YAML")
    parser.add_argument("-o", "--output", required=True, help="Path to write JSONL event log")
    args = parser.parse_args()

    from config.schema import load_config
    from core.engine import MotionFlowEngine
    from core.events import EventListener, MotionEvent

    class JSONFileListener(EventListener):
        """Writes every event as a JSON line to a file."""

        def __init__(self, path: str):
            self._path = path
            self._fh = open(path, "w")
            self._start_time: float | None = None

        def on_event(self, event: MotionEvent) -> None:
            if self._start_time is None:
                self._start_time = time.time()
            d = event.to_dict()
            # Add video-relative timestamp for easier test matching
            d["video_time"] = event.timestamp - self._start_time
            self._fh.write(json.dumps(d) + "\n")
            self._fh.flush()

        def close(self):
            self._fh.close()

    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config} ({len(config.streams)} streams)")

    listener = JSONFileListener(args.output)

    engine = MotionFlowEngine(config=config, config_path=args.config, visualize=False)

    # Attach JSON listener to all event managers
    for em in engine.event_managers:
        em.add_listener(listener)

    start = time.time()
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        listener.close()

    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s — events written to {args.output}")


if __name__ == "__main__":
    main()
