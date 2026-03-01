#!/usr/bin/env python3
"""
MotionFlow - Pose-based action detection

Exit codes:
- 0: Normal exit
- 1: Fatal error
- 42: Restart requested (via IPC from WebUI)
"""
import argparse
import logging
import os
import sys

from config.schema import load_config
from core.engine import MotionFlowEngine

# Exit code to signal wrapper script to restart
EXIT_CODE_RESTART = 42

# Create our own handler for MotionFlow loggers only
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure only MotionFlow loggers - don't touch root or SDK loggers
for name in ['core', 'core.engine', 'events', '__main__']:
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(_handler)
    _logger.propagate = False  # Don't propagate to root - SDK has its own root handler

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MotionFlow - Pose-based action detection")
    parser.add_argument(
        "-c", "--config",
        default="config/settings.yaml",
        help="Path to configuration file (default: config/settings.yaml)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable local OpenCV visualization window (for local debugging only, not for systemd)"
    )
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded config: {args.config} ({len(config.streams)} streams)")

        # Initialize and run engine (pass config_path for hot-reload support)
        engine = MotionFlowEngine(config, config_path=args.config, visualize=args.visualize)
        engine.run()
        
        # Check if restart was requested - exit with special code for wrapper script
        if engine._multi_stream_processor and engine._multi_stream_processor.restart_requested:
            logger.info("🔃 Exiting for restart (exit code 42)...")
            os._exit(EXIT_CODE_RESTART)  # Force exit - SDK threads may block sys.exit()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except KeyError as e:
        # Pipeline not found in SDK - engine already logged detailed instructions
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
