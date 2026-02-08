#!/bin/bash
# MotionFlow startup script for Orange Pi 5 Plus
# This script activates the Axelera SDK environment and runs MotionFlow
# 
# Features:
# - Auto-restart on exit code 42 (config-triggered restart)
# - Clean shutdown on Ctrl+C

cd ~/motionflow

# Activate Axelera SDK environment
source ~/voyager-sdk/venv/bin/activate

# Exit code 42 means "restart requested" (usually from webUI)
EXIT_CODE_RESTART=42

echo "Starting MotionFlow..."
echo "  Hot-reload: POST /api/reload (zones, doors, debounce)"
echo "  Full restart: POST /api/restart (streams, model)"
echo ""

while true; do
    # Run MotionFlow with target config
    AXELERA_CONFIGURE_BOARD=,30 python3 main.py --config config/settings.yaml --visualize
    exit_code=$?
    
    if [ $exit_code -eq $EXIT_CODE_RESTART ]; then
        echo ""
        echo "Restart requested, restarting in 2 seconds..."
        sleep 2
    else
        echo "MotionFlow exited with code $exit_code"
        break
    fi
done

# Deactivate on exit
deactivate
