#!/bin/bash
# Install MotionFlow systemd services on Orange Pi 5 Plus
#
# Usage (on target device):
#   cd ~/motionflow
#   ./systemd/install.sh
#
# This installs two services:
#   motionflow.service       - Main detection engine (auto-restarts on exit code 42)
#   motionflow-webui.service - Web UI for config & live preview
#
# Management:
#   sudo systemctl start motionflow motionflow-webui
#   sudo systemctl stop motionflow motionflow-webui
#   sudo systemctl restart motionflow
#   sudo systemctl status motionflow motionflow-webui
#   journalctl -u motionflow -f           # Follow engine logs
#   journalctl -u motionflow-webui -f     # Follow web UI logs
#   journalctl -u motionflow -u motionflow-webui -f  # Both

set -e

# Prevent running as root (sudo is used internally where needed)
if [ "$(id -u)" -eq 0 ]; then
    echo "Error: Do not run this script as root. Use: ./systemd/install.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_USER=$(whoami)
TARGET_WORKDIR=$(pwd)

# Verify we are in the project root roughly
if [ ! -f "$TARGET_WORKDIR/main.py" ]; then
    echo "Error: Please run this script from the project root directory."
    echo "Usage: ./systemd/install.sh"
    exit 1
fi

echo "=== Installing MotionFlow systemd services ==="
echo "Target User:      $TARGET_USER"
echo "Target Directory: $TARGET_WORKDIR"

# Function to process and install a service file
install_service() {
    local service_name=$1
    local src_file="$SCRIPT_DIR/$service_name"
    local tmp_file="/tmp/$service_name"

    echo "Processing $service_name..."
    
    # Replace placeholders
    sed -e "s|__USER__|$TARGET_USER|g" \
        -e "s|__WORKDIR__|$TARGET_WORKDIR|g" \
        "$src_file" > "$tmp_file"

    # Install
    echo "Installing to /etc/systemd/system/..."
    sudo cp "$tmp_file" "/etc/systemd/system/$service_name"
    rm "$tmp_file"
}

# Install services
install_service "motionflow.service"
install_service "motionflow-webui.service"

# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable motionflow.service
sudo systemctl enable motionflow-webui.service

echo ""
echo "Services installed and enabled."
echo ""
echo "Start both:    sudo systemctl start motionflow motionflow-webui"
echo "Stop both:     sudo systemctl stop motionflow motionflow-webui"
echo "View logs:     journalctl -u motionflow -u motionflow-webui -f"
echo ""
echo "Local display: To see visualization on the device's monitor,"
echo "  1. Stop the service: sudo systemctl stop motionflow"
echo "  2. Run: xhost +local:  (once after desktop login)"
echo "  3. Run manually: python3 main.py --config config/settings_target.yaml --visualize"
echo ""
echo "The services will start automatically on boot."
