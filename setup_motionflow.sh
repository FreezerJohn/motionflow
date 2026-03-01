#!/bin/bash
#
# MotionFlow Setup
#
# First-run setup for Orange Pi with Axelera Metis AIPU.
#
#   1. Locates the Axelera Voyager SDK
#   2. Installs Python dependencies
#   3. Builds the AIPU pipeline (~1 hour on first run)
#   4. Installs systemd services or starts standalone
#
# Usage:
#   ./setup_motionflow.sh              # Interactive setup
#   ./setup_motionflow.sh --check      # Status check only
#   ./setup_motionflow.sh --uninstall  # Remove services & stop
#

set -euo pipefail

# ---- Formatting -------------------------------------------------------------
BOLD='\033[1m'
DIM='\033[2m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()     { echo -e "  $*"; }
ok()      { echo -e "  ${GREEN}✓${NC} $*"; }
warn()    { echo -e "  ${YELLOW}⚠${NC}  $*"; }
fail()    { echo -e "  ${RED}✗${NC} $*"; }
step()    { echo -e "\n${BOLD}[$1/4]${NC} ${BOLD}$2${NC}\n"; }
banner()  { echo -e "\n${CYAN}${BOLD}  MotionFlow Setup${NC}\n"; }

ask_yes_no() {
    local prompt="$1" default="${2:-y}"
    local hint="[Y/n]"; [[ "$default" != "y" ]] && hint="[y/N]"
    while true; do
        read -rp "  $prompt $hint " answer
        answer="${answer:-$default}"
        case "${answer,,}" in
            y|yes) return 0 ;; n|no) return 1 ;;
            *) log "Please answer y or n." ;;
        esac
    done
}

get_local_ip() {
    hostname -I 2>/dev/null | awk '{print $1}'
}

show_webui_url() {
    local ip
    ip=$(get_local_ip)
    if [[ -n "$ip" ]]; then
        echo ""
        log "${BOLD}→ Web UI: http://${ip}:5000${NC}"
        echo ""
    fi
}

# ---- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MOTIONFLOW_DIR="$SCRIPT_DIR"
PIPELINE_NAME="yolo11lpose-coco-tracker"
PIPELINE_YAML="config/${PIPELINE_NAME}.yaml"
CONFIG="config/settings.yaml"
REQUIREMENTS="requirements.txt"

ENGINE_SERVICE="motionflow-engine"
WEBUI_SERVICE="motionflow-webui"

# ---- Arguments --------------------------------------------------------------
MODE="setup"

for arg in "$@"; do
    case "$arg" in
        --check)     MODE="check" ;;
        --uninstall) MODE="uninstall" ;;
        --help|-h)
            echo "Usage: $0 [--check] [--uninstall]"
            echo "  --check      Status check, no changes"
            echo "  --uninstall  Remove services and stop processes"
            exit 0
            ;;
        *) fail "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---- Uninstall --------------------------------------------------------------
if [[ "$MODE" == "uninstall" ]]; then
    echo -e "\n${BOLD}Uninstalling MotionFlow${NC}\n"

    for svc in "$ENGINE_SERVICE" "$WEBUI_SERVICE"; do
        if systemctl list-unit-files "${svc}.service" &>/dev/null && \
           systemctl list-unit-files "${svc}.service" | grep -q "$svc"; then
            log "Stopping ${svc}..."
            sudo systemctl stop "${svc}.service" 2>/dev/null || true
            sudo systemctl disable "${svc}.service" 2>/dev/null || true
            sudo rm -f "/etc/systemd/system/${svc}.service"
            ok "Removed ${svc}.service"
        fi
    done
    sudo systemctl daemon-reload 2>/dev/null || true

    pgrep -af "python.*main\.py" &>/dev/null && {
        pkill -f "python.*main\.py" 2>/dev/null || true; ok "Engine stopped."
    }
    pgrep -af "python.*web_ui/app\.py" &>/dev/null && {
        pkill -f "python.*web_ui/app\.py" 2>/dev/null || true; ok "Web UI stopped."
    }

    echo ""
    ok "Done. Config and build artifacts are preserved."
    exit 0
fi

# =============================================================================
banner

# ---- Step 1: Voyager SDK ----------------------------------------------------
step 1 "Axelera Voyager SDK"

SDK_DIR=""
VENV_ACTIVATE=""

for candidate in "$HOME/voyager-sdk" "$HOME/axelera/voyager-sdk" \
                 "/opt/voyager-sdk" "/opt/axelera/voyager-sdk"; do
    if [[ -f "$candidate/venv/bin/activate" ]]; then
        SDK_DIR="$candidate"
        VENV_ACTIVATE="$candidate/venv/bin/activate"
        break
    fi
done

if [[ -z "$SDK_DIR" ]]; then
    warn "Could not auto-detect Voyager SDK."
    read -rp "  Path to Voyager SDK: " SDK_DIR
    if [[ -f "$SDK_DIR/venv/bin/activate" ]]; then
        VENV_ACTIVATE="$SDK_DIR/venv/bin/activate"
    else
        fail "No venv at $SDK_DIR/venv/bin/activate"
        fail "Install the Voyager SDK first. See doc/README_orange_pi_setup.md"
        exit 1
    fi
fi

ok "SDK found: $SDK_DIR"

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

if python3 -c "import axelera" 2>/dev/null; then
    ok "axelera module OK"
else
    fail "Cannot import axelera — is the SDK installed correctly?"
    exit 1
fi

# ---- Step 2: Python Dependencies --------------------------------------------
step 2 "Python Dependencies"

MISSING=()
while IFS= read -r line; do
    line=$(echo "$line" | sed 's/#.*//' | xargs)
    [[ -z "$line" ]] && continue
    pkg=$(echo "$line" | sed 's/[>=<].*//' | xargs)
    case "$pkg" in
        pyyaml)        imp="yaml" ;;
        opencv-python) imp="cv2" ;;
        paho-mqtt)     imp="paho.mqtt" ;;
        flask-sock)    imp="flask_sock" ;;
        *)             imp="$pkg" ;;
    esac
    python3 -c "import $imp" 2>/dev/null || MISSING+=("$line")
done < "$REQUIREMENTS"

if [[ ${#MISSING[@]} -eq 0 ]]; then
    ok "All packages installed."
else
    warn "Missing: ${MISSING[*]}"
    if [[ "$MODE" == "check" ]]; then
        log "Run without --check to install."
    elif ask_yes_no "Install into SDK venv?" "y"; then
        pip install -r "$REQUIREMENTS" 2>&1 | tail -5
        ok "Installed."
    else
        warn "Skipped — some features may not work."
    fi
fi

# ---- Step 3: Pipeline -------------------------------------------------------
step 3 "AIPU Pipeline"

pipeline_available() {
    # Check for the compiled build artifacts (the .axnet file)
    local build_dir="$SDK_DIR/build/$PIPELINE_NAME"
    [[ -d "$build_dir" ]] && ls "$build_dir"/*.axnet &>/dev/null
}

register_pipeline_yaml() {
    cp "$MOTIONFLOW_DIR/$PIPELINE_YAML" "$SDK_DIR/ax_models/${PIPELINE_NAME}.yaml"
}

build_pipeline() {
    register_pipeline_yaml
    echo ""
    log "Building ${BOLD}$PIPELINE_NAME${NC} — this takes about ${BOLD}1 hour${NC}."
    log "${DIM}Compiles the ONNX model for the Metis AIPU chip.${NC}"
    echo ""
    cd "$SDK_DIR"
    if python3 deploy.py "$PIPELINE_NAME" 2>&1; then
        echo ""
        ok "Pipeline built successfully!"
        cd "$MOTIONFLOW_DIR"
        return 0
    else
        echo ""
        fail "Build failed. Check the output above."
        cd "$MOTIONFLOW_DIR"
        return 1
    fi
}

PIPELINE_READY=false

if pipeline_available; then
    ok "Pipeline ${BOLD}$PIPELINE_NAME${NC} is ready."
    PIPELINE_READY=true
else
    warn "Pipeline ${BOLD}$PIPELINE_NAME${NC} is not built yet."
    log "The first build compiles the model for the Metis AIPU chip."
    log "This is a one-time process that takes ${BOLD}~1 hour${NC}."
    echo ""

    if [[ "$MODE" == "check" ]]; then
        log "Run without --check to build."
    elif ask_yes_no "Build the pipeline now?" "y"; then
        build_pipeline && PIPELINE_READY=true
    else
        warn "Skipped — the engine won't start without a built pipeline."
        echo ""
        log "Build later:"
        log "  ${DIM}source $VENV_ACTIVATE${NC}"
        log "  ${DIM}cd $SDK_DIR && python3 deploy.py $PIPELINE_NAME${NC}"
    fi
fi

# ---- Check-only summary -----------------------------------------------------
if [[ "$MODE" == "check" ]]; then
    echo -e "\n${BOLD}Status${NC}\n"
    echo -e "  SDK:          ${GREEN}OK${NC}  ${DIM}($SDK_DIR)${NC}"
    if [[ ${#MISSING[@]} -eq 0 ]]; then
        echo -e "  Dependencies: ${GREEN}OK${NC}"
    else
        echo -e "  Dependencies: ${YELLOW}MISSING${NC}  ${DIM}(${MISSING[*]})${NC}"
    fi
    if $PIPELINE_READY; then
        echo -e "  Pipeline:     ${GREEN}OK${NC}"
    else
        echo -e "  Pipeline:     ${RED}NOT BUILT${NC}"
    fi
    if systemctl is-active --quiet "${ENGINE_SERVICE}.service" 2>/dev/null; then
        echo -e "  Engine:       ${GREEN}systemd${NC}"
    elif pgrep -af "python.*main\.py" &>/dev/null; then
        echo -e "  Engine:       ${GREEN}standalone${NC}"
    else
        echo -e "  Engine:       ${DIM}not running${NC}"
    fi
    if systemctl is-active --quiet "${WEBUI_SERVICE}.service" 2>/dev/null; then
        echo -e "  Web UI:       ${GREEN}systemd${NC}"
    elif pgrep -af "python.*web_ui/app\.py" &>/dev/null; then
        echo -e "  Web UI:       ${GREEN}standalone${NC}"
    else
        echo -e "  Web UI:       ${DIM}not running${NC}"
    fi
    show_webui_url
    exit 0
fi

# ---- Step 4: Startup Mode ---------------------------------------------------
step 4 "Startup"

EXISTING_MODE="none"
if systemctl is-enabled --quiet "${ENGINE_SERVICE}.service" 2>/dev/null; then
    EXISTING_MODE="systemd"
elif pgrep -af "python.*main\.py" &>/dev/null; then
    EXISTING_MODE="standalone"
fi

if [[ "$EXISTING_MODE" != "none" ]]; then
    log "Currently running as: ${BOLD}${EXISTING_MODE}${NC}"
    echo ""
fi

echo -e "  1) ${BOLD}Standalone${NC}   Start in this terminal"
echo -e "  2) ${BOLD}Systemd${NC}      Auto-start on boot (recommended)"
echo -e "  3) ${BOLD}Exit${NC}         Configure later"
echo ""

read -rp "  Choose [1/2/3]: " STARTUP_CHOICE
STARTUP_CHOICE="${STARTUP_CHOICE:-2}"

# Clean up old mode when switching
if [[ "$EXISTING_MODE" == "systemd" && "$STARTUP_CHOICE" == "1" ]]; then
    log "Removing existing systemd services..."
    for svc in "$ENGINE_SERVICE" "$WEBUI_SERVICE"; do
        sudo systemctl stop "${svc}.service" 2>/dev/null || true
        sudo systemctl disable "${svc}.service" 2>/dev/null || true
        sudo rm -f "/etc/systemd/system/${svc}.service"
    done
    sudo systemctl daemon-reload
    ok "Services removed."
fi
if [[ "$EXISTING_MODE" == "standalone" && "$STARTUP_CHOICE" == "2" ]]; then
    log "Stopping standalone processes..."
    pkill -f "python.*main\.py" 2>/dev/null || true
    pkill -f "python.*web_ui/app\.py" 2>/dev/null || true
    sleep 1
    ok "Stopped."
fi

case "$STARTUP_CHOICE" in
    2)
        CURRENT_USER=$(whoami)

        sudo tee /etc/systemd/system/${ENGINE_SERVICE}.service > /dev/null <<HEREDOC_ENGINE
[Unit]
Description=MotionFlow Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$MOTIONFLOW_DIR
Environment=AXELERA_CONFIGURE_BOARD=,30
Environment=AXELERA_YAML_PATH=$MOTIONFLOW_DIR/config
ExecStart=/bin/bash -c 'source $VENV_ACTIVATE && exec python3 main.py --config $CONFIG'
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
HEREDOC_ENGINE

        sudo tee /etc/systemd/system/${WEBUI_SERVICE}.service > /dev/null <<HEREDOC_WEBUI
[Unit]
Description=MotionFlow Web UI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$MOTIONFLOW_DIR
Environment=PYTHONPATH=$MOTIONFLOW_DIR
ExecStart=/bin/bash -c 'source $VENV_ACTIVATE && exec python3 web_ui/app.py --config $CONFIG'
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
HEREDOC_WEBUI

        sudo systemctl daemon-reload
        sudo systemctl enable "${ENGINE_SERVICE}.service" "${WEBUI_SERVICE}.service" --quiet

        sudo systemctl start "${WEBUI_SERVICE}.service"
        ok "Web UI service started."

        if $PIPELINE_READY; then
            sudo systemctl start "${ENGINE_SERVICE}.service"
            ok "Engine service started."
        else
            warn "Engine not started — build the pipeline first, then:"
            log "  ${DIM}sudo systemctl start ${ENGINE_SERVICE}${NC}"
        fi

        show_webui_url

        log "${DIM}Useful commands:${NC}"
        log "  ${DIM}sudo systemctl status  $ENGINE_SERVICE${NC}"
        log "  ${DIM}sudo journalctl -u $ENGINE_SERVICE -f${NC}"
        log "  ${DIM}sudo systemctl restart $ENGINE_SERVICE${NC}"
        ;;

    1)
        export PYTHONPATH="$MOTIONFLOW_DIR:${PYTHONPATH:-}"
        python3 "$MOTIONFLOW_DIR/web_ui/app.py" --config "$CONFIG" &
        WEBUI_PID=$!
        ok "Web UI started (PID: $WEBUI_PID)"
        show_webui_url

        if $PIPELINE_READY; then
            log "Starting engine... (Ctrl+C to stop)"
            export AXELERA_CONFIGURE_BOARD=",30"
            export AXELERA_YAML_PATH="$MOTIONFLOW_DIR/config"

            EXIT_CODE=0
            python3 "$MOTIONFLOW_DIR/main.py" --config "$CONFIG" || EXIT_CODE=$?

            while [[ $EXIT_CODE -eq 42 ]]; do
                log "Restarting engine..."
                sleep 2
                EXIT_CODE=0
                python3 "$MOTIONFLOW_DIR/main.py" --config "$CONFIG" || EXIT_CODE=$?
            done
        else
            warn "Engine not started — build the pipeline first."
            log "Press Ctrl+C to stop the Web UI."
            wait $WEBUI_PID 2>/dev/null || true
        fi

        kill "$WEBUI_PID" 2>/dev/null || true
        ;;

    3|*)
        ok "Setup complete."
        log "Start manually:"
        log "  ${DIM}./start_web_ui.sh${NC}"
        log "  ${DIM}./start_motionflow.sh${NC}"
        ;;
esac

echo ""
ok "${BOLD}MotionFlow is ready.${NC}"
echo ""
