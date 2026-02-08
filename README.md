# MotionFlow

**AI-powered activity sensing for smart homes** – transforms ordinary IP cameras into privacy-preserving presence and activity sensors, running entirely on local edge hardware.

MotionFlow uses real-time pose estimation on an **Orange Pi 5 Plus** with the **Axelera Metis** AI accelerator to detect *what people are doing* (standing, sitting, lying, walking) and *where* (user-defined zones), then publishes state changes via **MQTT** for integration with Home Assistant, Node-RED, or any automation platform.

---

## 🎯 Overview

```mermaid
graph LR
    Cameras["📷 IP Cameras"] -->|RTSP| Engine["⚡ MotionFlow<br/>Pose · Track · Classify"]
    Engine -->|MQTT| SmartHome["🏠 Home Assistant<br/>Node-RED"]
    WebUI["🖥️ Web UI"] -.->|Configure<br/>& Monitor| Engine
```

**Why?** Commercial smart home sensors tell you *someone is there*. MotionFlow tells you *who is doing what, where* — "person on the couch, sitting" vs "person at the front door, walking in" — using cameras you already have, processed entirely on-device with no cloud dependency.

---

## ✨ Features

- **Multi-camera processing** – Handles multiple RTSP streams in a single pipeline
- **Activity classification** – Recognizes standing, sitting, lying, walking, phone use, reading from pose geometry
- **Zones & doors** – Define polygon zones and tripwire doors; get occupancy counts and directional enter/exit events
- **MQTT integration** – Debounced state-change events for Home Assistant / Node-RED
- **Web UI** – Configure zones and doors visually; live preview shows the annotated inference stream
- **Hardware-accelerated** – Runs YOLOv11-Pose on Axelera Metis AIPU with HW video decoding
- **Fully local** – No cloud, no data leaves your network


---

## 📋 Prerequisites

**Hardware** – Orange Pi 5 Plus with an Axelera Metis M.2 AI accelerator, plus one or more RTSP-capable IP cameras.

**OS** – Ubuntu 24.04 for the RK3588, using [Joshua Riek's ubuntu-rockchip](https://github.com/Joshua-Riek/ubuntu-rockchip/) images.

**Software:**
- **Axelera Voyager SDK** – installed on the target (provides GStreamer pipeline, AIPU runtime, tracking)
- **Python 3.10+** (included in the SDK venv)
- **MQTT broker** – e.g. Mosquitto, for Home Assistant / Node-RED integration
- Python dependencies: see `requirements.txt`

---

## 🚀 Quick Start

### 1. Install dependencies

MotionFlow runs inside the **Axelera Voyager SDK venv** on the target device:

```bash
# Activate the SDK venv
source ~/voyager-sdk/venv/bin/activate

# Install MotionFlow's dependencies into it
pip install -r requirements.txt
```

### 2. Configure

Edit `config/settings.yaml` directly, or use the Web UI at `http://<device-ip>:5000`:
- Add RTSP stream URLs for your cameras
- Draw **zones** (polygons) and **doors** (tripwires) on the live video
- Set MQTT broker address and topic prefix
- Hit **Save** – zones and settings hot-reload instantly

### 3. Run

```bash
source ~/voyager-sdk/venv/bin/activate

# Start the engine
python main.py --config config/settings.yaml

# Start the web UI (separate terminal)
python web_ui/app.py --config config/settings.yaml --port 5000
```

Or use the provided **systemd services** for production:

```bash
./systemd/install.sh
sudo systemctl start motionflow motionflow-webui

# View logs
journalctl -u motionflow -u motionflow-webui -f
```

### 4. Integrate

Subscribe to MQTT topics in Home Assistant or Node-RED:

```
motionflow/{stream_name}/{zone_name}
```

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Input["RTSP Camera Streams"]
        CAM1[Camera 1]
        CAM2[Camera 2]
        CAM3[Camera N...]
    end

    subgraph Engine["MotionFlow Engine"]
        direction TB
        subgraph SDK["Axelera Voyager SDK"]
            GStreamer["GStreamer<br/>Video Decode"]
            AIPU["YOLOv11-Pose<br/>Metis AIPU × 4 cores"]
            ByteTrack["ByteTrack<br/>Multi-Object Tracking"]
        end

        subgraph CoreLogic["Core Processing"]
            DetMgr["DetectionManager<br/>Track lifecycle & velocity"]
            PoseFilter["PoseFilter<br/>OneEuro smoothing"]
            Actions["ActionClassifier<br/>Pose geometry"]
            ZoneDoor["Zone & Door<br/>Polygon & tripwire"]
        end

        subgraph Events["Event System"]
            EvtMgr["EventManager<br/>Debouncing"]
            MQTTPub["MqttPublisher"]
        end

        FrameBroadcaster["FrameBroadcaster<br/>Raw frame + metadata<br/>(pynng PUB/SUB)"]

        subgraph LocalViz["Local Display (--visualize)"]
            Visualizer["Visualizer<br/>Skeleton & zone overlay"]
            SDKWindow["SDK Window"]
        end
    end

    subgraph Output["Smart Home"]
        MQTTBroker["MQTT Broker"]
        HA["Home Assistant"]
        NodeRED["Node-RED"]
    end

    subgraph WebUI["Web UI (Flask)"]
        ConfigEditor["Config Editor<br/>Zones, doors, settings"]
        FrameSubscriber["FrameSubscriber<br/>IPC listener"]
        LivePreview["Live Preview<br/>JS renders overlays"]
    end

    subgraph Config["Configuration"]
        YAML["settings.yaml"]
        Schema["schema.py<br/>Pydantic models"]
    end

    CAM1 & CAM2 & CAM3 -->|RTSP| GStreamer
    GStreamer --> AIPU --> ByteTrack
    ByteTrack --> DetMgr --> PoseFilter
    PoseFilter --> Actions --> ZoneDoor

    ZoneDoor --> EvtMgr --> MQTTPub
    ZoneDoor --> FrameBroadcaster
    ZoneDoor --> Visualizer --> SDKWindow

    MQTTPub -->|Publish| MQTTBroker --> HA & NodeRED

    FrameBroadcaster -.->|IPC| FrameSubscriber --> LivePreview

    ConfigEditor -->|Save| YAML
    YAML -->|Hot-reload /<br/>Restart| Engine
    Schema -.->|Validates| YAML

    style Input fill:#e8f4f8,stroke:#5ba3c9,color:#1a3a4a
    style SDK fill:#fff3cd,stroke:#d4a843,color:#5c4a1e
    style CoreLogic fill:#e8f0fe,stroke:#6b9bd2,color:#1e3a5f
    style Events fill:#fce7f3,stroke:#c77dba,color:#5c2d52
    style LocalViz fill:#f3f4f6,stroke:#9ca3af,color:#374151
    style WebUI fill:#d1fae5,stroke:#6bc9a0,color:#1a4a3a
    style Config fill:#f5f3ff,stroke:#a78bfa,color:#3b2d6e
    style Output fill:#fef2f2,stroke:#e8a0a0,color:#5c2d2d
    style FrameBroadcaster fill:#e0f2fe,stroke:#64a8d4,color:#1e3a5f
```

---

## 🖥️ Web UI

The Flask-based Web UI (`http://<device-ip>:5000`) provides two views:

### Config Editor
Draw zones and doors directly on the camera feed. Changes are hot-reloaded into the running engine on apply.

![Config Editor](doc/webUI_config1.png)

### Live Preview
Shows the annotated inference stream in real-time, with detection and event feeds below.

![Live Preview](doc/webUI_preview1.png)

---

## 📡 MQTT Events

**Topic pattern:**
```
motionflow/{stream_name}/{zone_name}
```

**Example payload:**
```json
{
  "event": "zone_enter",
  "timestamp": 1738857600.123,
  "stream": "living_room",
  "zone": "Couch1",
  "person": {
    "id": 3,
    "action": "sitting"
  },
  "occupancy": 1
}
```

**Event types:** `zone_enter` · `zone_exit` · `door_enter` · `door_exit` · `action_change` · `track_new` · `track_lost`

---

## ⚙️ Configuration

Configuration lives in `config/settings.yaml`, validated by Pydantic models in `config/schema.py`.

| Section | What it controls |
|---------|-----------------|
| `general` | Pipeline model, FPS limits |
| `mqtt` | Broker address, topic prefix, QoS, credentials |
| `streams[]` | Per-camera RTSP URL, zones, doors, debounce timings, action filters |

**Hot-reload vs restart:**
- Zones, doors, debounce, display settings → **hot-reloaded** (no downtime)
- Stream URLs, enabled/disabled, pipeline model → **automatic restart** via IPC

---

## 📁 Project Structure

```
MotionFlow/
├── main.py                  # Entry point
├── config/
│   ├── schema.py            # Pydantic models (AppConfig, StreamConfig, Zone, Door)
│   └── settings.yaml        # Runtime configuration
├── core/
│   ├── engine.py            # AxeleraMultiStreamProcessor – SDK integration
│   ├── models.py            # Domain models (Detection, Zone, Door, DetectionManager)
│   ├── actions.py           # Rule-based action classifier
│   ├── events.py            # EventManager, EventListener, MotionEvent
│   ├── filters.py           # PoseFilter (OneEuro wrapper)
│   ├── mqtt.py              # MQTT publisher
│   ├── visualization.py     # Skeleton / zone / action overlay
│   └── frame_broadcaster.py # IPC frame publisher (pynng PUB/SUB + REQ/REP)
├── web_ui/
│   ├── app.py               # Flask app (config editor + live preview)
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS
├── systemd/                 # Production systemd service files + installer
└── tests/                   # pytest unit & integration tests
```

---

## 🛠️ Development

```bash
# Run tests
pytest tests/
```

**Dependencies** are installed into the SDK venv (see Quick Start).
