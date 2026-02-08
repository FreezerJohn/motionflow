import argparse
import gc
import json
import logging
import os
import sys
import threading
import time
from queue import Queue, Empty

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.schema import AppConfig, load_config, save_config

# Optional: flask-sock for WebSocket support
try:
    from flask_sock import Sock
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    Sock = None

# Optional: FrameSubscriber for receiving engine broadcasts
try:
    from core.frame_broadcaster import FrameSubscriber
    FRAME_SUBSCRIBER_AVAILABLE = True
except ImportError:
    FRAME_SUBSCRIBER_AVAILABLE = False
    FrameSubscriber = None

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# WebSocket support
if WEBSOCKET_AVAILABLE:
    sock = Sock(app)
else:
    sock = None

# Global lock for config file access
config_lock = threading.Lock()

# Config path - set via command line or environment, defaults to settings.yaml
# Use environment variable for flexibility (can be set before importing module)
CONFIG_PATH = os.path.abspath(os.environ.get('MOTIONFLOW_CONFIG', 'config/settings.yaml'))

# Target frame rate for web streaming (frames per second)
WEB_STREAM_FPS = 15
# Maximum frame dimension (resize if larger)
MAX_FRAME_DIM = 800
# JPEG quality (0-100, lower = smaller size)
JPEG_QUALITY = 70

# Track active streams for debugging
active_streams = {}
streams_lock = threading.Lock()

# Frame subscriber for receiving engine broadcasts (lazy initialized)
_frame_subscriber: 'FrameSubscriber | None' = None
_subscriber_lock = threading.Lock()

# Per-camera queues for distributing combined (frame, metadata) to clients
_combined_queues: dict[str, list[Queue]] = {}  # cam_id -> list of client queues
_queues_lock = threading.Lock()


def get_frame_subscriber() -> 'FrameSubscriber | None':
    """Get or create the singleton FrameSubscriber."""
    global _frame_subscriber
    if not FRAME_SUBSCRIBER_AVAILABLE:
        logging.warning("[WebUI] FrameSubscriber not available (pynng not installed?)")
        return None
    
    with _subscriber_lock:
        if _frame_subscriber is None:
            try:
                _frame_subscriber = FrameSubscriber()
                # Try to connect - may fail if engine isn't running yet
                if _frame_subscriber.connect():
                    logging.info("[WebUI] Connected to engine frame broadcaster")
                else:
                    logging.info("[WebUI] Engine not running yet - will retry in receiver thread")
                    
                # Start receiver thread (handles connection retries)
                thread = threading.Thread(
                    target=_frame_receiver_loop,
                    name="FrameReceiver",
                    daemon=True
                )
                thread.start()
            except Exception as e:
                logging.warning(f"[WebUI] Failed to create subscriber: {e}")
                _frame_subscriber = None
        return _frame_subscriber


def _frame_receiver_loop():
    """Background thread that receives frames from engine and distributes to client queues.
    
    Handles:
    - Initial connection if engine wasn't running at startup
    - Automatic reconnection when the main app restarts (detected by degraded FPS)
    """
    global _frame_subscriber
    logging.info("[WebUI] Frame receiver thread started")
    
    # FPS tracking per camera
    frame_counts: dict[str, int] = {}
    drop_counts: dict[str, int] = {}
    last_log_time = time.time()
    
    # Reconnection state
    total_frames_this_period = 0
    consecutive_low_fps_periods = 0
    MIN_EXPECTED_FPS = 5.0
    last_connect_attempt = 0.0
    CONNECT_RETRY_INTERVAL = 3.0  # Retry connection every 3 seconds
    
    while _frame_subscriber is not None:
        try:
            # If not connected, try to connect
            if not _frame_subscriber.is_connected:
                now = time.time()
                if now - last_connect_attempt >= CONNECT_RETRY_INTERVAL:
                    last_connect_attempt = now
                    if _frame_subscriber.connect():
                        logging.info("[WebUI] Connected to engine")
                    # else: will retry next interval
                time.sleep(0.5)  # Don't spin too fast
                continue
            
            result = _frame_subscriber.receive(timeout_ms=1000)
            
            if result is None:
                # Timeout - check if we have clients but no frames
                now = time.time()
                if now - last_log_time >= 10.0:
                    with _queues_lock:
                        has_clients = any(len(qs) > 0 for qs in _combined_queues.values())
                    
                    if has_clients and total_frames_this_period == 0:
                        logging.warning("[WebUI] No frames received with active clients, reconnecting...")
                        _frame_subscriber.reconnect()
                        consecutive_low_fps_periods = 0
                    
                    total_frames_this_period = 0
                    last_log_time = now
                continue
            
            cam_id, jpeg_bytes, metadata = result
            total_frames_this_period += 1
            
            # Track FPS per camera
            frame_counts[cam_id] = frame_counts.get(cam_id, 0) + 1
            now = time.time()
            elapsed = now - last_log_time
            
            if elapsed >= 10.0:
                # Log FPS
                for cid, count in frame_counts.items():
                    fps = count / elapsed
                    drops = drop_counts.get(cid, 0)
                    logging.info(f"[WebUI] Receiving {cid}: {fps:.1f} FPS, {drops} dropped")
                
                # Check for degraded connection
                with _queues_lock:
                    active_cameras = [cam for cam, qs in _combined_queues.items() if len(qs) > 0]
                
                if active_cameras:
                    total_fps = sum(frame_counts.get(cam, 0) for cam in active_cameras) / elapsed
                    expected_min = MIN_EXPECTED_FPS * len(active_cameras)
                    
                    if total_fps < expected_min:
                        consecutive_low_fps_periods += 1
                        logging.warning(f"[WebUI] Low FPS: {total_fps:.1f} < {expected_min:.0f} expected")
                        
                        if consecutive_low_fps_periods >= 2:
                            logging.warning("[WebUI] Consecutive low FPS, forcing reconnect...")
                            _frame_subscriber.reconnect()
                            consecutive_low_fps_periods = 0
                    else:
                        consecutive_low_fps_periods = 0
                
                frame_counts.clear()
                drop_counts.clear()
                total_frames_this_period = 0
                last_log_time = now
            
            # Distribute to all subscribed client queues
            with _queues_lock:
                if cam_id in _combined_queues:
                    for q in _combined_queues[cam_id]:
                        try:
                            if q.full():
                                drop_counts[cam_id] = drop_counts.get(cam_id, 0) + 1
                                try:
                                    q.get_nowait()
                                except Empty:
                                    pass
                            q.put_nowait((jpeg_bytes, metadata))
                        except Exception:
                            pass
                            
        except Exception as e:
            if _frame_subscriber is not None:
                logging.warning(f"[WebUI] Frame receiver error: {e}")
    
    logging.info("[WebUI] Frame receiver thread stopped")


def _subscribe_client(cam_id: str, combined_queue: Queue):
    """Subscribe a client to receive frames/metadata for a camera."""
    subscriber = get_frame_subscriber()
    if subscriber is None:
        return False
    
    with _queues_lock:
        if cam_id not in _combined_queues:
            _combined_queues[cam_id] = []
        _combined_queues[cam_id].append(combined_queue)
        
        # Subscribe to camera if this is the first client
        if len(_combined_queues[cam_id]) == 1:
            subscriber.subscribe(cam_id)
    
    return True


def _unsubscribe_client(cam_id: str, combined_queue: Queue):
    """Unsubscribe a client from receiving frames/metadata."""
    subscriber = get_frame_subscriber()
    
    with _queues_lock:
        if cam_id in _combined_queues:
            try:
                _combined_queues[cam_id].remove(combined_queue)
            except ValueError:
                pass
            if not _combined_queues[cam_id]:
                del _combined_queues[cam_id]
        
        # Unsubscribe from camera if no more clients
        remaining = len(_combined_queues.get(cam_id, []))
        if remaining == 0 and subscriber is not None:
            subscriber.unsubscribe(cam_id)


def set_config_path(path: str):
    """Set the config file path. Call before starting the app."""
    global CONFIG_PATH
    CONFIG_PATH = os.path.abspath(path)


def _open_rtsp_capture(rtsp_url: str, timeout: float = 8.0):
    """Open an RTSP VideoCapture with a timeout.
    
    Returns the opened capture, or None if connection failed.
    """
    result = [None]
    
    def _open():
        if rtsp_url.startswith('rtsp://'):
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            result[0] = cap
        else:
            cap.release()
    
    t = threading.Thread(target=_open, daemon=True)
    t.start()
    t.join(timeout)
    
    if t.is_alive() or result[0] is None:
        logging.warning(f"RTSP connection timed out or failed: {rtsp_url}")
        return None
    return result[0]


def get_video_frame(rtsp_url):
    """Generator for video frames with memory-safe streaming."""
    stream_id = threading.current_thread().ident
    cap = None

    with streams_lock:
        active_streams[stream_id] = rtsp_url
        logging.info(f"Starting stream {stream_id}: {rtsp_url} (active: {len(active_streams)})")

    try:
        cap = _open_rtsp_capture(rtsp_url)
        
        if cap is None:
            logging.error(f"Failed to open video stream: {rtsp_url}")
            return

        frame_interval = 1.0 / WEB_STREAM_FPS
        last_frame_time = 0
        consecutive_failures = 0
        max_failures = 30  # Give up after 30 consecutive failures

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        while True:
            # Rate limiting - don't read faster than target FPS
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                # Skip frames to maintain target FPS and reduce memory pressure
                time.sleep(frame_interval - elapsed)

            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logging.warning(f"Stream ended or failed: {rtsp_url}")
                    break
                time.sleep(0.1)  # Brief pause before retry
                continue

            consecutive_failures = 0
            last_frame_time = time.time()

            # Resize large frames to reduce memory and bandwidth
            h, w = frame.shape[:2]
            if max(h, w) > MAX_FRAME_DIM:
                scale = MAX_FRAME_DIM / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Encode with quality setting
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            # Explicitly delete to help garbage collection
            del frame
            del buffer

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            del frame_bytes

    except GeneratorExit:
        # Client disconnected - clean up gracefully
        logging.info(f"Client disconnected from stream: {rtsp_url}")
    except Exception as e:
        logging.error(f"Stream error for {rtsp_url}: {e}")
    finally:
        if cap is not None:
            cap.release()
        with streams_lock:
            active_streams.pop(stream_id, None)
            logging.info(f"Released stream {stream_id}: {rtsp_url} (active: {len(active_streams)})")
        # Force garbage collection after stream ends
        gc.collect()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/schema')
def get_schema():
    """Return the JSON schema for AppConfig.

    Note: Zones and doors are hidden via CSS because they're managed by the
    visual drawing tool on the video feed, not the JSON editor.
    """
    schema = AppConfig.model_json_schema()
    return jsonify(schema)

@app.route('/api/config', methods=['GET'])
def get_config():
    try:
        config = load_config(CONFIG_PATH)
        return jsonify(config.model_dump())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _signal_engine_restart():
    """Signal engine to restart via IPC."""
    subscriber = get_frame_subscriber()
    if subscriber:
        if subscriber.request_restart():
            logging.info("[WebUI] Engine restart requested via IPC")
            return True
        else:
            logging.warning("[WebUI] IPC restart request failed")
    else:
        logging.warning("[WebUI] Cannot signal engine: no IPC connection")
    return False


def _signal_engine_reload():
    """Signal engine to hot-reload config via IPC."""
    subscriber = get_frame_subscriber()
    if subscriber:
        result = subscriber.request_reload()
        if result is not None:
            logging.info(f"[WebUI] Engine config reloaded via IPC: {result}")
            return True
        else:
            logging.warning("[WebUI] IPC reload request failed")
    else:
        logging.warning("[WebUI] Cannot signal engine: no IPC connection")
    return False


def _config_needs_restart(old_config: AppConfig, new_config: AppConfig) -> bool:
    """Check if config changes require a full restart (vs hot-reload).
    
    Restart needed for: stream URLs, enabled/disabled streams, pipeline model.
    Hot-reload sufficient for: zones, doors, debounce settings, MQTT.
    """
    # Pipeline model changed
    if old_config.general.pipeline != new_config.general.pipeline:
        return True
    
    # Stream list or URLs changed
    old_streams = {s.name: (s.rtsp_url, s.enabled) for s in old_config.streams}
    new_streams = {s.name: (s.rtsp_url, s.enabled) for s in new_config.streams}
    if old_streams != new_streams:
        return True
    
    return False


@app.route('/api/config', methods=['POST'])
def update_config():
    try:
        new_data = request.json
        # Validate with Pydantic
        new_config = AppConfig(**new_data)
        
        with config_lock:
            # Load current config to compare
            try:
                old_config = load_config(CONFIG_PATH)
            except Exception:
                old_config = None
            
            # Save new config
            save_config(new_config, CONFIG_PATH)
            
            # Decide: restart (structural change) or reload (zones/doors/settings)
            if old_config is None or _config_needs_restart(old_config, new_config):
                _signal_engine_restart()
                action = "restart"
            else:
                _signal_engine_reload()
                action = "reload"
        
        return jsonify({"status": "success", "action": action})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/video_feed')
def video_feed():
    url = request.args.get('url')
    if not url:
        return "No URL provided", 400
    return Response(
        get_video_frame(url),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )


@app.route('/api/streams/active')
def active_stream_count():
    """Debug endpoint to check active streams."""
    with streams_lock:
        return jsonify({
            "active_count": len(active_streams),
            "streams": list(active_streams.values())
        })


@app.route('/api/reload', methods=['POST'])
def trigger_reload():
    """Trigger hot-reload of config (zones, doors, debounce settings) via IPC."""
    if _signal_engine_reload():
        return jsonify({"status": "ok", "action": "reload"})
    return jsonify({"error": "reload request failed"}), 500


@app.route('/api/restart', methods=['POST'])
def trigger_restart():
    """Trigger full engine restart via IPC."""
    if _signal_engine_restart():
        return jsonify({"status": "ok", "action": "restart"})
    return jsonify({"error": "restart request failed"}), 500


# =============================================================================
# Live Preview Endpoints - Live visualization from engine
# =============================================================================

@app.route('/live')
def live_view():
    """Render the live preview visualization page."""
    return render_template('live.html')


@app.route('/live/status')
def live_status():
    """Check if live preview connection to engine is available."""
    subscriber = get_frame_subscriber()
    connected = subscriber is not None
    
    # Get current subscriptions
    with _queues_lock:
        subscriptions = {
            cam_id: len(queues) 
            for cam_id, queues in _combined_queues.items()
        }
    
    return jsonify({
        "connected": connected,
        "websocket_available": WEBSOCKET_AVAILABLE,
        "subscriptions": subscriptions
    })


# WebSocket endpoint for synced frame + metadata (only if flask-sock is available)
if WEBSOCKET_AVAILABLE and sock is not None:
    @sock.route('/live/ws/<cam_id>')
    def live_websocket(ws, cam_id):
        """
        WebSocket endpoint for synced frame + metadata from engine.
        
        Sends JPEG frame as base64 along with metadata JSON for perfect sync.
        This eliminates the flickering caused by separate MJPEG and metadata streams.
        """
        import base64
        
        # Use combined queue to get frame+metadata together
        combined_queue = Queue(maxsize=2)  # Small buffer, drop old frames
        
        if not _subscribe_client(cam_id, combined_queue):
            ws.send(json.dumps({"error": "Could not connect to engine"}))
            return
        
        logging.info(f"[WebUI] WebSocket client connected for {cam_id}")
        
        # FPS tracking for WebSocket sends
        send_count = 0
        last_log_time = time.time()
        
        try:
            while True:
                try:
                    # Get combined (frame, metadata) - they arrive together
                    jpeg_bytes, metadata = combined_queue.get(timeout=5.0)
                    
                    # Encode JPEG as base64 and include in message
                    frame_b64 = base64.b64encode(jpeg_bytes).decode('ascii')
                    metadata['frame'] = frame_b64
                    
                    ws.send(json.dumps(metadata))
                    
                    # Track send FPS
                    send_count += 1
                    now = time.time()
                    if now - last_log_time >= 10.0:
                        fps = send_count / (now - last_log_time)
                        logging.info(f"[WebUI] Sending {cam_id}: {fps:.1f} FPS")
                        send_count = 0
                        last_log_time = now
                    
                except Empty:
                    # Send heartbeat to keep connection alive
                    ws.send(json.dumps({"heartbeat": True}))
        except Exception as e:
            logging.info(f"[WebUI] WebSocket client disconnected from {cam_id}: {e}")
        finally:
            _unsubscribe_client(cam_id, combined_queue)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MotionFlow Web UI")
    parser.add_argument(
        "-c", "--config",
        default="config/settings.yaml",
        help="Path to configuration file (default: config/settings.yaml)"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=5000,
        help="Port to run on (default: 5000)"
    )
    args = parser.parse_args()
    
    # Set config path before starting
    set_config_path(args.config)
    logging.info(f"Using config: {CONFIG_PATH}")
    
    app.run(host='0.0.0.0', port=args.port, debug=True, threaded=True)
