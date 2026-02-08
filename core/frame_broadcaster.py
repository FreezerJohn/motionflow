"""
FrameBroadcaster - IPC publisher for live preview.

Publishes processed frames and metadata to external consumers (web UI)
via pynng PUB/SUB pattern. The engine only encodes JPEG and serializes
metadata when there are active listeners, saving CPU on the edge device.

IPC Protocol:
- PUB socket: ipc:///tmp/motionflow_frames.ipc
  Message format: {cam_id}\x00{jpeg_bytes}\x00{json_metadata}
  
- REP socket: ipc:///tmp/motionflow_ctrl.ipc
  For listener count updates from web UI
"""

import json
import logging
import struct
import threading
import time
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

try:
    import pynng
    PYNNG_AVAILABLE = True
except ImportError:
    PYNNG_AVAILABLE = False
    pynng = None  # type: ignore

if TYPE_CHECKING:
    from core.models import Detection, Door, Zone
    import pynng as pynng_types  # For type hints only

logger = logging.getLogger(__name__)


# Message separator (null byte)
MSG_SEP = b'\x00'


def normalize_bbox(bbox: np.ndarray, width: int, height: int) -> list[float]:
    """Convert pixel bbox [x1, y1, x2, y2] to normalized [x, y, w, h]."""
    x1, y1, x2, y2 = bbox
    return [
        float(x1 / width),
        float(y1 / height),
        float((x2 - x1) / width),
        float((y2 - y1) / height)
    ]


def normalize_keypoints(keypoints: np.ndarray, width: int, height: int) -> list[list[float]]:
    """
    Convert pixel keypoints to normalized coordinates.
    
    Input: (17, 3) array with [x, y, confidence]
    Output: List of [x, y, confidence] with x,y normalized to 0.0-1.0
    """
    result = []
    for kp in keypoints:
        x, y, conf = kp[0], kp[1], kp[2] if len(kp) > 2 else 1.0
        result.append([
            float(x / width),
            float(y / height),
            float(conf)
        ])
    return result


def normalize_trail(trail: list[tuple[float, float]], width: int, height: int) -> list[list[float]]:
    """Convert pixel trail coordinates to normalized 0.0-1.0."""
    return [[p[0] / width, p[1] / height] for p in trail]


def normalize_velocity(velocity: np.ndarray, width: int, height: int) -> list[float]:
    """Normalize velocity vector relative to frame dimensions."""
    return [float(velocity[0] / width), float(velocity[1] / height)]


def serialize_frame_metadata(
    frame_id: int,
    timestamp: float,
    cam_id: str,
    detections: list['Detection'],
    zones: list['Zone'],
    doors: list['Door'],
    frame_shape: tuple[int, int],
    events: list | None = None
) -> dict:
    """
    Serialize frame data for WebSocket transmission.
    
    All coordinates are normalized to 0.0-1.0 range.
    Trail history is included (pre-calculated, JS is stateless).
    
    Args:
        frame_id: Sequential frame counter
        timestamp: Unix timestamp
        cam_id: Camera/stream identifier
        detections: List of Detection objects
        zones: List of Zone objects
        doors: List of Door objects
        frame_shape: (height, width) of the frame
        events: List of MotionEvent objects generated this frame
        
    Returns:
        JSON-serializable dict matching the WebSocket protocol
    """
    h, w = frame_shape[:2]
    
    result = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "cam_id": cam_id,
        "zones": [
            {
                "id": z.name,
                "poly": z.points  # Already normalized in config!
            }
            for z in zones
        ],
        "doors": [
            {
                "id": d.name,
                "poly": d.points,  # Already normalized
                "normal_angle": d.normal_angle
            }
            for d in doors
        ],
        "detections": [
            {
                "id": det.track_id,
                "class": "person",
                "action": det.action,
                "action_confidence": det.action_confidence,
                "bbox": normalize_bbox(det.bbox, w, h),
                "velocity": normalize_velocity(det.velocity, w, h),
                "speed": det.speed,
                "skeleton": normalize_keypoints(det.keypoints, w, h),
                "trail": normalize_trail(det.track_history, w, h),
                "zones": det.zones  # List of zone names currently in
            }
            for det in detections
        ]
    }
    
    # Include events if any occurred this frame
    if events:
        result["events"] = [
            _serialize_event(evt)
            for evt in events
        ]
    
    return result


def _serialize_event(evt) -> dict:
    """Serialize a MotionEvent for WebSocket transmission."""
    event_data = {
        "type": evt.type.name.lower(),
        "timestamp": evt.timestamp,
        "track_id": evt.detection_id,
        "zone": evt.zone_id,
    }
    
    # Include non-person data fields
    for k, v in evt.data.items():
        if k == 'person':
            # Extract action from person data
            if 'action' in v:
                event_data['action'] = v['action']
        else:
            event_data[k] = v
    
    return event_data


class FrameBroadcaster:
    """
    Broadcasts processed frames and metadata via IPC for remote debugging.
    
    The engine calls `push()` only when `has_listeners()` returns True,
    avoiding expensive JPEG encoding when no one is watching.
    
    Uses pynng for efficient IPC:
    - PUB socket for frame/metadata broadcast (topic = cam_id)
    - REP socket for control messages (listener count updates)
    
    Thread-safe: can be called from inference thread while control
    listener runs in separate thread.
    """
    
    DEFAULT_FRAMES_URL = "ipc:///tmp/motionflow_frames.ipc"
    DEFAULT_CTRL_URL = "ipc:///tmp/motionflow_ctrl.ipc"
    
    def __init__(
        self,
        frames_url: str | None = None,
        ctrl_url: str | None = None,
        on_restart_requested: 'Callable[[], None] | None' = None,
        on_reload_requested: 'Callable[[], None] | None' = None,
    ):
        """
        Initialize the broadcaster.
        
        Args:
            frames_url: IPC URL for frame publishing (PUB socket)
            ctrl_url: IPC URL for control messages (REP socket)
            on_restart_requested: Callback when restart is requested via IPC
            on_reload_requested: Callback when config reload is requested via IPC
        """
        if not PYNNG_AVAILABLE:
            raise ImportError(
                "pynng is not installed. Install with: pip install pynng"
            )
        
        self._frames_url = frames_url or self.DEFAULT_FRAMES_URL
        self._ctrl_url = ctrl_url or self.DEFAULT_CTRL_URL
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
        
        # Listener tracking per camera
        self._listener_counts: dict[str, int] = {}
        self._listener_lock = threading.Lock()
        
        # Sockets (lazy initialized)
        self._pub_socket: "pynng_types.Pub0 | None" = None
        self._rep_socket: "pynng_types.Rep0 | None" = None
        
        # Control listener thread
        self._ctrl_thread: threading.Thread | None = None
        self._running = False
        
        # Frame counter per camera
        self._frame_counters: dict[str, int] = {}
        
        # Restart/reload callbacks
        self._on_restart_requested = on_restart_requested
        self._on_reload_requested = on_reload_requested
        
        logger.info(f"[FrameBroadcaster] Initialized (frames={self._frames_url}, ctrl={self._ctrl_url})")
    
    def start(self) -> None:
        """Start the broadcaster (open sockets, start control listener)."""
        if self._running:
            return
        
        # Create PUB socket for frames
        self._pub_socket = pynng.Pub0()
        # Minimize send buffer to avoid frame queuing/bursting
        self._pub_socket.send_buffer_size = 2
        self._pub_socket.listen(self._frames_url)
        logger.info(f"[FrameBroadcaster] PUB socket listening on {self._frames_url}")
        
        # Create REP socket for control messages
        self._rep_socket = pynng.Rep0()
        self._rep_socket.listen(self._ctrl_url)
        logger.info(f"[FrameBroadcaster] REP socket listening on {self._ctrl_url}")
        
        # Start control listener thread
        self._running = True
        self._ctrl_thread = threading.Thread(
            target=self._control_listener_loop,
            name="FrameBroadcaster-Ctrl",
            daemon=True
        )
        self._ctrl_thread.start()
    
    def stop(self) -> None:
        """Stop the broadcaster and clean up sockets."""
        self._running = False
        
        if self._pub_socket:
            try:
                self._pub_socket.close()
            except Exception:
                pass
            self._pub_socket = None
        
        if self._rep_socket:
            try:
                self._rep_socket.close()
            except Exception:
                pass
            self._rep_socket = None
        
        logger.info("[FrameBroadcaster] Stopped")
    
    def has_listeners(self, cam_id: str) -> bool:
        """Check if any clients are listening to this camera."""
        with self._listener_lock:
            return self._listener_counts.get(cam_id, 0) > 0
    
    def get_listener_count(self, cam_id: str) -> int:
        """Get the number of listeners for a camera."""
        with self._listener_lock:
            return self._listener_counts.get(cam_id, 0)
    
    def push(
        self,
        cam_id: str,
        frame: np.ndarray,
        detections: list['Detection'],
        zones: list['Zone'],
        doors: list['Door'],
        timestamp: float | None = None,
        events: list | None = None
    ) -> bool:
        """
        Encode and publish frame + metadata.
        
        Only call this if has_listeners() returns True!
        
        Args:
            cam_id: Camera identifier (used as topic)
            frame: BGR frame to encode as JPEG
            detections: List of Detection objects
            zones: List of Zone objects
            doors: List of Door objects
            timestamp: Optional timestamp (defaults to current time)
            events: Optional list of MotionEvent objects generated this frame
            
        Returns:
            True if published successfully, False on error
        """
        if not self._pub_socket:
            return False
        
        try:
            # Get frame counter
            frame_id = self._frame_counters.get(cam_id, 0)
            self._frame_counters[cam_id] = frame_id + 1
            
            # Encode JPEG
            ret, jpeg_buffer = cv2.imencode('.jpg', frame, self._encode_params)
            if not ret:
                logger.warning(f"[FrameBroadcaster] JPEG encode failed for {cam_id}")
                return False
            jpeg_bytes = jpeg_buffer.tobytes()
            
            # Serialize metadata
            ts = timestamp if timestamp is not None else time.time()
            metadata = serialize_frame_metadata(
                frame_id=frame_id,
                timestamp=ts,
                cam_id=cam_id,
                detections=detections,
                zones=zones,
                doors=doors,
                frame_shape=frame.shape[:2],
                events=events
            )
            meta_bytes = json.dumps(metadata).encode('utf-8')
            
            # Build message: topic\x00jpeg\x00json
            message = cam_id.encode('utf-8') + MSG_SEP + jpeg_bytes + MSG_SEP + meta_bytes
            
            # Publish (non-blocking)
            self._pub_socket.send(message)
            
            return True
            
        except Exception as e:
            logger.warning(f"[FrameBroadcaster] Error publishing frame: {e}")
            return False
    
    def _control_listener_loop(self) -> None:
        """
        Listen for control messages from web UI.
        
        Protocol:
        - Request: JSON {"action": "subscribe"|"unsubscribe"|"restart"|"reload", "cam_id": "..."}
        - Response: JSON {"status": "ok", "listeners": N}
        
        The "restart" action triggers engine restart via callback.
        The "reload" action triggers config hot-reload via callback.
        """
        logger.info("[FrameBroadcaster] Control listener started")
        
        while self._running:
            try:
                # Receive with timeout so we can check _running flag
                self._rep_socket.recv_timeout = 1000  # 1 second
                
                try:
                    msg = self._rep_socket.recv()
                except pynng.Timeout:
                    continue
                
                # Parse request
                try:
                    request = json.loads(msg.decode('utf-8'))
                    action = request.get('action')
                    cam_id = request.get('cam_id')
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    self._rep_socket.send(json.dumps({"error": str(e)}).encode())
                    continue
                
                # Handle restart action (no cam_id required)
                if action == 'restart':
                    logger.info("[FrameBroadcaster] Restart requested via IPC")
                    self._rep_socket.send(json.dumps({"status": "ok", "action": "restart"}).encode())
                    if self._on_restart_requested:
                        self._on_restart_requested()
                    continue
                
                # Handle reload action (hot-reload config without restarting pipeline)
                if action == 'reload':
                    logger.info("[FrameBroadcaster] Config reload requested via IPC")
                    result = {}
                    if self._on_reload_requested:
                        result = self._on_reload_requested()
                    self._rep_socket.send(json.dumps({"status": "ok", "action": "reload", "result": result}).encode())
                    continue
                
                if not cam_id:
                    self._rep_socket.send(json.dumps({"error": "missing cam_id"}).encode())
                    continue
                
                # Process action
                with self._listener_lock:
                    if action == 'subscribe':
                        self._listener_counts[cam_id] = self._listener_counts.get(cam_id, 0) + 1
                        count = self._listener_counts[cam_id]
                        logger.info(f"[FrameBroadcaster] Subscribe {cam_id}: {count} listeners")
                    elif action == 'unsubscribe':
                        self._listener_counts[cam_id] = max(0, self._listener_counts.get(cam_id, 0) - 1)
                        count = self._listener_counts[cam_id]
                        logger.info(f"[FrameBroadcaster] Unsubscribe {cam_id}: {count} listeners")
                    else:
                        self._rep_socket.send(json.dumps({"error": f"unknown action: {action}"}).encode())
                        continue
                
                # Send response
                response = {"status": "ok", "listeners": count}
                self._rep_socket.send(json.dumps(response).encode())
                
            except Exception as e:
                if self._running:
                    logger.warning(f"[FrameBroadcaster] Control loop error: {e}")
        
        logger.info("[FrameBroadcaster] Control listener stopped")


class FrameSubscriber:
    """
    Subscribes to frame broadcasts from the engine.
    
    Used by the web UI to receive frames and metadata.
    Handles topic-based filtering per camera.
    Supports automatic reconnection when engine restarts.
    """
    
    def __init__(
        self,
        frames_url: str | None = None,
        ctrl_url: str | None = None
    ):
        """
        Initialize the subscriber.
        
        Args:
            frames_url: IPC URL for frame subscription (SUB socket)
            ctrl_url: IPC URL for control messages (REQ socket)
        """
        if not PYNNG_AVAILABLE:
            raise ImportError(
                "pynng is not installed. Install with: pip install pynng"
            )
        
        self._frames_url = frames_url or FrameBroadcaster.DEFAULT_FRAMES_URL
        self._ctrl_url = ctrl_url or FrameBroadcaster.DEFAULT_CTRL_URL
        
        # Sockets
        self._sub_socket: "pynng_types.Sub0 | None" = None
        self._req_socket: "pynng_types.Req0 | None" = None
        
        # Track subscriptions (persisted across reconnects)
        self._subscriptions: set[str] = set()
        self._sub_lock = threading.Lock()
        
        # Connection state
        self._connected = False
        self._consecutive_failures = 0
        
        logger.info(f"[FrameSubscriber] Initialized (frames={self._frames_url})")
    
    def connect(self) -> bool:
        """
        Connect to the broadcaster.
        
        Returns:
            True if connection successful, False otherwise.
            Safe to call multiple times - will close existing sockets first.
        """
        # Close existing sockets first
        self._close_sockets()
        
        try:
            # SUB socket for receiving frames
            self._sub_socket = pynng.Sub0(
                recv_timeout=1000,  # 1 second receive timeout
            )
            self._sub_socket.recv_buffer_size = 2
            
            # Try blocking dial first - this will fail fast if publisher not running
            # dial() with block=None (default) tries blocking first, then non-blocking
            self._sub_socket.dial(self._frames_url, block=True)
            logger.info(f"[FrameSubscriber] SUB socket connected to {self._frames_url}")
            
            # Subscribe to all messages immediately after connecting
            self._sub_socket.subscribe(b'')
            logger.info("[FrameSubscriber] Subscribed to all messages (empty topic)")
            
            # REQ socket for control messages (notifications to broadcaster)
            self._req_socket = pynng.Req0()
            self._req_socket.dial(self._ctrl_url, block=True)
            logger.info(f"[FrameSubscriber] REQ socket connected to {self._ctrl_url}")
            
            self._connected = True
            self._consecutive_failures = 0
            logger.info("[FrameSubscriber] Connected successfully")
            return True
            
        except pynng.exceptions.ConnectionRefused:
            logger.info("[FrameSubscriber] Publisher not running (connection refused)")
            self._close_sockets()
            self._connected = False
            return False
        except Exception as e:
            logger.warning(f"[FrameSubscriber] Connect failed: {e}")
            self._close_sockets()
            self._connected = False
            return False
    
    def subscribe_all(self) -> bool:
        """
        Subscribe to ALL messages (empty topic matches everything).
        
        This is more robust than per-camera subscriptions because nng's
        automatic reconnection preserves the empty subscription. Filtering
        by camera is then done at the application layer.
        
        Returns:
            True if subscription successful
        """
        if not self._sub_socket:
            logger.warning("[FrameSubscriber] Cannot subscribe_all: no socket")
            return False
        
        try:
            # Empty string matches all messages
            self._sub_socket.subscribe(b'')
            logger.info("[FrameSubscriber] Subscribed to all messages (empty topic)")
            return True
        except Exception as e:
            logger.warning(f"[FrameSubscriber] subscribe_all failed: {e}")
            return False
    
    def _close_sockets(self) -> None:
        """Close sockets without unsubscribing (internal helper)."""
        if self._sub_socket:
            try:
                self._sub_socket.close()
            except Exception:
                pass
            self._sub_socket = None
        
        if self._req_socket:
            try:
                self._req_socket.close()
            except Exception:
                pass
            self._req_socket = None
        
        self._connected = False
    
    def reconnect(self) -> bool:
        """
        Force a full reconnect to the broadcaster.
        
        Closes existing sockets and creates new ones. This is necessary
        when the publisher restarts because IPC sockets become stale.
        
        Returns:
            True if reconnection successful.
        """
        logger.info("[FrameSubscriber] Forcing full reconnect...")
        
        # Remember what cameras we were tracking (for broadcaster notifications)
        with self._sub_lock:
            cameras_to_notify = list(self._subscriptions)
        
        # Full reconnect - close and reopen everything
        if not self.connect():
            logger.warning("[FrameSubscriber] Reconnect failed - publisher may be down")
            return False
        
        # Re-notify broadcaster about our subscriptions (best effort)
        for cam_id in cameras_to_notify:
            try:
                if self._req_socket:
                    request = {"action": "subscribe", "cam_id": cam_id}
                    self._req_socket.send(json.dumps(request).encode())
                    self._req_socket.recv_timeout = 1000
                    self._req_socket.recv()
                    logger.info(f"[FrameSubscriber] Re-notified: subscribe {cam_id}")
            except Exception as e:
                logger.debug(f"[FrameSubscriber] Re-notify {cam_id} failed: {e}")
        
        logger.info("[FrameSubscriber] Reconnected successfully")
        return True
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected
    
    def close(self) -> None:
        """Close connections and unsubscribe from all cameras."""
        # Unsubscribe from all
        for cam_id in list(self._subscriptions):
            self.unsubscribe(cam_id)
        
        if self._sub_socket:
            try:
                self._sub_socket.close()
            except Exception:
                pass
            self._sub_socket = None
        
        if self._req_socket:
            try:
                self._req_socket.close()
            except Exception:
                pass
            self._req_socket = None
        
        logger.info("[FrameSubscriber] Closed")
    
    def subscribe(self, cam_id: str) -> bool:
        """
        Register interest in a camera's frames and notify the broadcaster.
        
        Note: With subscribe_all(), the SUB socket already receives all messages.
        This method just notifies the broadcaster to start sending frames for
        this camera (so it knows there are listeners).
        
        Args:
            cam_id: Camera identifier to subscribe to
            
        Returns:
            True if notification successful (or queued for later)
        """
        with self._sub_lock:
            # Track this camera as subscribed
            self._subscriptions.add(cam_id)
            
            if not self._req_socket:
                logger.info(f"[FrameSubscriber] Queued subscription to {cam_id} (not connected)")
                return True  # Will be notified on reconnect
            
            try:
                # Notify broadcaster via REQ/REP (with short timeout)
                request = {"action": "subscribe", "cam_id": cam_id}
                self._req_socket.send(json.dumps(request).encode())
                
                # Wait for response with timeout
                self._req_socket.recv_timeout = 2000  # 2 seconds
                response = json.loads(self._req_socket.recv().decode())
                
                if response.get("status") == "ok":
                    logger.info(f"[FrameSubscriber] Notified broadcaster: subscribe {cam_id}")
                    return True
                else:
                    logger.warning(f"[FrameSubscriber] Subscribe notification failed: {response}")
                    return False
                    
            except Exception as e:
                # REQ/REP failed - broadcaster might be down, that's ok
                # We're still subscribed locally, frames will flow when it's back
                logger.debug(f"[FrameSubscriber] Subscribe notification failed for {cam_id}: {e}")
                return True  # Still return True - we're subscribed locally
    
    def unsubscribe(self, cam_id: str) -> bool:
        """
        Unregister interest in a camera's frames and notify the broadcaster.
        
        Args:
            cam_id: Camera identifier to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        with self._sub_lock:
            # Remove from subscriptions set
            was_subscribed = cam_id in self._subscriptions
            self._subscriptions.discard(cam_id)
            
            if not was_subscribed:
                return True  # Already unsubscribed
            
            try:
                # Notify broadcaster (best effort - don't block long)
                if self._req_socket:
                    request = {"action": "unsubscribe", "cam_id": cam_id}
                    self._req_socket.send(json.dumps(request).encode())
                    
                    self._req_socket.recv_timeout = 1000  # Short timeout
                    self._req_socket.recv()  # Wait for ack
                
                logger.info(f"[FrameSubscriber] Notified broadcaster: unsubscribe {cam_id}")
                return True
                
            except Exception as e:
                # REQ/REP failed - that's ok, broadcaster will notice we're gone
                logger.debug(f"[FrameSubscriber] Unsubscribe notification failed for {cam_id}: {e}")
                return True  # Still return true - we removed from set
    
    def receive(self, timeout_ms: int = 1000) -> tuple[str, bytes, dict] | None:
        """
        Receive a frame from any subscribed camera.
        
        Args:
            timeout_ms: Receive timeout in milliseconds
            
        Returns:
            Tuple of (cam_id, jpeg_bytes, metadata_dict) or None on timeout.
            Returns None on timeout (normal) or connection issues.
            Check needs_reconnect after consecutive None returns to determine
            if reconnection is needed.
        """
        if not self._sub_socket or not self._connected:
            self._consecutive_failures += 1
            return None
        
        try:
            self._sub_socket.recv_timeout = timeout_ms
            msg = self._sub_socket.recv()
            
            # Success - reset failure counter
            self._consecutive_failures = 0
            
            # Parse message: topic\x00jpeg\x00json
            # IMPORTANT: JPEG can contain \x00 bytes, so we can't use split()!
            # Find first \x00 (end of cam_id) and last \x00 (start of JSON)
            first_sep = msg.find(MSG_SEP)
            last_sep = msg.rfind(MSG_SEP)
            
            if first_sep == -1 or last_sep == -1 or first_sep == last_sep:
                logger.warning("[FrameSubscriber] Invalid message format")
                return None
            
            cam_id = msg[:first_sep].decode('utf-8')
            jpeg_bytes = msg[first_sep + 1:last_sep]
            metadata = json.loads(msg[last_sep + 1:].decode('utf-8'))
            
            return (cam_id, jpeg_bytes, metadata)
            
        except pynng.Timeout:
            # Timeout is normal - don't count as failure
            return None
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures <= 3:  # Only log first few
                logger.warning(f"[FrameSubscriber] Receive error: {e}")
            if self._consecutive_failures >= 5:
                self._connected = False  # Mark as disconnected
            return None
    
    @property
    def needs_reconnect(self) -> bool:
        """Check if we've had too many consecutive failures and need to reconnect."""
        return self._consecutive_failures >= 5 or (not self._connected and len(self._subscriptions) > 0)

    def request_restart(self) -> bool:
        """
        Request engine restart via IPC control channel.
        
        Returns:
            True if request was sent and acknowledged, False otherwise
        """
        with self._sub_lock:
            if not self._req_socket:
                logger.warning("[FrameSubscriber] Cannot request restart: not connected")
                return False
            
            try:
                request = {"action": "restart"}
                self._req_socket.send(json.dumps(request).encode())
                
                # Wait for response with timeout
                self._req_socket.recv_timeout = 2000  # 2 seconds
                response = json.loads(self._req_socket.recv().decode())
                
                if response.get("status") == "ok":
                    logger.info("[FrameSubscriber] Restart request acknowledged by engine")
                    return True
                else:
                    logger.warning(f"[FrameSubscriber] Restart request failed: {response}")
                    return False
                    
            except Exception as e:
                logger.warning(f"[FrameSubscriber] Failed to send restart request: {e}")
                return False

    def request_reload(self) -> dict | None:
        """
        Request config hot-reload via IPC control channel.
        
        Returns:
            Dict with reload result from engine, or None on failure.
        """
        with self._sub_lock:
            if not self._req_socket:
                logger.warning("[FrameSubscriber] Cannot request reload: not connected")
                return None
            
            try:
                request = {"action": "reload"}
                self._req_socket.send(json.dumps(request).encode())
                
                self._req_socket.recv_timeout = 5000  # 5 seconds (reload may take a moment)
                response = json.loads(self._req_socket.recv().decode())
                
                if response.get("status") == "ok":
                    logger.info(f"[FrameSubscriber] Reload acknowledged: {response.get('result', {})}")
                    return response.get("result", {})
                else:
                    logger.warning(f"[FrameSubscriber] Reload request failed: {response}")
                    return None
                    
            except Exception as e:
                logger.warning(f"[FrameSubscriber] Failed to send reload request: {e}")
                return None