"""
MotionFlow Engine - Main processing pipeline.

Orchestrates multi-stream video processing using the Axelera Voyager SDK:
- Capture frames from RTSP streams (GStreamer hardware decoding)
- Run pose estimation on Metis AIPU
- Manage tracked detections and zone/door events
- Render visualization and publish events
"""

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

from config.schema import AppConfig
from core.actions import ActionFilter, classify_detections_filtered
from core.events import EventManager, LoggingListener
from core.filters import PoseFilter
from core.models import Detection, DetectionManager, Door, Zone
from core.visualization import Visualizer

class AxeleraMultiStreamProcessor:
    """
    Multi-stream processor using Axelera SDK's streaming API.

    Handles multiple RTSP sources in a single GStreamer pipeline with
    hardware decoding, pipelined AIPU inference, and per-stream tracking.

    Threading: main thread runs display.App.run(), SDK-managed thread
    runs the inference loop via app.start_thread().
    """

    def __init__(
        self,
        stream_configs: list,  # List of stream configs
        model_name: str,
        visualization_enabled: bool,
        general_config: AppConfig | None = None,
        event_managers: list[EventManager] | None = None,
        config_path: str | None = None,
    ):
        self._stream_configs = stream_configs
        self._visualization_enabled = visualization_enabled
        self._config_path = config_path
        
        # Restart flag - set when restart requested via IPC
        self._restart_requested = False
        self._general_config = general_config
        
        # Map stream_id (0, 1, ...) to stream name
        self._stream_names = [cfg.name for cfg in stream_configs]
        self._rtsp_urls = [cfg.rtsp_url for cfg in stream_configs]

        # State flags
        self._running = False
        self._stream = None
        
        # Global FPS tracking
        self._total_frames = 0
        self._total_fps_time = time.time()

        # Import SDK components
        from axelera.app import config as ax_config, create_inference_stream, display
        self._ax_config = ax_config
        self._create_stream = create_inference_stream
        self._display = display

        # Pipeline YAML name (e.g. yolo11lpose-coco-tracker)
        self._network = model_name

        # FPS for filter frequency
        g = getattr(general_config, 'general', None) if general_config else None
        self._fps_filter_freq = int(getattr(g, 'max_fps_per_stream', 20)) if g else 20

        # Per-stream state
        self._per_stream: dict[int, dict] = {}
        for i, cfg in enumerate(stream_configs):
            em = event_managers[i] if event_managers else None
            self._per_stream[i] = {
                'name': cfg.name,
                'config': cfg,
                'detection_manager': DetectionManager(max_age=2.0),
                'zones': [Zone.from_config(z) for z in cfg.zones],
                'doors': [Door.from_config(d) for d in getattr(cfg, 'doors', [])],
                'event_manager': em,
                'filters': {},  # PoseFilter per track_id
                'filter_last_seen': {},  # Last seen time per track_id (for cleanup grace period)
                'action_filter': ActionFilter(confirm_frames=5, use_frames=True),
                'display_frame': None,
                'display_lock': threading.Lock(),
                'fps_counter': 0,
                'fps_time': time.time(),
                'fps': 0.0,
            }

        # Visualizer (shared, stateless)
        if visualization_enabled:
            self.visualizer = Visualizer()
        else:
            self.visualizer = None
        
        # SDK display window (set by engine when visualization enabled)
        self._display_window = None
        
        # Visualization worker threads (one per stream) - offload drawing from inference thread
        import queue
        self._viz_queues: dict[int, queue.Queue] = {}
        self._viz_threads: list[threading.Thread] = []
        self._viz_running = False
        
        # Frame broadcaster for live preview (web UI)
        # Always initialized - only encodes/sends when there are listeners
        self._frame_broadcaster = None
        try:
            from core.frame_broadcaster import FrameBroadcaster
            self._frame_broadcaster = FrameBroadcaster(
                on_restart_requested=self._request_restart,
                on_reload_requested=self._request_reload,
            )
            logger.info("[AxeleraMultiStream] FrameBroadcaster initialized for live preview")
        except ImportError as e:
            logger.warning(f"[AxeleraMultiStream] FrameBroadcaster not available: {e}")

        logger.info(f"[AxeleraMultiStream] Initialized with {len(stream_configs)} sources: {self._stream_names}")

    # IoU threshold for matching tracker boxes to keypoint detection boxes.
    # The tracker uses Kalman-filtered (smoothed) boxes which can diverge from
    # the raw detection boxes, especially during fast motion. A threshold that
    # is too high causes intermittent match failures, leading to pose stutter
    # (skeleton freezes while video continues). 0.3 is safe because tracks
    # are already associated to detections by the SDK's ByteTrack; we just
    # need to correlate which keypoint blob belongs to which track.
    _IOU_MATCH_THRESHOLD = 0.3

    def _extract_detections(self, frame_result) -> list[dict]:
        """Extract detections from SDK FrameResult.
        
        Iterates over tracked objects (stable IDs from ByteTrack) and matches
        them to keypoint detections by bbox IoU.
        """
        detections = []

        tracked_objects = getattr(frame_result, 'tracking', None)
        keypoint_objects = getattr(frame_result, 'keypoint_detections', None)
        
        if not tracked_objects:
            return detections
        
        # Build list of (box, keypoints, score) from keypoint detections for matching
        kp_data = []
        if keypoint_objects:
            for kp_obj in keypoint_objects:
                kp_box = np.array(kp_obj.box, dtype=np.float32) if hasattr(kp_obj, 'box') and kp_obj.box is not None else None
                kp_score = float(kp_obj.score) if hasattr(kp_obj, 'score') and kp_obj.score is not None else 0.0
                kp_keypoints = None
                if hasattr(kp_obj, 'keypoints') and kp_obj.keypoints is not None:
                    kpts = np.array(kp_obj.keypoints, dtype=np.float32)
                    if kpts.ndim == 1:
                        kpts = kpts.reshape(-1, 3)
                    kp_keypoints = kpts
                if kp_box is not None and kp_keypoints is not None:
                    kp_data.append({'box': kp_box, 'keypoints': kp_keypoints, 'score': kp_score, 'used': False})
        
        # Iterate over tracked objects (has stable track_id)
        for tracked_obj in tracked_objects:
            track_id = tracked_obj.track_id
            
            # Get tracked bbox from history (last entry is current position)
            tracked_box = None
            if hasattr(tracked_obj, 'history') and tracked_obj.history is not None:
                hist = tracked_obj.history
                if hasattr(hist, '__len__') and len(hist) > 0:
                    tracked_box = np.array(hist[-1], dtype=np.float32)
            
            if tracked_box is None:
                continue
            
            # Find best matching keypoint detection by IoU
            best_match = None
            best_iou = self._IOU_MATCH_THRESHOLD
            
            for idx, kp in enumerate(kp_data):
                if kp['used']:
                    continue
                iou = self._compute_iou(tracked_box, kp['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = kp
            
            if best_match is not None:
                best_match['used'] = True
                det = {
                    'bbox': tracked_box,  # Use tracker's smoothed box
                    'id': track_id,
                    'score': best_match['score'],
                    'keypoints': best_match['keypoints']
                }
                detections.append(det)

        return detections
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area

    def set_display_window(self, window):
        """Set the SDK display window for visualization."""
        self._display_window = window

    def create_stream(self):
        """Create the SDK inference stream. Call before running."""
        pipeline_config = self._ax_config.PipelineConfig(
            network=self._network,
            sources=self._rtsp_urls,
            aipu_cores=4,
            pipe_type='gst',
        )
        
        self._stream = self._create_stream(
            self._ax_config.SystemConfig(),
            self._ax_config.InferenceStreamConfig(),
            pipeline_config,
        )
        logger.info(f"[AxeleraMultiStream] Created stream with {len(self._rtsp_urls)} sources")
        return self._stream

    def reload_config(self) -> dict:
        """
        Hot-reload configuration from the config file.
        
        Updates zones, doors, event manager settings, and visualization.
        Does NOT update: stream URLs, model (requires restart).
        
        Returns dict with what was updated.
        """
        if not self._config_path:
            logger.warning("Cannot reload: no config path set")
            return {'error': 'no config path'}
        
        from config.schema import load_config
        try:
            new_config = load_config(self._config_path)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {'error': str(e)}
        
        updated = {'streams': [], 'visualizer': False}
        
        # Find streams by name and update their config
        enabled_streams = [s for s in new_config.streams if s.enabled]
        stream_name_to_config = {s.name: s for s in enabled_streams}
        
        for stream_id, state in self._per_stream.items():
            stream_name = state['name']
            if stream_name not in stream_name_to_config:
                continue
                
            new_stream_cfg = stream_name_to_config[stream_name]
            old_cfg = state['config']
            changes = []
            
            # Update zones if changed
            old_zones = [(z.name, z.points) for z in old_cfg.zones]
            new_zones = [(z.name, [(p.x, p.y) for p in z.points]) for z in new_stream_cfg.zones]
            if old_zones != new_zones:
                state['zones'] = [Zone.from_config(z) for z in new_stream_cfg.zones]
                changes.append('zones')
            
            # Update doors if changed
            old_doors = [(d.name, d.normal_angle) for d in getattr(old_cfg, 'doors', [])]
            new_doors = [(d.name, d.normal_angle) for d in getattr(new_stream_cfg, 'doors', [])]
            if old_doors != new_doors:
                state['doors'] = [Door.from_config(d) for d in new_stream_cfg.doors]
                changes.append('doors')
            
            # Update event manager debounce settings
            em = state['event_manager']
            if em:
                if em.zone_enter_confirm_frames != new_stream_cfg.zone_enter_confirm_frames:
                    em.zone_enter_confirm_frames = new_stream_cfg.zone_enter_confirm_frames
                    changes.append('zone_enter_confirm')
                if em.zone_exit_confirm_frames != new_stream_cfg.zone_exit_confirm_frames:
                    em.zone_exit_confirm_frames = new_stream_cfg.zone_exit_confirm_frames
                    changes.append('zone_exit_confirm')
                if em.zone_missing_grace_seconds != new_stream_cfg.zone_missing_grace_seconds:
                    em.zone_missing_grace_seconds = new_stream_cfg.zone_missing_grace_seconds
                    changes.append('zone_grace')
                if em.track_new_confirm_frames != new_stream_cfg.track_new_confirm_frames:
                    em.track_new_confirm_frames = new_stream_cfg.track_new_confirm_frames
                    changes.append('track_confirm')
            
            # Store new config reference
            state['config'] = new_stream_cfg
            
            if changes:
                updated['streams'].append({'name': stream_name, 'changes': changes})
                logger.info(f"♻️ Reloaded [{stream_name}]: {', '.join(changes)}")
        
        return updated

    @property
    def restart_requested(self) -> bool:
        """Returns True if a restart was requested via IPC."""
        return self._restart_requested

    def _request_restart(self) -> None:
        """Request engine restart. Called by FrameBroadcaster when restart IPC command received."""
        logger.info("Restart requested via IPC, stopping for restart...")
        self._restart_requested = True
        self._running = False

    def _request_reload(self) -> dict:
        """Hot-reload config. Called by FrameBroadcaster when reload IPC command received."""
        logger.info("Config reload requested via IPC")
        return self.reload_config()

    def inference_loop(self, app):
        """
        Main inference loop, run in SDK-managed thread via app.start_thread().
        """
        logger.info(f"[AxeleraMultiStream] Inference thread started")
        self._running = True
        frame_count = 0
        start_time = time.time()
        
        for frame_result in self._stream:
            if not self._running:
                break

            try:
                current_time = time.time()
                stream_id = frame_result.stream_id
                
                # Extract detections
                raw_detections = self._extract_detections(frame_result)
                
                # Get frame as numpy array
                frame = None
                if hasattr(frame_result, 'image') and frame_result.image is not None:
                    frame = frame_result.image.asarray('BGR')
                
                self._process_frame_result(stream_id, raw_detections, frame, current_time)
                
                frame_count += 1
                self._total_frames += 1
                    
            except Exception as e:
                logger.warning(f"[AxeleraMultiStream] Error processing frame: {e}")
                
        logger.info(f"📊 Inference stopped: {frame_count} total frames")

    def _process_frame_result(self, stream_id: int, raw_detections: list[dict], 
                               frame: np.ndarray | None, current_time: float):
        """Process a single frame: filter, detect, classify, broadcast, visualize."""
        # Get per-stream state
        if stream_id not in self._per_stream:
            return
            
        state = self._per_stream[stream_id]
        stream_name = state['name']
        
        # Apply filters
        filtered_detections = self._apply_filters(raw_detections, current_time, state)
        
        # Update detection manager
        detections = state['detection_manager'].update(filtered_detections, current_time)

        # Classify actions using angle-based algorithm
        classify_detections_filtered(detections, state['action_filter'], current_time)

        # Update event manager and capture events
        events = []
        if state['event_manager']:
            frame_shape = (1080, 1920)
            if frame is not None:
                frame_shape = frame.shape[:2]
            events = state['event_manager'].update(
                detections, state['zones'], frame_shape, current_time, doors=state['doors']
            )

        # Broadcast to web UI live preview (only if someone is watching)
        if self._frame_broadcaster and frame is not None:
            if self._frame_broadcaster.has_listeners(stream_name):
                self._frame_broadcaster.push(
                    cam_id=stream_name,
                    frame=frame,
                    detections=detections,
                    zones=state['zones'],
                    doors=state['doors'],
                    timestamp=current_time,
                    events=events
                )

        # Queue frame for local visualization (viz worker drops frames if slow)
        if self.visualizer and frame is not None and self._viz_running:
            self.queue_frame_for_viz(stream_id, frame, detections)

        # Per-stream FPS tracking
        state['fps_counter'] += 1
        elapsed = current_time - state['fps_time']
        if elapsed >= 5.0:
            state['fps'] = state['fps_counter'] / elapsed
            state['fps_counter'] = 0
            state['fps_time'] = current_time

        # Global FPS logging (every 10s)
        global_elapsed = current_time - self._total_fps_time
        if global_elapsed >= 10.0:
            total_fps = self._total_frames / global_elapsed
            logger.info(f"📊 Pipeline: {total_fps:.1f} FPS ({self._total_frames} frames in {global_elapsed:.0f}s)")
            self._total_frames = 0
            self._total_fps_time = current_time

    def _apply_filters(self, raw_detections: list[dict], current_time: float, state: dict) -> list[dict]:
        """Apply One Euro temporal smoothing to keypoints.
        
        Filters are kept for a grace period when tracks temporarily disappear,
        so the filter state is preserved when the track reappears.
        """
        filtered = []
        filters = state['filters']
        filter_last_seen = state['filter_last_seen']
        seen_ids = set()
        
        for det in raw_detections:
            track_id = det.get('id', -1)
            keypoints = det.get('keypoints')

            if track_id != -1 and keypoints is not None:
                seen_ids.add(track_id)
                filter_last_seen[track_id] = current_time
                
                if track_id not in filters:
                    filters[track_id] = PoseFilter(
                        min_cutoff=0.5,
                        beta=0.01,
                        freq=self._fps_filter_freq
                    )
                det['keypoints'] = filters[track_id].filter(keypoints, current_time)

            filtered.append(det)

        # Cleanup stale filters (grace period matches DetectionManager.max_age)
        grace_period = 2.0
        stale = [
            tid for tid in filters 
            if tid not in seen_ids and (current_time - filter_last_seen.get(tid, 0)) > grace_period
        ]
        for tid in stale:
            del filters[tid]
            if tid in filter_last_seen:
                del filter_last_seen[tid]

        return filtered

    def get_display_frame(self, stream_id: int) -> np.ndarray | None:
        """Get the latest rendered frame for a specific stream.
        
        Returns a copy to minimize lock contention with inference thread.
        """
        if stream_id in self._per_stream:
            state = self._per_stream[stream_id]
            with state['display_lock']:
                if state['display_frame'] is not None:
                    return state['display_frame'].copy()
        return None

    def start_viz_workers(self):
        """Start visualization worker threads (one per stream)."""
        self._viz_running = True
        
        for stream_id in self._per_stream:
            self._viz_queues[stream_id] = {
                'frame': None,
                'detections': None,
                'lock': threading.Lock(),
                'new_frame': threading.Event()
            }
            
            thread = threading.Thread(
                target=self._viz_worker,
                args=(stream_id,),
                name=f'VizWorker-{stream_id}',
                daemon=True
            )
            thread.start()
            self._viz_threads.append(thread)
        
        logger.info(f"[AxeleraMultiStream] Started {len(self._viz_threads)} visualization workers")

    def stop_viz_workers(self):
        """Stop visualization worker threads."""
        self._viz_running = False
        # Signal all workers to wake up and exit
        for slot in self._viz_queues.values():
            slot['new_frame'].set()

    def _viz_worker(self, stream_id: int):
        """Worker thread for visualization - draws on frames and sends to display."""
        from axelera import types
        
        state = self._per_stream[stream_id]
        stream_name = state['name']
        slot = self._viz_queues[stream_id]
        
        logger.debug(f"[VizWorker-{stream_id}] Started for stream {stream_name}")
        
        while self._viz_running:
            # Wait for a new frame (with timeout to check running flag)
            if not slot['new_frame'].wait(timeout=0.1):
                continue
            
            # Get the latest frame
            with slot['lock']:
                frame = slot['frame']
                detections = slot['detections']
                slot['frame'] = None  # Mark as consumed
                slot['new_frame'].clear()
            
            if frame is None:
                continue
                
            try:
                frame = frame.copy()
                
                self.visualizer.draw_zones(frame, state['zones'])
                self.visualizer.draw_doors(frame, state['doors'])

                if state['event_manager'] and state['doors']:
                    tripwire_detectors = state['event_manager'].get_tripwire_detectors()
                    if tripwire_detectors:
                        self.visualizer.draw_tripwire_debug(frame, tripwire_detectors, state['doors'])

                self.visualizer.draw_detections(frame, detections)
                self.visualizer.draw_action_list(frame, detections, position='bottom-left')

                if self._display_window is not None:
                    image = types.Image.fromarray(frame, types.ColorFormat.BGR)
                    self._display_window.show(image, None, stream_id)
                    
            except Exception as e:
                if self._viz_running:
                    logger.warning(f"[VizWorker-{stream_id}] Error: {e}")
        
        logger.debug(f"[VizWorker-{stream_id}] Stopped")

    def queue_frame_for_viz(self, stream_id: int, frame: np.ndarray, detections: list):
        """Queue a frame for async visualization. Always overwrites - latest frame wins."""
        if stream_id in self._viz_queues:
            slot = self._viz_queues[stream_id]
            with slot['lock']:
                # Just overwrite - if viz is slow, it gets the latest frame
                slot['frame'] = frame.copy()
                slot['detections'] = detections  # detections are small, no copy needed
            slot['new_frame'].set()  # Signal worker

    def stop(self):
        """Signal the inference loop to stop."""
        self._running = False

    def release(self):
        """Release resources."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None


class MotionFlowEngine:
    """
    Main engine that orchestrates multi-stream processing.
    
    Supports hot-reloading of configuration (zones, doors, event settings)
    without restarting the pipeline.
    """

    def __init__(self, config: AppConfig, config_path: str | None = None, visualize: bool = False):
        self.config = config
        self.config_path = config_path  # For hot-reload
        self.visualize = visualize
        self.running = False
        self.event_managers: list[EventManager] = []
        self._multi_stream_processor: AxeleraMultiStreamProcessor | None = None

        # Get pipeline name from config
        pipeline = config.general.pipeline
        logger.info(f"Using pipeline: {pipeline}")

        # Create MQTT publisher if enabled
        self.mqtt_publisher = self._setup_mqtt()

        # Get enabled streams
        enabled_streams = [s for s in config.streams if s.enabled]
        num_enabled = len(enabled_streams)

        if num_enabled == 0:
            raise ValueError("No enabled streams in configuration")

        # Verify SDK is available
        try:
            from axelera.app import create_inference_stream
        except ImportError as e:
            raise RuntimeError(
                "Axelera SDK not available. This application requires the Voyager SDK. "
                "Please ensure you're running on the target device with the SDK installed."
            ) from e

        # Create event managers
        for stream_conf in enabled_streams:
            event_manager = EventManager(
                stream_id=stream_conf.name,
                zone_enter_confirm_frames=getattr(stream_conf, 'zone_enter_confirm_frames', 2),
                zone_exit_confirm_frames=getattr(stream_conf, 'zone_exit_confirm_frames', 3),
                zone_missing_grace_seconds=getattr(stream_conf, 'zone_missing_grace_seconds', 0.35),
                track_new_confirm_frames=getattr(stream_conf, 'track_new_confirm_frames', 5),
            )
            event_manager.add_listener(LoggingListener())
            if self.mqtt_publisher:
                event_manager.add_listener(self.mqtt_publisher)
            self.event_managers.append(event_manager)

        # Create multi-stream processor
        self._multi_stream_processor = AxeleraMultiStreamProcessor(
            stream_configs=enabled_streams,
            model_name=pipeline,
            visualization_enabled=self.visualize,
            general_config=config,
            event_managers=self.event_managers,
            config_path=config_path,
        )
        logger.info(f"Initialized AxeleraMultiStreamProcessor for {num_enabled} streams")

    def _setup_mqtt(self):
        """Set up MQTT publisher if enabled and available."""
        try:
            from core.mqtt import create_mqtt_publisher
            publisher = create_mqtt_publisher(self.config.mqtt)
            if publisher:
                publisher.start()
            return publisher
        except ImportError:
            # mqtt module not available
            return None
        except Exception as e:
            logger.error(f"Failed to setup MQTT: {e}")
            return None

    def reload_config(self) -> dict:
        """
        Hot-reload configuration from the config file.
        
        Updates: zones, doors, event settings.
        Does NOT update: stream URLs, model (requires restart).
        
        Returns dict describing what was updated.
        """
        # Delegate to processor (it loads from file)
        result = {}
        if self._multi_stream_processor:
            result = self._multi_stream_processor.reload_config()
        
        # Update our config reference
        if 'error' not in result and self.config_path:
            from config.schema import load_config
            try:
                self.config = load_config(self.config_path)
            except Exception:
                pass  # Processor already logged the error
        
        return result

    def run(self):
        """Main processing loop with SDK-managed threading."""
        self.running = True
        logger.info("Starting MotionFlow Engine...")

        self._run_with_sdk_threading(self.visualize)
        self.cleanup()

    def _run_with_sdk_threading(self, visualization_enabled: bool):
        """
        Run using SDK's display.App threading model.
        
        Main thread drives the GStreamer event loop, inference runs
        in a separate SDK-managed thread.
        """
        from axelera.app import display
        
        try:
            self._multi_stream_processor.create_stream()
            
            renderer = True if visualization_enabled else False
            
            with display.App(renderer=renderer) as app:
                window = None
                if visualization_enabled:
                    window = app.create_window("MotionFlow", display.FULL_SCREEN)
                    self._multi_stream_processor.set_display_window(window)
                    self._multi_stream_processor.start_viz_workers()
                
                # Start frame broadcaster for web UI live preview
                if self._multi_stream_processor._frame_broadcaster:
                    self._multi_stream_processor._frame_broadcaster.start()
                    logger.info("[MotionFlowEngine] FrameBroadcaster started for live preview")
                
                app.start_thread(
                    self._multi_stream_processor.inference_loop, 
                    (app,), 
                    name='MotionFlowInference'
                )
                
                logger.info(f"[MotionFlowEngine] Inference started (renderer={renderer})...")
                
                # app.run() blocks until the pipeline stops
                app.run()
            
            logger.info("[MotionFlowEngine] Processing complete")
        finally:
            # Stop visualization workers
            if visualization_enabled:
                self._multi_stream_processor.stop_viz_workers()
            
            # Stop frame broadcaster
            if self._multi_stream_processor._frame_broadcaster:
                self._multi_stream_processor._frame_broadcaster.stop()
            
            self.running = False

    def cleanup(self):
        """Release all resources."""
        # Release multi-stream processor
        if self._multi_stream_processor:
            self._multi_stream_processor.release()

        # Shutdown event managers
        for em in self.event_managers:
            em.shutdown()

        # Stop MQTT publisher
        if self.mqtt_publisher:
            self.mqtt_publisher.stop()

        logger.info("MotionFlow Engine stopped.")
