"""
Event system for MotionFlow.

This module provides an abstract event system that decouples detection logic
from external communication (MQTT, logging, webhooks, etc.).

The EventManager tracks state changes and dispatches events to registered listeners.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.models import TripwireDetector


class EventType(Enum):
    """Types of events that can be emitted by the system."""
    ZONE_ENTER = auto()      # Person entered a zone
    ZONE_EXIT = auto()       # Person left a zone
    DOOR_ENTER = auto()      # Person crossed door in "enter" direction (with normal)
    DOOR_EXIT = auto()       # Person crossed door in "exit" direction (against normal)
    ACTION_CHANGE = auto()   # Person's action changed (e.g., standing -> sitting)
    TRACK_NEW = auto()       # New person detected
    TRACK_LOST = auto()      # Person no longer detected (after timeout)


@dataclass
class MotionEvent:
    """
    Represents an event in the MotionFlow system.

    Events are generated when meaningful state changes occur, such as
    a person entering or exiting a zone, or changing their action.
    """
    type: EventType
    timestamp: float
    stream_id: str
    detection_id: int
    zone_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to a dictionary for serialization."""
        result = {
            'event': self.type.name.lower(),
            'timestamp': self.timestamp,
            'stream': self.stream_id,
        }
        if self.zone_id:
            result['zone'] = self.zone_id
        if self.detection_id >= 0:
            result['person'] = {'id': self.detection_id}
            result['person'].update(self.data.get('person', {}))
        if 'occupancy' in self.data:
            result['occupancy'] = self.data['occupancy']
        return result


class EventListener(ABC):
    """
    Abstract base class for event listeners.

    Implement this interface to receive events from the EventManager.
    Examples: MqttPublisher, LoggingListener, WebhookSender
    """

    @abstractmethod
    def on_event(self, event: MotionEvent) -> None:
        """
        Called when an event is dispatched.

        Args:
            event: The MotionEvent that occurred
        """
        pass

    def start(self) -> None:
        """Called when the listener is registered. Override if needed."""
        pass

    def stop(self) -> None:
        """Called when the listener is unregistered or system shuts down."""
        pass


class LoggingListener(EventListener):
    """Simple listener that logs events to the standard logger with compact formatting."""

    # Event type symbols for visual hierarchy
    SYMBOLS = {
        'TRACK_NEW': '→',
        'TRACK_LOST': '←',
        'ZONE_ENTER': '▶',
        'ZONE_EXIT': '◀',
        'DOOR_ENTER': '⊕',
        'DOOR_EXIT': '⊖',
        'ACTION_CHANGE': '◆',
    }

    def __init__(self, level: int = logging.INFO):
        self.level = level
        self.logger = logging.getLogger('events')

    def on_event(self, event: MotionEvent) -> None:
        """Log event in compact, readable format (max ~100 chars)."""
        sym = self.SYMBOLS.get(event.type.name, '•')
        etype = event.type.name.replace('_', ' ').title()
        
        # Build compact event line
        parts = [f"{sym} {etype}"]
        
        # Add stream (abbreviated)
        stream = event.stream_id[:12] if len(event.stream_id) > 12 else event.stream_id
        parts.append(f"[{stream}]")
        
        # Add zone/door if present
        if event.zone_id:
            parts.append(f"zone={event.zone_id}")
        
        # Add person info from data
        data = event.data or {}
        person = data.get('person', {})
        if person:
            action = person.get('action', '')
            if action:
                parts.append(f"action={action}")
        
        # Add track ID
        parts.append(f"id={event.detection_id}")
        
        # Add direction for door events
        if 'door' in data:
            direction = data.get('direction', '')
            if direction:
                parts.append(f"dir={direction}")
        
        # Add prev_action for action change
        if 'prev_action' in data:
            parts.append(f"was={data['prev_action']}")
        
        self.logger.log(self.level, " ".join(parts))


@dataclass
class ZoneState:
    """Tracks the state of detections within a zone."""
    detection_ids: set[int] = field(default_factory=set)
    actions: dict[int, str] = field(default_factory=dict)  # detection_id -> action

    @property
    def occupancy(self) -> int:
        return len(self.detection_ids)


class EventManager:
    """
    Manages event generation and dispatching for a stream.

    The EventManager tracks the state of detections relative to zones
    and generates events when state changes occur. It maintains:
    - Which detection IDs are in which zones
    - What action each detection is performing
    - When detections were last seen (for TRACK_LOST events)

    Usage:
        manager = EventManager(stream_id="living_room")
        manager.add_listener(LoggingListener())

        # In the processing loop:
        events = manager.update(detections, zones, frame_shape)
    """

    def __init__(
        self,
        stream_id: str,
        track_lost_timeout: float = 2.0,
        zone_enter_confirm_frames: int = 2,
        zone_exit_confirm_frames: int = 3,
        zone_missing_grace_seconds: float = 0.35,
        track_new_confirm_frames: int = 5,
    ):
        """
        Initialize the EventManager.

        Args:
            stream_id: Identifier for the stream this manager handles
            track_lost_timeout: Seconds after which an unseen track is considered lost
            track_new_confirm_frames: Frames a track must be seen before TRACK_NEW event
        """
        self.stream_id = stream_id
        self.track_lost_timeout = track_lost_timeout

        # Debounce/grace against 1-2 frame dropouts:
        # - Require N consecutive frames of "inside" before ZONE_ENTER.
        # - Require N consecutive frames of "outside" before ZONE_EXIT.
        # - If a track disappears from detections briefly, suppress zone exit until grace expires.
        self.zone_enter_confirm_frames = max(1, int(zone_enter_confirm_frames))
        self.zone_exit_confirm_frames = max(1, int(zone_exit_confirm_frames))
        self.zone_missing_grace_seconds = max(0.0, float(zone_missing_grace_seconds))

        # New track confirmation: require N consecutive frames before TRACK_NEW
        self.track_new_confirm_frames = max(1, int(track_new_confirm_frames))

        # State tracking
        self._zone_states: dict[str, ZoneState] = {}  # zone_id -> ZoneState
        self._last_seen: dict[int, float] = {}  # detection_id -> timestamp
        self._known_tracks: set[int] = set()  # All known track IDs

        # Pending tracks awaiting confirmation (track_id -> consecutive frames seen)
        self._pending_tracks: dict[int, int] = {}

        # Debounce state
        # zone_id -> {track_id: consecutive_frames_inside}
        self._zone_inside_counts: dict[str, dict[int, int]] = {}
        # zone_id -> {track_id: consecutive_frames_outside}
        self._zone_outside_counts: dict[str, dict[int, int]] = {}

        # Door crossing state using TripwireDetector
        # door_id -> TripwireDetector instance (created lazily)
        self._tripwire_detectors: dict[str, TripwireDetector] = {}
        # door_id -> {track_id: timestamp of last crossing event}
        # This prevents duplicate events for the same crossing
        self._door_last_crossing: dict[str, dict[int, float]] = {}
        # Minimum time between crossings for the same person/door (seconds)
        self._door_crossing_cooldown: float = 1.0

        # Listeners
        self._listeners: list[EventListener] = []

    def get_tripwire_detectors(self) -> dict[str, 'TripwireDetector']:
        """Get all tripwire detectors for visualization."""
        return self._tripwire_detectors

    def add_listener(self, listener: EventListener) -> None:
        """Register an event listener."""
        self._listeners.append(listener)
        listener.start()

    def remove_listener(self, listener: EventListener) -> None:
        """Unregister an event listener."""
        if listener in self._listeners:
            listener.stop()
            self._listeners.remove(listener)

    def _dispatch(self, event: MotionEvent) -> None:
        """Send an event to all registered listeners."""
        for listener in self._listeners:
            try:
                listener.on_event(event)
            except Exception as e:
                logging.error(f"Error in event listener: {e}")

    def _get_zone_state(self, zone_id: str) -> ZoneState:
        """Get or create the state for a zone."""
        if zone_id not in self._zone_states:
            self._zone_states[zone_id] = ZoneState()
        return self._zone_states[zone_id]

    def _get_inside_counts(self, zone_id: str) -> dict[int, int]:
        if zone_id not in self._zone_inside_counts:
            self._zone_inside_counts[zone_id] = {}
        return self._zone_inside_counts[zone_id]

    def _get_outside_counts(self, zone_id: str) -> dict[int, int]:
        if zone_id not in self._zone_outside_counts:
            self._zone_outside_counts[zone_id] = {}
        return self._zone_outside_counts[zone_id]

    def update(
        self,
        detections: list[Any],  # List of Detection objects
        zones: list[Any],       # List of Zone objects
        frame_shape: tuple,
        timestamp: float | None = None,
        doors: list[Any] | None = None  # List of Door objects
    ) -> list[MotionEvent]:
        """
        Update state and generate events based on current detections.

        This is the main method to call each frame. It will:
        1. Check for new tracks (TRACK_NEW)
        2. Check zone enter/exit (ZONE_ENTER, ZONE_EXIT)
        3. Check door crossings (DOOR_ENTER, DOOR_EXIT)
        4. Check action changes (ACTION_CHANGE)
        5. Check for lost tracks (TRACK_LOST)

        Args:
            detections: List of Detection objects from DetectionManager
            zones: List of Zone objects to check
            frame_shape: (height, width) of the frame
            timestamp: Current time (defaults to time.time())
            doors: List of Door objects to check for crossings

        Returns:
            List of MotionEvent objects generated this frame
        """
        if timestamp is None:
            timestamp = time.time()

        events: list[MotionEvent] = []

        # 1. Process tracks: confirm new, build zone membership
        current_ids, current_zone_members = self._process_tracks(
            detections, zones, frame_shape, timestamp, events
        )

        # 2. Process zone enter/exit (with debounce + missing grace)
        self._process_zones(
            detections, zones, current_ids, current_zone_members, timestamp, events
        )

        # 3. Process door crossings
        if doors:
            self._process_doors(
                detections, doors, frame_shape, timestamp, events
            )

        # 4. Clean up pending/lost tracks
        self._process_lost_tracks(current_ids, timestamp, events)

        # Dispatch all events to listeners
        for event in events:
            self._dispatch(event)

        return events

    def _process_tracks(
        self,
        detections: list[Any],
        zones: list[Any],
        frame_shape: tuple,
        timestamp: float,
        events: list[MotionEvent],
    ) -> tuple[set[int], dict[str, set[int]]]:
        """
        Process detections: update last-seen times, confirm new tracks,
        and build zone membership map.

        Returns:
            (current_ids, current_zone_members) where current_zone_members
            maps zone_id -> set of detection IDs currently inside that zone.
        """
        current_ids: set[int] = set()
        current_zone_members: dict[str, set[int]] = {z.name: set() for z in zones}

        for det in detections:
            track_id = det.track_id
            if track_id < 0:
                continue

            current_ids.add(track_id)
            self._last_seen[track_id] = timestamp

            # Check for new track (with confirmation frames to filter false positives)
            if track_id not in self._known_tracks:
                if track_id in self._pending_tracks:
                    self._pending_tracks[track_id] += 1
                else:
                    self._pending_tracks[track_id] = 1

                if self._pending_tracks[track_id] >= self.track_new_confirm_frames:
                    self._known_tracks.add(track_id)
                    del self._pending_tracks[track_id]
                    events.append(MotionEvent(
                        type=EventType.TRACK_NEW,
                        timestamp=timestamp,
                        stream_id=self.stream_id,
                        detection_id=track_id,
                        data={
                            'person': self._get_person_data(det),
                            'occupancy': len(self._known_tracks)
                        }
                    ))

            # Check which zones this detection is in
            if det.centroid is not None:
                point = (float(det.centroid[0]), float(det.centroid[1]))
                for zone in zones:
                    if zone.contains_point(point, frame_shape):
                        current_zone_members[zone.name].add(track_id)

        return current_ids, current_zone_members

    def _process_zones(
        self,
        detections: list[Any],
        zones: list[Any],
        current_ids: set[int],
        current_zone_members: dict[str, set[int]],
        timestamp: float,
        events: list[MotionEvent],
    ) -> None:
        """
        Process zone enter/exit events with debounce and missing-detection grace.

        Also detects action changes for people inside zones.
        """
        for zone in zones:
            zone_id = zone.name
            state = self._get_zone_state(zone_id)
            inside_counts = self._get_inside_counts(zone_id)
            outside_counts = self._get_outside_counts(zone_id)
            current_in_zone = current_zone_members.get(zone_id, set())
            previous_in_zone = state.detection_ids.copy()

            # Consider all relevant track IDs for this zone.
            candidate_ids = set(current_in_zone) | set(previous_in_zone)

            for track_id in candidate_ids:
                is_detected_now = track_id in current_ids
                is_inside_now = track_id in current_in_zone
                was_inside_confirmed = track_id in previous_in_zone

                # If track is missing from detections, keep it "sticky" in the zone
                # until grace expires, to avoid exit/enter flip when pose/model drops.
                if not is_detected_now:
                    if was_inside_confirmed:
                        last_time = self._last_seen.get(track_id)
                        if last_time is not None and (timestamp - last_time) <= self.zone_missing_grace_seconds:
                            outside_counts.pop(track_id, None)
                            continue
                    continue

                if is_inside_now:
                    outside_counts.pop(track_id, None)
                    inside_counts[track_id] = inside_counts.get(track_id, 0) + 1

                    if (not was_inside_confirmed) and inside_counts[track_id] >= self.zone_enter_confirm_frames:
                        det = self._find_detection(detections, track_id)
                        state.detection_ids.add(track_id)
                        if det:
                            action = getattr(det, 'action', None) or 'unknown'
                            state.actions[track_id] = action
                            events.append(MotionEvent(
                                type=EventType.ZONE_ENTER,
                                timestamp=timestamp,
                                stream_id=self.stream_id,
                                detection_id=track_id,
                                zone_id=zone_id,
                                data={
                                    'person': self._get_person_data(det),
                                    'occupancy': len(state.detection_ids)
                                }
                            ))
                else:
                    inside_counts.pop(track_id, None)

                    if was_inside_confirmed:
                        outside_counts[track_id] = outside_counts.get(track_id, 0) + 1
                        if outside_counts[track_id] >= self.zone_exit_confirm_frames:
                            state.detection_ids.discard(track_id)
                            state.actions.pop(track_id, None)
                            outside_counts.pop(track_id, None)
                            events.append(MotionEvent(
                                type=EventType.ZONE_EXIT,
                                timestamp=timestamp,
                                stream_id=self.stream_id,
                                detection_id=track_id,
                                zone_id=zone_id,
                                data={
                                    'occupancy': len(state.detection_ids)
                                }
                            ))

            # Check for action changes for people still in zone
            still_in_zone = current_in_zone & state.detection_ids
            for track_id in still_in_zone:
                det = self._find_detection(detections, track_id)
                if det:
                    current_action = getattr(det, 'action', None) or 'unknown'
                    previous_action = state.actions.get(track_id, 'unknown')
                    if current_action != previous_action:
                        state.actions[track_id] = current_action
                        events.append(MotionEvent(
                            type=EventType.ACTION_CHANGE,
                            timestamp=timestamp,
                            stream_id=self.stream_id,
                            detection_id=track_id,
                            zone_id=zone_id,
                            data={
                                'person': self._get_person_data(det),
                                'prev_action': previous_action,
                                'occupancy': len(state.detection_ids)
                            }
                        ))

    def _process_doors(
        self,
        detections: list[Any],
        doors: list[Any],
        frame_shape: tuple,
        timestamp: float,
        events: list[MotionEvent],
    ) -> None:
        """
        Process door crossings using TripwireDetector.

        For each detection, checks all doors for crossings. When a track
        crosses multiple doors in the same frame, only the closest door
        is used. Also handles disappeared tracks via prediction mode.
        """
        from core.models import TripwireDetector

        active_track_ids = {det.track_id for det in detections if det.track_id >= 0}

        # Ensure all doors have tripwire detectors
        for door in doors:
            door_id = door.name
            if door_id not in self._tripwire_detectors:
                self._tripwire_detectors[door_id] = TripwireDetector.from_door(door)
            if door_id not in self._door_last_crossing:
                self._door_last_crossing[door_id] = {}

        # Collect potential crossings: {track_id: (door_id, direction, distance, detection)}
        # We'll keep only the closest door for each track
        potential_crossings: dict[int, tuple] = {}

        # Process active detections against all doors
        for det in detections:
            track_id = det.track_id
            if track_id < 0:
                continue

            # Get ground point for distance calculation
            ground_point = None
            for door in doors:
                tripwire = self._tripwire_detectors[door.name]
                gp_info = tripwire.get_ground_point(det.keypoints, det.bbox)
                if gp_info:
                    ground_point = gp_info.point
                    break

            for door in doors:
                door_id = door.name
                tripwire = self._tripwire_detectors[door_id]

                # Skip if we recently fired an event for this track+door
                last_crossing = self._door_last_crossing[door_id].get(track_id, 0)
                if timestamp - last_crossing < self._door_crossing_cooldown:
                    continue

                crossing_dir = tripwire.check_crossing(
                    track_id=track_id,
                    keypoints=det.keypoints,
                    bbox=det.bbox,
                    frame_shape=frame_shape,
                    velocity=det.velocity if hasattr(det, 'velocity') else None,
                    timestamp=timestamp,
                    detection_present=True
                )

                if crossing_dir and ground_point:
                    dist = tripwire.get_distance_to_tripwire(ground_point, frame_shape)
                    if track_id not in potential_crossings or dist < potential_crossings[track_id][2]:
                        potential_crossings[track_id] = (door_id, crossing_dir, dist, det)

        # Check disappeared tracks against all doors
        self._process_disappeared_tracks_doors(
            doors, active_track_ids, frame_shape, timestamp, potential_crossings
        )

        # Emit events for the closest door per track
        for track_id, (door_id, crossing_dir, _dist, det) in potential_crossings.items():
            event_type = EventType.DOOR_ENTER if crossing_dir == 'enter' else EventType.DOOR_EXIT
            self._door_last_crossing[door_id][track_id] = timestamp

            event_data = {
                'door': door_id,
                'direction': crossing_dir
            }
            if det:
                event_data['person'] = self._get_person_data(det)
            else:
                event_data['predicted'] = True

            events.append(MotionEvent(
                type=event_type,
                timestamp=timestamp,
                stream_id=self.stream_id,
                detection_id=track_id,
                zone_id=door_id,
                data=event_data
            ))

        # Cleanup stale tracks from all tripwire detectors
        for door in doors:
            self._tripwire_detectors[door.name].cleanup_stale_tracks(active_track_ids)

    def _process_disappeared_tracks_doors(
        self,
        doors: list[Any],
        active_track_ids: set[int],
        frame_shape: tuple,
        timestamp: float,
        potential_crossings: dict[int, tuple],
    ) -> None:
        """Check disappeared tracks for door crossings using prediction mode."""
        import numpy as np

        for door in doors:
            door_id = door.name
            tripwire = self._tripwire_detectors[door_id]

            tracked_by_tripwire = set(tripwire._track_states.keys())
            disappeared_tracks = tracked_by_tripwire - active_track_ids

            for track_id in disappeared_tracks:
                last_crossing = self._door_last_crossing[door_id].get(track_id, 0)
                if timestamp - last_crossing < self._door_crossing_cooldown:
                    continue

                state = tripwire._track_states.get(track_id)
                if not state or not state.last_ground_point:
                    continue

                dummy_kpts = np.zeros((17, 3))
                dummy_bbox = np.zeros(4)

                crossing_dir = tripwire.check_crossing(
                    track_id=track_id,
                    keypoints=dummy_kpts,
                    bbox=dummy_bbox,
                    frame_shape=frame_shape,
                    velocity=None,
                    timestamp=timestamp,
                    detection_present=False
                )

                if crossing_dir:
                    dist = tripwire.get_distance_to_tripwire(state.last_ground_point, frame_shape)
                    if track_id not in potential_crossings or dist < potential_crossings[track_id][2]:
                        potential_crossings[track_id] = (door_id, crossing_dir, dist, None)

    def _process_lost_tracks(
        self,
        current_ids: set[int],
        timestamp: float,
        events: list[MotionEvent],
    ) -> None:
        """
        Clean up pending tracks that disappeared (false positives)
        and emit TRACK_LOST for confirmed tracks that timed out.
        """
        # Clean up pending tracks not seen this frame (require consecutive frames)
        pending_to_remove = [
            tid for tid in self._pending_tracks if tid not in current_ids
        ]
        for tid in pending_to_remove:
            del self._pending_tracks[tid]
            self._last_seen.pop(tid, None)

        # Check for lost tracks
        lost_ids = []
        for track_id, last_time in list(self._last_seen.items()):
            if track_id not in current_ids:
                if timestamp - last_time > self.track_lost_timeout:
                    lost_ids.append(track_id)

        for track_id in lost_ids:
            self._known_tracks.discard(track_id)
            del self._last_seen[track_id]

            # Remove from all zone states and emit zone exit events
            for zone_id, state in self._zone_states.items():
                if track_id in state.detection_ids:
                    state.detection_ids.discard(track_id)
                    state.actions.pop(track_id, None)
                    self._zone_inside_counts.get(zone_id, {}).pop(track_id, None)
                    self._zone_outside_counts.get(zone_id, {}).pop(track_id, None)
                    events.append(MotionEvent(
                        type=EventType.ZONE_EXIT,
                        timestamp=timestamp,
                        stream_id=self.stream_id,
                        detection_id=track_id,
                        zone_id=zone_id,
                        data={'occupancy': len(state.detection_ids)}
                    ))

            # Ensure debounce state is cleared even if not in any known zone state
            for zmap in (self._zone_inside_counts, self._zone_outside_counts):
                for _zone_id, counts in zmap.items():
                    counts.pop(track_id, None)

            events.append(MotionEvent(
                type=EventType.TRACK_LOST,
                timestamp=timestamp,
                stream_id=self.stream_id,
                detection_id=track_id,
                data={'occupancy': len(self._known_tracks)}
            ))

    def _find_detection(self, detections: list[Any], track_id: int) -> Any | None:
        """Find a detection by track ID."""
        for det in detections:
            if det.track_id == track_id:
                return det
        return None

    def _get_person_data(self, det: Any) -> dict[str, Any]:
        """Extract person data from a Detection object for event payload."""
        data = {
            'id': det.track_id
        }

        if hasattr(det, 'action') and det.action:
            data['action'] = det.action

        if hasattr(det, 'speed'):
            data['speed'] = round(det.speed, 2)

        if hasattr(det, 'get_direction_name'):
            data['direction'] = det.get_direction_name()
        elif hasattr(det, 'velocity') and det.velocity is not None:
            # Fallback to raw velocity if method not available
            data['velocity'] = [round(v, 2) for v in det.velocity]

        return data

    def shutdown(self) -> None:
        """Clean up and stop all listeners."""
        for listener in self._listeners:
            try:
                listener.stop()
            except Exception as e:
                logging.error(f"Error stopping listener: {e}")
        self._listeners.clear()
