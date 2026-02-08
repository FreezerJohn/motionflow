"""
Stateless visualization module for MotionFlow.

The Visualizer receives Detection and Zone objects and draws them.
It does NOT maintain any state - all state is owned by the domain models.
"""

# Type hints for domain models (avoid circular imports)
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from core.models import Detection, Door, Zone


class Colors:
    """
    Simple color palette for visualization.
    Replaces ultralytics.utils.plotting.Colors to remove dependency.
    """
    def __init__(self):
        # Ultralytics default palette (hex -> BGR)
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231",
            "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC",
            "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self._hex2bgr(h) for h in hexs]
        self.n = len(self.palette)

    def _hex2bgr(self, h: str) -> tuple[int, int, int]:
        """Convert hex color to BGR tuple."""
        return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

    def __call__(self, i: int, bgr: bool = False) -> tuple[int, int, int]:
        """Return color for index i."""
        c = self.palette[int(i) % self.n]
        return c if bgr else c[::-1]


class Visualizer:
    """
    Stateless visualizer for drawing detections and zones on video frames.

    All state (track history, zone membership, etc.) is owned by the
    Detection and Zone objects passed in. This class simply renders them.
    """

    def __init__(self):
        self.colors = Colors()

        # Distinct colors for zones (matching Web UI)
        # Hex colors converted to BGR
        self.zone_colors = [
            (0, 0, 255),      # #FF0000 Red
            (0, 255, 0),      # #00FF00 Green
            (255, 0, 0),      # #0000FF Blue
            (0, 255, 255),    # #FFFF00 Yellow
            (255, 255, 0),    # #00FFFF Cyan
            (255, 0, 255),    # #FF00FF Magenta
            (192, 192, 192),  # #C0C0C0 Silver
            (128, 128, 128),  # #808080 Gray
            (0, 0, 128),      # #800000 Maroon
            (0, 128, 128),    # #808000 Olive
            (0, 128, 0),      # #008000 Green
            (128, 0, 128),    # #800080 Purple
            (128, 128, 0),    # #008080 Teal
            (128, 0, 0),      # #000080 Navy
            (0, 165, 255),    # #FFA500 Orange
            (42, 42, 165),    # #A52A2A Brown
            (135, 184, 222),  # #DEB887 Burlywood
            (160, 158, 95),   # #5F9EA0 Cadet Blue
            (0, 255, 127),    # #7FFF00 Chartreuse
            (30, 105, 210),   # #D2691E Chocolate
            (80, 127, 255),   # #FF7F50 Coral
            (237, 149, 100),  # #6495ED Cornflower Blue
            (60, 20, 220),    # #DC143C Crimson
            (255, 255, 0)     # #00FFFF Cyan
        ]

        # Skeleton connections for YOLO pose (1-indexed in original, we use 0-indexed)
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        self.limb_colors = [
            self.colors(x, True) for x in [
                9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16
            ]
        ]
        self.kpt_color = self.colors(16, True)
        self.track_color = (230, 230, 230)  # Light gray for track lines
        self.centroid_color = (255, 0, 255)  # Magenta for centroid
        self.box_color = (200, 200, 200)  # Light gray for bounding box

    def draw_zones(self, frame: np.ndarray, zones: list['Zone']) -> None:
        """
        Draw zone polygons on the frame.

        Args:
            frame: The video frame to draw on
            zones: List of Zone objects to draw
        """
        h, w = frame.shape[:2]

        for i, zone in enumerate(zones):
            if not zone.points:
                continue

            # Use Zone's method if available, otherwise convert manually
            if hasattr(zone, 'get_pixel_points'):
                pts = zone.get_pixel_points((h, w)).reshape((-1, 1, 2))
            else:
                # Fallback for config Zone objects
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone.points], np.int32)
                pts = pts.reshape((-1, 1, 2))

            # Get zone color from list based on index
            color = self.zone_colors[i % len(self.zone_colors)]

            # Draw polygon outline
            cv2.polylines(frame, [pts], True, color, 2)

            # Draw label
            label_pos = pts[0][0]
            cv2.putText(frame, zone.name, (label_pos[0], label_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_doors(self, frame: np.ndarray, doors: list['Door']) -> None:
        """
        Draw door polygons with normal arrows and threshold line on the frame.

        Args:
            frame: The video frame to draw on
            doors: List of Door objects to draw
        """
        h, w = frame.shape[:2]
        frame_shape = (h, w)

        # Door-specific color (orange-ish to distinguish from zones)
        door_color = (0, 165, 255)  # Orange in BGR
        arrow_color = (0, 255, 0)   # Green for "enter" direction arrow
        tripwire_color = (0, 255, 255)  # Yellow for threshold/tripwire line

        for door in doors:
            if not door.points or len(door.points) < 3:
                continue

            # Get pixel points
            if hasattr(door, 'get_pixel_points'):
                pts = door.get_pixel_points(frame_shape).reshape((-1, 1, 2))
            else:
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in door.points], np.int32)
                pts = pts.reshape((-1, 1, 2))

            # Draw polygon outline (thicker than zones)
            cv2.polylines(frame, [pts], True, door_color, 3)

            # Fill with semi-transparent color
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts.reshape((-1, 2))], door_color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # Draw the threshold line (tripwire) - the bottom edge
            # This is the line that triggers crossing detection
            pts_flat = pts.reshape((-1, 2))
            if len(pts_flat) >= 4:
                # Find the two points with highest Y (bottom of frame)
                sorted_by_y = sorted(range(len(pts_flat)), key=lambda i: pts_flat[i][1], reverse=True)
                idx1, idx2 = sorted_by_y[0], sorted_by_y[1]
                p1, p2 = pts_flat[idx1], pts_flat[idx2]

                # Draw thick threshold line
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                         tripwire_color, 4)

                # Add small circles at endpoints
                cv2.circle(frame, (int(p1[0]), int(p1[1])), 6, tripwire_color, -1)
                cv2.circle(frame, (int(p2[0]), int(p2[1])), 6, tripwire_color, -1)

            # Calculate center for arrow
            if hasattr(door, 'get_center'):
                center = door.get_center(frame_shape)
            else:
                center = np.mean(pts.reshape((-1, 2)), axis=0)

            # Get the user-defined normal vector
            if hasattr(door, 'get_normal_vector'):
                normal = door.get_normal_vector()
            else:
                # Fallback: default pointing right
                normal = np.array([1.0, 0.0])

            # Fixed arrow length
            arrow_length = 50

            # Draw normal arrow (shows "enter" direction)
            arrow_end = center + normal * arrow_length

            cv2.arrowedLine(
                frame,
                (int(center[0]), int(center[1])),
                (int(arrow_end[0]), int(arrow_end[1])),
                arrow_color,
                3,
                tipLength=0.3
            )

            # Draw "ENTER" label at arrow tip
            label_pos = (int(arrow_end[0]) + 5, int(arrow_end[1]))
            cv2.putText(frame, "ENTER", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, arrow_color, 1)

            # Draw door name
            name_pos = pts[0][0]
            cv2.putText(frame, f"[Door] {door.name}", (name_pos[0], name_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, door_color, 2)

    def draw_tripwire_debug(
        self,
        frame: np.ndarray,
        tripwire_detectors: dict[str, Any],
        doors: list['Door']
    ) -> None:
        """
        Draw tripwire debug visualization: threshold lines and ground points.

        Args:
            frame: The video frame to draw on
            tripwire_detectors: Dict of door_id -> TripwireDetector
            doors: List of Door objects (for matching)
        """
        h, w = frame.shape[:2]
        frame_shape = (h, w)

        # Colors
        tripwire_color = (0, 255, 255)  # Yellow for tripwire line
        ground_point_color_real = (0, 255, 0)  # Green for real ankle
        ground_point_color_estimated = (0, 165, 255)  # Orange for estimated
        ground_point_color_predicted = (255, 0, 255)  # Magenta for predicted

        for door in doors:
            door_id = door.name
            if door_id not in tripwire_detectors:
                continue

            tripwire = tripwire_detectors[door_id]

            # Draw the threshold line (tripwire)
            p1, p2 = tripwire.get_threshold_line(frame_shape)
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     tripwire_color, 2, cv2.LINE_AA)

            # Draw small circles at endpoints
            cv2.circle(frame, (int(p1[0]), int(p1[1])), 5, tripwire_color, -1)
            cv2.circle(frame, (int(p2[0]), int(p2[1])), 5, tripwire_color, -1)

            # Draw ground points for all tracked persons
            ground_points = tripwire.get_ground_points_for_visualization()
            for track_id, info in ground_points.items():
                px, py = info.point

                # Choose color based on source
                if info.source == 'predicted':
                    color = ground_point_color_predicted
                    radius = 8
                elif info.is_estimated:
                    color = ground_point_color_estimated
                    radius = 6
                else:
                    color = ground_point_color_real
                    radius = 6

                # Draw ground point
                cv2.circle(frame, (int(px), int(py)), radius, color, -1)
                cv2.circle(frame, (int(px), int(py)), radius + 2, (255, 255, 255), 1)

                # Draw label showing source
                label = f"#{track_id} {info.source}"
                cv2.putText(frame, label, (int(px) + 10, int(py) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def draw_detections(self, frame: np.ndarray, detections: list['Detection']) -> None:
        """
        Draw all detections on the frame.

        This is the main method for rendering Detection objects. It draws:
        - Track history (movement path)
        - Centroid
        - Skeleton (keypoints and limbs)
        - Bounding box with ID and zone labels

        Args:
            frame: The video frame to draw on
            detections: List of Detection objects to draw
        """
        for det in detections:
            self._draw_track_history(frame, det)
            self._draw_centroid(frame, det)
            self._draw_velocity_vector(frame, det)
            self._draw_skeleton(frame, det)
            self._draw_label(frame, det)

    def _draw_velocity_vector(self, frame: np.ndarray, det: 'Detection') -> None:
        """Draw velocity vector arrow starting from centroid."""
        if det.centroid is None or det.velocity is None:
            return

        cx, cy = det.centroid
        vx, vy = det.velocity

        # Scale factor for visualization (pixels per second -> pixels on screen)
        # Adjust this value to make the arrow length reasonable
        scale = 0.5

        # Calculate end point
        end_x = int(cx + vx * scale)
        end_y = int(cy + vy * scale)

        # Only draw if there is significant movement
        if abs(vx) > 1 or abs(vy) > 1:
            cv2.arrowedLine(frame, (int(cx), int(cy)), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

    def _draw_track_history(self, frame: np.ndarray, det: 'Detection') -> None:
        """Draw the movement path from detection's track history."""
        if len(det.track_history) > 1:
            points = np.array(det.track_history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=self.track_color, thickness=1)

    def _draw_centroid(self, frame: np.ndarray, det: 'Detection') -> None:
        """Draw the centroid point."""
        if det.centroid is not None:
            cx, cy = det.centroid
            cv2.circle(frame, (int(cx), int(cy)), 4, self.centroid_color, -1)

    def _draw_skeleton(self, frame: np.ndarray, det: 'Detection') -> None:
        """Draw keypoints and skeleton limbs."""
        k = det.keypoints
        ndim = k.shape[-1]

        # Draw keypoints
        for _idx, point in enumerate(k):
            x, y = point[0], point[1]
            conf = point[2] if ndim == 3 else 1.0

            if conf < 0.5 or (x == 0 and y == 0):
                continue

            cv2.circle(frame, (int(x), int(y)), 3, self.kpt_color, -1)

        # Draw limbs
        for i, sk in enumerate(self.skeleton):
            idx1, idx2 = sk[0] - 1, sk[1] - 1  # Convert to 0-indexed

            pos1 = (int(k[idx1][0]), int(k[idx1][1]))
            pos2 = (int(k[idx2][0]), int(k[idx2][1]))

            if ndim == 3 and (k[idx1][2] < 0.5 or k[idx2][2] < 0.5):
                continue

            if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                continue

            cv2.line(frame, pos1, pos2, self.limb_colors[i], 2)

    def _draw_label(self, frame: np.ndarray, det: 'Detection') -> None:
        """Draw bounding box and label with ID and zones."""
        k = det.keypoints
        ndim = k.shape[-1]

        # Calculate bounding box from visible keypoints
        visible = k[:, 2] > 0.3 if ndim == 3 else np.ones(len(k), dtype=bool)
        if not np.any(visible):
            return

        min_x = np.min(k[visible, 0])
        max_x = np.max(k[visible, 0])
        min_y = np.min(k[visible, 1])
        max_y = np.max(k[visible, 1])

        # Build label
        label_parts = []
        if det.track_id != -1:
            label_parts.append(f"ID: {det.track_id}")
        else:
            label_parts.append("ID: ?")

        if det.zones:
            label_parts.append(f"In: {', '.join(det.zones)}")

        if det.speed > 0:
            label_parts.append(f"{det.speed:.1f} px/s")

        if det.action:
            label_parts.append(f"Action: {det.action}")

        label = " | ".join(label_parts)

        # Draw bounding box (subtle, light gray, thin)
        cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), self.box_color, 1)

        # Draw label background for readability
        (w, _h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (int(min_x), int(min_y) - 20), (int(min_x) + w, int(min_y)), self.box_color, -1)
        cv2.putText(frame, label, (int(min_x), int(min_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter on the frame."""
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def draw_action_list(
        self,
        frame: np.ndarray,
        detections: list['Detection'],
        position: str = 'bottom-left',
        margin: int = 10,
        font_scale: float = 0.6,
        thickness: int = 1,
        bg_alpha: float = 0.7
    ) -> None:
        """
        Draw a list of tracked people with their IDs and actions.

        This provides an easy-to-read summary of all detected people and their
        current actions, independent of where they are in the frame.

        Args:
            frame: BGR image to draw on (modified in-place)
            detections: List of Detection objects
            position: 'bottom-left', 'bottom-right', 'top-left', 'top-right'
            margin: Pixels from edge
            font_scale: Font size multiplier
            thickness: Text thickness
            bg_alpha: Background transparency (0-1)
        """
        if not detections:
            return

        # Build list of lines: "ID X: action"
        lines = []
        header = "People:"
        lines.append(header)

        # Sort by track ID for consistent ordering
        sorted_dets = sorted(detections, key=lambda d: d.track_id)

        for det in sorted_dets:
            action = det.action if det.action else "unknown"
            # Add confidence indicator
            conf_indicator = ""
            if det.action_confidence >= 0.8:
                conf_indicator = " ●"  # High confidence
            elif det.action_confidence >= 0.5:
                conf_indicator = " ○"  # Medium confidence

            line = f"  #{det.track_id}: {action}{conf_indicator}"
            lines.append(line)

        if len(lines) <= 1:  # Only header, no people
            return

        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = int(22 * font_scale / 0.6)
        padding = 8

        # Get max text width
        max_width = 0
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, w)

        # Calculate box dimensions
        box_width = max_width + 2 * padding
        box_height = len(lines) * line_height + padding

        # Calculate position
        h, w = frame.shape[:2]
        if 'right' in position:
            x = w - box_width - margin
        else:
            x = margin
        if 'bottom' in position:
            y = h - box_height - margin
        else:
            y = margin

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (100, 100, 100), 1)

        # Draw text
        for i, line in enumerate(lines):
            text_y = y + padding + (i + 1) * line_height - 5

            # Header in different color
            if i == 0:
                color = (200, 200, 200)  # Light gray for header
            else:
                color = (255, 255, 255)  # White for entries

            cv2.putText(frame, line, (x + padding, text_y), font, font_scale,
                       color, thickness, cv2.LINE_AA)
