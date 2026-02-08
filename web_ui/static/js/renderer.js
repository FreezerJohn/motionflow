/**
 * OverlayRenderer - Client-side visualization for MotionFlow live preview.
 * 
 * Renders detection metadata on a canvas overlay positioned over the MJPEG stream.
 * All coordinates from the server are normalized (0.0 - 1.0).
 * 
 * Renders:
 * - Zones (polygon overlays)
 * - Doors (polygons with direction arrows)
 * - Skeletons (COCO pose keypoints and limbs)
 * - Trails (movement history as polylines)
 * - Velocity vectors (arrows from centroid)
 * - Bounding boxes with ID labels
 */

class OverlayRenderer {
    constructor(canvas, img) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.img = img;
        
        // Image bounds within canvas (for proper coordinate mapping)
        this.imgBounds = { x: 0, y: 0, width: 0, height: 0 };
        
        // COCO skeleton connectivity (0-indexed)
        // Format: [from_keypoint, to_keypoint]
        this.skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
            [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6]
        ];
        
        // Keypoint names for reference
        this.keypointNames = [
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ];
        
        // Colors (matching OpenCV visualizer palette)
        // COCO skeleton limb colors - index into Ultralytics palette
        this.limbColorIndices = [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16];
        
        // Ultralytics color palette (hex)
        this.ultralyticsPalette = [
            '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231',
            '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
            '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC',
            '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7'
        ];
        
        this.colors = {
            keypoint: this.ultralyticsPalette[16],  // Purple (index 16)
            trail: 'rgba(230, 230, 230, 0.7)',      // Light gray
            velocity: '#FFFF00',                     // Yellow (matches OpenCV)
            bbox: 'rgba(200, 200, 200, 0.8)',       // Light gray
            label: '#FFFFFF',
            labelBg: 'rgba(0, 0, 0, 0.7)',
            centroid: '#FF00FF',                    // Magenta
        };
        
        // Zone colors (matching Python visualizer)
        this.zoneColors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF',
            '#C0C0C0', '#808080', '#800000', '#808000', '#008000', '#800080',
            '#008080', '#000080', '#FFA500', '#A52A2A', '#DEB887', '#5F9EA0'
        ];
        
        this.doorColor = '#FFA500';  // Orange for doors
        
        // Action color mapping
        this.actionColors = {
            'walking': '#4CAF50',
            'standing': '#2196F3',
            'sitting': '#FF9800',
            'lying': '#F44336',
            'falling': '#E91E63',
            'unknown': '#9E9E9E'
        };
        
        this.updateBounds();
    }
    
    /**
     * Update the image bounds within the canvas.
     * For simplicity, we use the full canvas dimensions (no letterboxing).
     * This matches how the config page zone editor works.
     */
    updateBounds() {
        // Use full canvas - matches config page behavior
        // The MJPEG stream and canvas should have matching aspect ratios
        this.imgBounds = {
            x: 0,
            y: 0,
            width: this.canvas.width,
            height: this.canvas.height
        };
    }
    
    /**
     * Convert normalized coordinates (0.0-1.0) to canvas pixels.
     */
    toPixel(normX, normY) {
        return [
            this.imgBounds.x + normX * this.imgBounds.width,
            this.imgBounds.y + normY * this.imgBounds.height
        ];
    }
    
    /**
     * Clear the canvas.
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    /**
     * Main render function - called on each WebSocket message.
     */
    render(data) {
        this.clear();
        this.updateBounds();
        
        // Draw zones first (background layer)
        if (data.zones) {
            this.drawZones(data.zones);
        }
        
        // Draw doors
        if (data.doors) {
            this.drawDoors(data.doors);
        }
        
        // Draw detections
        if (data.detections) {
            for (const det of data.detections) {
                this.drawTrail(det.trail);
                this.drawSkeleton(det.skeleton);
                this.drawVelocity(det);
                this.drawBbox(det);
                this.drawLabel(det);
            }
        }
    }
    
    /**
     * Draw zone polygons.
     */
    drawZones(zones) {
        zones.forEach((zone, i) => {
            if (!zone.poly || zone.poly.length < 3) return;
            
            const color = this.zoneColors[i % this.zoneColors.length];
            
            this.ctx.beginPath();
            const [startX, startY] = this.toPixel(zone.poly[0][0], zone.poly[0][1]);
            this.ctx.moveTo(startX, startY);
            
            for (let j = 1; j < zone.poly.length; j++) {
                const [x, y] = this.toPixel(zone.poly[j][0], zone.poly[j][1]);
                this.ctx.lineTo(x, y);
            }
            this.ctx.closePath();
            
            // Fill with transparency
            this.ctx.fillStyle = this.hexToRgba(color, 0.15);
            this.ctx.fill();
            
            // Stroke
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            // Label
            this.ctx.fillStyle = color;
            this.ctx.font = '12px sans-serif';
            this.ctx.fillText(zone.id, startX, startY - 5);
        });
    }
    
    /**
     * Draw door polygons with direction arrows.
     */
    drawDoors(doors) {
        doors.forEach(door => {
            if (!door.poly || door.poly.length < 3) return;
            
            this.ctx.beginPath();
            const [startX, startY] = this.toPixel(door.poly[0][0], door.poly[0][1]);
            this.ctx.moveTo(startX, startY);
            
            for (let j = 1; j < door.poly.length; j++) {
                const [x, y] = this.toPixel(door.poly[j][0], door.poly[j][1]);
                this.ctx.lineTo(x, y);
            }
            this.ctx.closePath();
            
            // Fill with transparency
            this.ctx.fillStyle = this.hexToRgba(this.doorColor, 0.2);
            this.ctx.fill();
            
            // Stroke
            this.ctx.strokeStyle = this.doorColor;
            this.ctx.lineWidth = 3;
            this.ctx.stroke();

            // Draw tripwire (bottom edge) — the two points with highest Y
            if (door.poly.length >= 4) {
                const sorted = [...door.poly].sort((a, b) => b[1] - a[1]);
                const [x1, y1] = this.toPixel(sorted[0][0], sorted[0][1]);
                const [x2, y2] = this.toPixel(sorted[1][0], sorted[1][1]);

                this.ctx.beginPath();
                this.ctx.moveTo(x1, y1);
                this.ctx.lineTo(x2, y2);
                this.ctx.strokeStyle = '#FFFF00';
                this.ctx.lineWidth = 4;
                this.ctx.stroke();

                // Endpoint circles
                for (const [cx, cy] of [[x1, y1], [x2, y2]]) {
                    this.ctx.beginPath();
                    this.ctx.arc(cx, cy, 5, 0, Math.PI * 2);
                    this.ctx.fillStyle = '#FFFF00';
                    this.ctx.fill();
                }
            }
            
            // Draw direction arrow
            if (door.normal_angle !== undefined) {
                // Calculate center
                let cx = 0, cy = 0;
                for (const p of door.poly) {
                    cx += p[0];
                    cy += p[1];
                }
                cx /= door.poly.length;
                cy /= door.poly.length;
                
                const [centerX, centerY] = this.toPixel(cx, cy);
                
                // Arrow direction from angle (degrees)
                const rad = door.normal_angle * Math.PI / 180;
                const arrowLen = 40;
                const endX = centerX + Math.cos(rad) * arrowLen;
                const endY = centerY + Math.sin(rad) * arrowLen;
                
                // Draw arrow
                this.drawArrow(centerX, centerY, endX, endY, '#00FF00', 3);
                
                // Label "ENTER"
                this.ctx.fillStyle = '#00FF00';
                this.ctx.font = '10px sans-serif';
                this.ctx.fillText('ENTER', endX + 5, endY);
            }
            
            // Door label
            this.ctx.fillStyle = this.doorColor;
            this.ctx.font = '12px sans-serif';
            this.ctx.fillText(`[Door] ${door.id}`, startX, startY - 5);
        });
    }
    
    /**
     * Draw skeleton from keypoints with per-limb colors matching OpenCV.
     */
    drawSkeleton(keypoints) {
        if (!keypoints || keypoints.length < 17) return;
        
        const minConf = 0.3;
        
        // Draw limbs with per-limb colors
        this.ctx.lineWidth = 2;
        
        for (let limbIdx = 0; limbIdx < this.skeleton.length; limbIdx++) {
            const [i, j] = this.skeleton[limbIdx];
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];
            
            if (!kp1 || !kp2) continue;
            
            const conf1 = kp1[2] !== undefined ? kp1[2] : 1.0;
            const conf2 = kp2[2] !== undefined ? kp2[2] : 1.0;
            
            if (conf1 < minConf || conf2 < minConf) continue;
            if (kp1[0] === 0 && kp1[1] === 0) continue;
            if (kp2[0] === 0 && kp2[1] === 0) continue;
            
            const [x1, y1] = this.toPixel(kp1[0], kp1[1]);
            const [x2, y2] = this.toPixel(kp2[0], kp2[1]);
            
            // Get limb color from palette
            const colorIdx = this.limbColorIndices[limbIdx] || 0;
            this.ctx.strokeStyle = this.ultralyticsPalette[colorIdx % this.ultralyticsPalette.length];
            
            this.ctx.beginPath();
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);
            this.ctx.stroke();
        }
        
        // Draw keypoints
        this.ctx.fillStyle = this.colors.keypoint;
        
        for (const kp of keypoints) {
            if (!kp) continue;
            
            const conf = kp[2] !== undefined ? kp[2] : 1.0;
            if (conf < minConf) continue;
            if (kp[0] === 0 && kp[1] === 0) continue;
            
            const [x, y] = this.toPixel(kp[0], kp[1]);
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
            this.ctx.fill();
        }
    }
    
    /**
     * Draw movement trail.
     */
    drawTrail(trail) {
        if (!trail || trail.length < 2) return;
        
        this.ctx.strokeStyle = this.colors.trail;
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        
        const [startX, startY] = this.toPixel(trail[0][0], trail[0][1]);
        this.ctx.moveTo(startX, startY);
        
        for (let i = 1; i < trail.length; i++) {
            const [x, y] = this.toPixel(trail[i][0], trail[i][1]);
            this.ctx.lineTo(x, y);
        }
        
        this.ctx.stroke();
    }
    
    /**
     * Draw velocity vector as arrow from pelvis center.
     * Clamped to max half body height for reasonable visualization.
     */
    drawVelocity(det) {
        if (!det.velocity || !det.bbox) return;
        
        const vx = det.velocity[0];
        const vy = det.velocity[1];
        
        // Skip if velocity is negligible
        let mag = Math.sqrt(vx * vx + vy * vy);
        if (mag < 0.001) return;
        
        // Calculate pelvis center from hip keypoints (11=left hip, 12=right hip in COCO)
        // Fall back to bbox center if hip keypoints not available
        let cx, cy;
        const kps = det.skeleton;
        if (kps && kps.length >= 13 && kps[11] && kps[12]) {
            const leftHip = kps[11];
            const rightHip = kps[12];
            const minConf = 0.3;
            
            // Check if both hips are valid
            const leftConf = leftHip[2] !== undefined ? leftHip[2] : 1.0;
            const rightConf = rightHip[2] !== undefined ? rightHip[2] : 1.0;
            
            if (leftConf >= minConf && rightConf >= minConf &&
                !(leftHip[0] === 0 && leftHip[1] === 0) &&
                !(rightHip[0] === 0 && rightHip[1] === 0)) {
                // Pelvis is average of left and right hip
                cx = (leftHip[0] + rightHip[0]) / 2;
                cy = (leftHip[1] + rightHip[1]) / 2;
            } else {
                // Fallback to bbox center
                cx = det.bbox[0] + det.bbox[2] / 2;
                cy = det.bbox[1] + det.bbox[3] / 2;
            }
        } else {
            // Fallback to bbox center
            cx = det.bbox[0] + det.bbox[2] / 2;
            cy = det.bbox[1] + det.bbox[3] / 2;
        }
        
        // Clamp velocity arrow to max half body height
        const maxLen = det.bbox[3] * 0.5;  // Half body height (normalized)
        let endVx = vx;
        let endVy = vy;
        
        // Scale factor to convert velocity to reasonable arrow length
        // Velocity is normalized (px/s / frame_dim), so we scale it back
        const scale = 0.5;  // Match OpenCV scale factor
        endVx *= scale;
        endVy *= scale;
        
        // Clamp to max length
        mag = Math.sqrt(endVx * endVx + endVy * endVy);
        if (mag > maxLen) {
            const clampFactor = maxLen / mag;
            endVx *= clampFactor;
            endVy *= clampFactor;
        }
        
        const [startX, startY] = this.toPixel(cx, cy);
        const [endX, endY] = this.toPixel(cx + endVx, cy + endVy);
        
        this.drawArrow(startX, startY, endX, endY, this.colors.velocity, 2);
    }
    
    /**
     * Draw bounding box.
     */
    drawBbox(det) {
        if (!det.bbox) return;
        
        const [x, y, w, h] = det.bbox;
        const [px, py] = this.toPixel(x, y);
        const pw = w * this.imgBounds.width;
        const ph = h * this.imgBounds.height;
        
        this.ctx.strokeStyle = this.colors.bbox;
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(px, py, pw, ph);
    }
    
    /**
     * Draw label with ID and action.
     */
    drawLabel(det) {
        if (!det.bbox) return;
        
        const [x, y] = det.bbox;
        const [px, py] = this.toPixel(x, y);
        
        // Build label text
        let label = `#${det.id}`;
        if (det.action) {
            label += ` ${det.action}`;
        }
        
        // Measure text
        this.ctx.font = 'bold 12px sans-serif';
        const metrics = this.ctx.measureText(label);
        const textWidth = metrics.width;
        const textHeight = 14;
        const padding = 4;
        
        // Background
        const bgX = px;
        const bgY = py - textHeight - padding * 2;
        
        const actionColor = this.actionColors[det.action] || this.actionColors['unknown'];
        this.ctx.fillStyle = this.colors.labelBg;
        this.ctx.fillRect(bgX, bgY, textWidth + padding * 2, textHeight + padding * 2);
        
        // Left border color based on action
        this.ctx.fillStyle = actionColor;
        this.ctx.fillRect(bgX, bgY, 3, textHeight + padding * 2);
        
        // Text
        this.ctx.fillStyle = this.colors.label;
        this.ctx.fillText(label, bgX + padding + 3, py - padding);
        
        // Zone indicators
        if (det.zones && det.zones.length > 0) {
            const zoneText = `[${det.zones.join(', ')}]`;
            this.ctx.font = '10px sans-serif';
            this.ctx.fillStyle = '#AAAAAA';
            this.ctx.fillText(zoneText, bgX + textWidth + padding * 3, py - padding);
        }
    }
    
    /**
     * Draw an arrow from (x1, y1) to (x2, y2).
     */
    drawArrow(x1, y1, x2, y2, color, lineWidth) {
        const headLen = 10;
        const angle = Math.atan2(y2 - y1, x2 - x1);
        
        this.ctx.strokeStyle = color;
        this.ctx.fillStyle = color;
        this.ctx.lineWidth = lineWidth;
        
        // Line
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
        
        // Arrowhead
        this.ctx.beginPath();
        this.ctx.moveTo(x2, y2);
        this.ctx.lineTo(
            x2 - headLen * Math.cos(angle - Math.PI / 6),
            y2 - headLen * Math.sin(angle - Math.PI / 6)
        );
        this.ctx.lineTo(
            x2 - headLen * Math.cos(angle + Math.PI / 6),
            y2 - headLen * Math.sin(angle + Math.PI / 6)
        );
        this.ctx.closePath();
        this.ctx.fill();
    }
    
    /**
     * Convert hex color to rgba.
     */
    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
}
