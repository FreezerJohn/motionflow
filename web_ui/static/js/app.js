// Initialize JSON Editor
let editor;
let currentConfig = null;
let currentPoints = []; // Points for the NEW zone/door being drawn
let currentDoorAngle = 0; // Current normal angle for door being drawn (degrees)
let canvas, ctx;
let draggingPoint = null; // { type: 'new'|'existing-zone'|'existing-door', zoneIndex: int, doorIndex: int, pointIndex: int }
let draggingArrow = false; // Whether we're dragging a door arrow
let draggingArrowDoorIndex = null; // If dragging an existing door's arrow, which door index (null = new door)
let drawMode = null; // null = edit mode, 'zone' = drawing new zone, 'door' = drawing new door
let isDrawingMode = false; // Whether we're in drawing mode (vs edit mode)

// Distinct colors generator
const distinctColors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF',
    '#C0C0C0', '#808080', '#800000', '#808000', '#008000', '#800080',
    '#008080', '#000080', '#FFA500', '#A52A2A', '#DEB887', '#5F9EA0',
    '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#00FFFF'
];

const doorColor = '#FFA500'; // Orange for doors

function getZoneColor(index) {
    return distinctColors[index % distinctColors.length];
}

document.addEventListener('DOMContentLoaded', function() {
    canvas = document.getElementById('video-overlay');
    ctx = canvas.getContext('2d');
    
    // Resize canvas to match video container
    function resizeCanvas() {
        const container = document.getElementById('video-container');
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
        drawOverlay();
    }
    window.addEventListener('resize', resizeCanvas);
    
    // Fetch Schema and Config
    Promise.all([
        fetch('/api/schema').then(res => res.json()),
        fetch('/api/config').then(res => res.json())
    ]).then(([schema, config]) => {
        currentConfig = config;
        
        // Configure JSON Editor
        JSONEditor.defaults.options.theme = 'bootstrap5';
        JSONEditor.defaults.options.iconlib = 'fontawesome5';
        
        editor = new JSONEditor(document.getElementById('editor_holder'), {
            schema: schema,
            startval: config,
            disable_edit_json: true,
            disable_properties: true,
            no_additional_properties: true
        });

        // Hook into changes to update the stream selector if streams change
        editor.on('change', function() {
            updateStreamSelect();
            updateZoneList();
            updateDoorList();
            drawOverlay(); // Redraw overlay when config changes (e.g. zone name edit)
        });

        updateStreamSelect();
        updateZoneList();
        updateDoorList();
        customizeEditorLayout();
        updateModeUI(); // Initialize mode UI
    });
    
    // New Zone button - enter zone drawing mode
    document.getElementById('new-zone-mode-btn').addEventListener('click', function() {
        enterDrawingMode('zone');
    });
    
    // New Door button - enter door drawing mode
    document.getElementById('new-door-mode-btn').addEventListener('click', function() {
        enterDrawingMode('door');
    });
    
    // Cancel drawing button
    document.getElementById('cancel-draw-btn').addEventListener('click', function() {
        exitDrawingMode();
    });
    
    function enterDrawingMode(mode) {
        const streamIndex = document.getElementById('stream-select').value;
        if (streamIndex === "") {
            alert('Please select a stream first');
            return;
        }
        
        isDrawingMode = true;
        drawMode = mode;
        currentPoints = [];
        currentDoorAngle = 0;
        document.getElementById('new-zone-name').value = '';
        updateModeUI();
        drawOverlay();
    }
    
    function exitDrawingMode() {
        isDrawingMode = false;
        drawMode = null;
        currentPoints = [];
        currentDoorAngle = 0;
        document.getElementById('new-zone-name').value = '';
        updateModeUI();
        drawOverlay();
    }
    
    function updateModeUI() {
        const editPanel = document.getElementById('edit-mode-panel');
        const drawPanel = document.getElementById('draw-mode-panel');
        
        if (isDrawingMode) {
            editPanel.style.display = 'none';
            drawPanel.style.display = 'block';
            
            const isZone = drawMode === 'zone';
            const badge = document.getElementById('draw-mode-badge');
            badge.className = isZone ? 'badge bg-primary' : 'badge bg-warning text-dark';
            badge.innerHTML = isZone 
                ? '<i class="fas fa-pencil-alt"></i> Drawing Zone' 
                : '<i class="fas fa-pencil-alt"></i> Drawing Door';
            
            document.getElementById('new-zone-name').placeholder = isZone ? 'Zone name (e.g. living_room)' : 'Door name (e.g. front_door)';
            document.getElementById('add-zone-btn').style.display = isZone ? '' : 'none';
            document.getElementById('add-door-btn').style.display = isZone ? 'none' : '';
            
            // Update canvas cursor and class
            canvas.style.cursor = 'crosshair';
            canvas.classList.add('drawing-mode');
        } else {
            editPanel.style.display = 'block';
            drawPanel.style.display = 'none';
            
            // Update canvas cursor and class
            canvas.style.cursor = 'grab';
            canvas.classList.remove('drawing-mode');
        }
        
        updatePointsCount();
        updateDoorRotationControls();
        updateButtons();
    }
    
    function updatePointsCount() {
        const countEl = document.getElementById('points-count');
        if (countEl) {
            countEl.textContent = `${currentPoints.length}/4 points`;
            countEl.className = currentPoints.length === 4 ? 'badge bg-success' : 'badge bg-info';
        }
    }
    
    function updateDoorRotationControls() {
        const controls = document.getElementById('door-rotation-controls');
        const show = isDrawingMode && drawMode === 'door' && currentPoints.length === 4;
        controls.style.display = show ? '' : 'none';
        if (show) {
            document.getElementById('door-angle-slider').value = currentDoorAngle;
            document.getElementById('door-angle-value').textContent = Math.round(currentDoorAngle) + '°';
        }
    }
    
    // Door angle slider
    document.getElementById('door-angle-slider').addEventListener('input', function() {
        currentDoorAngle = parseFloat(this.value);
        document.getElementById('door-angle-value').textContent = Math.round(currentDoorAngle) + '°';
        drawOverlay();
    });
    
    // Zone/Door name input - update button state when typing
    document.getElementById('new-zone-name').addEventListener('input', function() {
        updateButtons();
    });

    // Save Button
    document.getElementById('save-btn').addEventListener('click', function() {
        const value = editor.getValue();
        fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(value)
        })
        .then(res => res.json())
        .then(data => {
            if(data.status === 'success') {
                // Backend automatically signals engine (reload or restart based on changes)
                const action = data.action || 'reload';
                if (action === 'restart') {
                    alert('Configuration saved! Engine restart triggered (stream/model changes detected).');
                } else {
                    alert('Configuration saved! Hot-reload triggered.');
                }
            } else {
                alert('Error saving: ' + data.error);
            }
        })
        .catch(err => {
            alert('Error: ' + err.message);
        });
    });

    // Stream Selection
    const streamSelect = document.getElementById('stream-select');
    const videoImg = document.getElementById('video-stream');
    
    streamSelect.addEventListener('change', function() {
        const streamIndex = this.value;
        
        // Exit drawing mode when switching streams
        if (isDrawingMode) {
            exitDrawingMode();
        }
        
        if (streamIndex === "") {
            videoImg.src = "";
            videoImg.style.display = 'none';
            drawOverlay();
            return;
        }
        
        const config = editor.getValue();
        const stream = config.streams[streamIndex];
        
        if (stream && stream.rtsp_url) {
            videoImg.src = `/video_feed?url=${encodeURIComponent(stream.rtsp_url)}`;
            videoImg.style.display = 'block';
            // Wait for image to load to resize canvas
            videoImg.onload = resizeCanvas;
        }
        updateZoneList();
        updateDoorList();
        drawOverlay();
    });

    // Canvas Interaction - Mouse Down (Start Drag or Add Point)
    canvas.addEventListener('mousedown', function(e) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left);
        const y = (e.clientY - rect.top);
        const normX = x / canvas.width;
        const normY = y / canvas.height;

        // Check if clicking near an existing point
        const hitThreshold = 10; // pixels
        
        // 0. Check if clicking on NEW door arrow tip (for rotation while drawing)
        if (isDrawingMode && drawMode === 'door' && currentPoints.length === 4) {
            const arrowInfo = getArrowEndpoint(currentPoints, currentDoorAngle);
            if (arrowInfo && Math.hypot(arrowInfo.arrowEndX - x, arrowInfo.arrowEndY - y) < hitThreshold + 5) {
                draggingArrow = true;
                return;
            }
        }
        
        // 0b. Check if clicking on EXISTING door arrow tips (for rotation of saved doors)
        const streamIndex = streamSelect.value;
        if (streamIndex !== "") {
            const config = editor.getValue();
            const doors = config.streams[streamIndex].doors || [];
            
            for (let dIdx = 0; dIdx < doors.length; dIdx++) {
                const door = doors[dIdx];
                if (door.points && door.points.length === 4) {
                    const arrowInfo = getArrowEndpoint(door.points, door.normal_angle || 0);
                    if (arrowInfo && Math.hypot(arrowInfo.arrowEndX - x, arrowInfo.arrowEndY - y) < hitThreshold + 5) {
                        draggingArrow = true;
                        draggingArrowDoorIndex = dIdx;
                        return;
                    }
                }
            }
        }

        // 1. Check "New Zone/Door" points (only in drawing mode)
        if (isDrawingMode) {
            for (let i = 0; i < currentPoints.length; i++) {
                const p = denormalize(currentPoints[i]);
                if (Math.hypot(p.x - x, p.y - y) < hitThreshold) {
                    draggingPoint = { type: 'new', index: i };
                    return;
                }
            }
        }

        // 2. Check Existing Zones points (always available for editing)
        if (streamIndex !== "") {
            const config = editor.getValue();
            const zones = config.streams[streamIndex].zones || [];
            
            for (let zIdx = 0; zIdx < zones.length; zIdx++) {
                const points = zones[zIdx].points || [];
                for (let pIdx = 0; pIdx < points.length; pIdx++) {
                    const p = denormalize(points[pIdx]);
                    if (Math.hypot(p.x - x, p.y - y) < hitThreshold) {
                        draggingPoint = { type: 'existing-zone', zoneIndex: zIdx, pointIndex: pIdx };
                        return;
                    }
                }
            }
            
            // 3. Check Existing Doors points
            const doors = config.streams[streamIndex].doors || [];
            
            for (let dIdx = 0; dIdx < doors.length; dIdx++) {
                const points = doors[dIdx].points || [];
                for (let pIdx = 0; pIdx < points.length; pIdx++) {
                    const p = denormalize(points[pIdx]);
                    if (Math.hypot(p.x - x, p.y - y) < hitThreshold) {
                        draggingPoint = { type: 'existing-door', doorIndex: dIdx, pointIndex: pIdx };
                        return;
                    }
                }
            }
        }

        // 4. ONLY add a point if we're in drawing mode and haven't reached 4 points yet
        if (isDrawingMode && currentPoints.length < 4) {
            currentPoints.push([normX, normY]);
            drawOverlay();
            updateButtons();
            updatePointsCount();
            updateDoorRotationControls();
        }
    });

    // Canvas Interaction - Mouse Move (Drag)
    canvas.addEventListener('mousemove', function(e) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left);
        const y = (e.clientY - rect.top);
        const normX = Math.max(0, Math.min(1, x / canvas.width));
        const normY = Math.max(0, Math.min(1, y / canvas.height));
        
        // Handle arrow dragging
        if (draggingArrow) {
            const streamIndex = streamSelect.value;
            
            if (draggingArrowDoorIndex !== null && streamIndex !== "") {
                // Dragging an EXISTING door's arrow
                const config = editor.getValue();
                const door = config.streams[streamIndex].doors[draggingArrowDoorIndex];
                if (door && door.points && door.points.length === 4) {
                    const arrowInfo = getArrowEndpoint(door.points, door.normal_angle || 0);
                    if (arrowInfo) {
                        const dx = x - arrowInfo.cx;
                        const dy = y - arrowInfo.cy;
                        let newAngle = Math.atan2(dy, dx) * 180 / Math.PI;
                        if (newAngle < 0) newAngle += 360;
                        
                        // Update config (visual feedback during drag)
                        config.streams[streamIndex].doors[draggingArrowDoorIndex].normal_angle = newAngle;
                        editor.setValue(config);
                        drawOverlay();
                    }
                }
            } else if (currentPoints.length === 4) {
                // Dragging NEW door's arrow
                const arrowInfo = getArrowEndpoint(currentPoints, currentDoorAngle);
                if (arrowInfo) {
                    const dx = x - arrowInfo.cx;
                    const dy = y - arrowInfo.cy;
                    let newAngle = Math.atan2(dy, dx) * 180 / Math.PI;
                    if (newAngle < 0) newAngle += 360;
                    currentDoorAngle = newAngle;
                    
                    // Update slider
                    document.getElementById('door-angle-slider').value = currentDoorAngle;
                    document.getElementById('door-angle-value').textContent = Math.round(currentDoorAngle) + '°';
                    drawOverlay();
                }
            }
            return;
        }
        
        if (!draggingPoint) return;

        if (draggingPoint.type === 'new') {
            currentPoints[draggingPoint.index] = [normX, normY];
            drawOverlay();
        } else if (draggingPoint.type === 'existing-zone') {
            // Pass drag override for smooth visual feedback
            drawOverlay(draggingPoint, {x: normX, y: normY});
        } else if (draggingPoint.type === 'existing-door') {
            // Pass drag override for smooth visual feedback
            drawOverlay(draggingPoint, {x: normX, y: normY});
        }
    });

    // Canvas Interaction - Mouse Up (End Drag)
    canvas.addEventListener('mouseup', function(e) {
        // End arrow dragging
        if (draggingArrow) {
            draggingArrow = false;
            draggingArrowDoorIndex = null;
            return;
        }
        
        if (!draggingPoint) return;

        const rect = canvas.getBoundingClientRect();
        const normX = Math.max(0, Math.min(1, (e.clientX - rect.left) / canvas.width));
        const normY = Math.max(0, Math.min(1, (e.clientY - rect.top) / canvas.height));
        
        const config = editor.getValue();
        const streamIndex = streamSelect.value;

        if (draggingPoint.type === 'existing-zone') {
            config.streams[streamIndex].zones[draggingPoint.zoneIndex].points[draggingPoint.pointIndex] = {x: normX, y: normY};
            editor.setValue(config);
        } else if (draggingPoint.type === 'existing-door') {
            config.streams[streamIndex].doors[draggingPoint.doorIndex].points[draggingPoint.pointIndex] = {x: normX, y: normY};
            editor.setValue(config);
        }
        
        draggingPoint = null;
        drawOverlay();
    });
    
    // Mouse leave to cancel drag
    canvas.addEventListener('mouseleave', function() {
        draggingPoint = null;
        draggingArrow = false;
        draggingArrowDoorIndex = null;
        drawOverlay();
    });

    document.getElementById('clear-points-btn').addEventListener('click', function() {
        currentPoints = [];
        currentDoorAngle = 0;
        drawOverlay();
        updateButtons();
        updatePointsCount();
        updateDoorRotationControls();
    });

    document.getElementById('add-zone-btn').addEventListener('click', function() {
        const name = document.getElementById('new-zone-name').value.trim();
        if (!name) {
            alert('Please enter a zone name');
            return;
        }
        
        if (currentPoints.length !== 4) {
            alert('Please click 4 points on the video to define the zone');
            return;
        }
        
        const streamIndex = streamSelect.value;
        if (streamIndex === "") {
            alert('Please select a stream first');
            return;
        }

        // Add to editor
        const editorValue = editor.getValue();
        if (!editorValue.streams[streamIndex].zones) {
            editorValue.streams[streamIndex].zones = [];
        }
        
        editorValue.streams[streamIndex].zones.push({
            name: name,
            points: currentPoints.map(p => ({x: p[0], y: p[1]})),
            triggers: ["person"] // Default
        });
        
        editor.setValue(editorValue);
        
        // Exit drawing mode after successful add
        exitDrawingMode();
    });
    
    // Add Door button
    document.getElementById('add-door-btn').addEventListener('click', function() {
        const name = document.getElementById('new-zone-name').value.trim();
        if (!name) {
            alert('Please enter a door name');
            return;
        }
        
        if (currentPoints.length !== 4) {
            alert('Please click 4 points on the video to define the door');
            return;
        }
        
        const streamIndex = streamSelect.value;
        if (streamIndex === "") {
            alert('Please select a stream first');
            return;
        }

        // Add to editor
        const editorValue = editor.getValue();
        if (!editorValue.streams[streamIndex].doors) {
            editorValue.streams[streamIndex].doors = [];
        }
        
        editorValue.streams[streamIndex].doors.push({
            name: name,
            points: currentPoints.map(p => ({x: p[0], y: p[1]})),
            normal_angle: currentDoorAngle, // User-defined angle
            direction_tolerance: 60.0, // Default tolerance
            triggers: ["person"] // Default
        });
        
        editor.setValue(editorValue);
        
        // Exit drawing mode after successful add
        exitDrawingMode();
    });
});

function updateStreamSelect() {
    const select = document.getElementById('stream-select');
    const currentVal = select.value;
    const config = editor.getValue();
    
    // Clear options (keep first)
    while (select.options.length > 1) {
        select.remove(1);
    }
    
    if (config.streams) {
        config.streams.forEach((stream, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.text = stream.name || `Stream ${index + 1}`;
            select.add(option);
        });
    }
    
    select.value = currentVal;
}

function drawOverlay(dragOverride = null, dragPos = null) {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const streamIndex = document.getElementById('stream-select').value;
    
    // 1. Draw Existing Zones
    if (streamIndex !== "" && editor) {
        const config = editor.getValue();
        const zones = config.streams[streamIndex].zones || [];
        
        zones.forEach((zone, zIdx) => {
            const color = getZoneColor(zIdx);
            const points = zone.points || [];
            if (points.length === 0) return;

            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.fillStyle = color + '4D'; // 30% opacity hex approx

            // Handle dragging override
            let renderPoints = points;
            if (dragOverride && dragOverride.type === 'existing-zone' && dragOverride.zoneIndex === zIdx) {
                renderPoints = [...points]; // Shallow copy
                renderPoints[dragOverride.pointIndex] = dragPos;
            }

            if (renderPoints.length > 0) {
                const p0 = denormalize(renderPoints[0]);
                ctx.moveTo(p0.x, p0.y);
                for (let i = 1; i < renderPoints.length; i++) {
                    const p = denormalize(renderPoints[i]);
                    ctx.lineTo(p.x, p.y);
                }
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                // Draw vertices
                ctx.fillStyle = '#FFFFFF';
                for (const p of renderPoints) {
                    const dp = denormalize(p);
                    ctx.beginPath();
                    ctx.arc(dp.x, dp.y, 4, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke(); // White fill, colored stroke
                }
                
                // Draw Label
                const p0_lbl = denormalize(renderPoints[0]);
                ctx.fillStyle = color;
                ctx.font = "12px Arial";
                ctx.fillText(zone.name, p0_lbl.x, p0_lbl.y - 10);
            }
        });
        
        // 2. Draw Existing Doors
        const doors = config.streams[streamIndex].doors || [];
        
        doors.forEach((door, dIdx) => {
            const points = door.points || [];
            if (points.length === 0) return;

            // Handle dragging override
            let renderPoints = points;
            if (dragOverride && dragOverride.type === 'existing-door' && dragOverride.doorIndex === dIdx) {
                renderPoints = [...points];
                renderPoints[dragOverride.pointIndex] = dragPos;
            }

            // Draw door polygon (orange)
            ctx.beginPath();
            ctx.strokeStyle = doorColor;
            ctx.lineWidth = 3;
            ctx.fillStyle = 'rgba(255, 165, 0, 0.3)';

            if (renderPoints.length > 0) {
                const p0 = denormalize(renderPoints[0]);
                ctx.moveTo(p0.x, p0.y);
                for (let i = 1; i < renderPoints.length; i++) {
                    const p = denormalize(renderPoints[i]);
                    ctx.lineTo(p.x, p.y);
                }
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                // Draw vertices
                ctx.fillStyle = '#FFFFFF';
                ctx.strokeStyle = doorColor;
                for (const p of renderPoints) {
                    const dp = denormalize(p);
                    ctx.beginPath();
                    ctx.arc(dp.x, dp.y, 5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                }
                
                // Draw normal arrow (shows "enter" direction)
                if (renderPoints.length >= 2) {
                    drawDoorNormal(renderPoints, door.normal_angle);
                }
                
                // Draw Label
                const p0_lbl = denormalize(renderPoints[0]);
                ctx.fillStyle = doorColor;
                ctx.font = "bold 12px Arial";
                ctx.fillText("[Door] " + door.name, p0_lbl.x, p0_lbl.y - 10);
            }
        });
    }
    
    // 3. Draw New Zone/Door (Current Points) - ONLY when in drawing mode
    if (isDrawingMode && currentPoints.length > 0) {
        const isDrawingDoor = drawMode === 'door';
        ctx.strokeStyle = isDrawingDoor ? doorColor : '#00FF00';
        ctx.lineWidth = isDrawingDoor ? 3 : 2;
        ctx.fillStyle = isDrawingDoor ? 'rgba(255, 165, 0, 0.3)' : 'rgba(0, 255, 0, 0.3)';
        
        ctx.beginPath();
        const p0 = denormalize(currentPoints[0]);
        ctx.moveTo(p0.x, p0.y);
        
        for (let i = 1; i < currentPoints.length; i++) {
            const p = denormalize(currentPoints[i]);
            ctx.lineTo(p.x, p.y);
        }
        
        if (currentPoints.length === 4) {
            ctx.closePath();
            ctx.fill();
            
            // If drawing a door, also show the normal preview
            if (isDrawingDoor) {
                drawDoorNormal(currentPoints, currentDoorAngle);
            }
        }
        
        ctx.stroke();
        
        // Draw vertices with numbers for doors
        ctx.fillStyle = '#FFFF00';
        for (let i = 0; i < currentPoints.length; i++) {
            const dp = denormalize(currentPoints[i]);
            ctx.beginPath();
            ctx.arc(dp.x, dp.y, isDrawingDoor ? 6 : 4, 0, Math.PI * 2);
            ctx.fill();
            
            // Number the points for doors to show order matters
            if (isDrawingDoor) {
                ctx.fillStyle = '#000000';
                ctx.font = "bold 10px Arial";
                ctx.fillText((i + 1).toString(), dp.x - 3, dp.y + 3);
                ctx.fillStyle = '#FFFF00';
            }
        }
    }
}

// Helper to get arrow endpoint for hit testing (without drawing)
function getArrowEndpoint(points, normalAngle) {
    if (points.length < 4) return null;
    
    let cx = 0, cy = 0;
    for (const p of points) {
        const dp = denormalize(p);
        cx += dp.x;
        cy += dp.y;
    }
    cx /= points.length;
    cy /= points.length;
    
    const angleRad = (normalAngle || 0) * Math.PI / 180;
    const nx = Math.cos(angleRad);
    const ny = Math.sin(angleRad);
    const arrowLength = 50;
    
    return {
        cx,
        cy,
        arrowEndX: cx + nx * arrowLength,
        arrowEndY: cy + ny * arrowLength
    };
}

// Draw normal arrow for a door showing "enter" direction
function drawDoorNormal(points, normalAngle) {
    if (points.length < 4) return;
    
    // Calculate center of polygon
    let cx = 0, cy = 0;
    for (const p of points) {
        const dp = denormalize(p);
        cx += dp.x;
        cy += dp.y;
    }
    cx /= points.length;
    cy /= points.length;
    
    // Use the angle directly (simple user-defined direction)
    const angleRad = (normalAngle || 0) * Math.PI / 180;
    const nx = Math.cos(angleRad);
    const ny = Math.sin(angleRad);
    const arrowLength = 50;
    
    // Draw arrow
    const arrowEndX = cx + nx * arrowLength;
    const arrowEndY = cy + ny * arrowLength;
    
    ctx.save();
    ctx.strokeStyle = '#00FF00';
    ctx.fillStyle = '#00FF00';
    ctx.lineWidth = 3;
    
    // Arrow line
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(arrowEndX, arrowEndY);
    ctx.stroke();
    
    // Arrow head
    const headLength = 12;
    const headAngle = Math.PI / 6;
    const angle = Math.atan2(ny, nx);
    
    ctx.beginPath();
    ctx.moveTo(arrowEndX, arrowEndY);
    ctx.lineTo(
        arrowEndX - headLength * Math.cos(angle - headAngle),
        arrowEndY - headLength * Math.sin(angle - headAngle)
    );
    ctx.lineTo(
        arrowEndX - headLength * Math.cos(angle + headAngle),
        arrowEndY - headLength * Math.sin(angle + headAngle)
    );
    ctx.closePath();
    ctx.fill();
    
    // "ENTER" label
    ctx.font = "bold 11px Arial";
    ctx.fillText("ENTER", arrowEndX + 5, arrowEndY + 4);
    
    // Draw a small drag handle circle at arrow end
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(arrowEndX, arrowEndY, 8, 0, Math.PI * 2);
    ctx.stroke();
    
    ctx.restore();
    
    // Return arrow endpoint for hit testing
    return { cx, cy, arrowEndX, arrowEndY };
}

function denormalize(point) {
    // Handle both {x, y} objects and [x, y] arrays
    if (Array.isArray(point)) {
        return {
            x: point[0] * canvas.width,
            y: point[1] * canvas.height
        };
    } else {
        return {
            x: point.x * canvas.width,
            y: point.y * canvas.height
        };
    }
}

// Convert point to normalized {x, y} format for saving
function toPointObject(normX, normY) {
    return { x: normX, y: normY };
}

function updateButtons() {
    const zoneBtn = document.getElementById('add-zone-btn');
    const doorBtn = document.getElementById('add-door-btn');
    const hasEnoughPoints = currentPoints.length === 4;
    const hasName = document.getElementById('new-zone-name').value.trim() !== '';
    
    zoneBtn.disabled = !hasEnoughPoints || !hasName;
    doorBtn.disabled = !hasEnoughPoints || !hasName;
}

// Update the zone list in the right panel
function updateZoneList() {
    const streamIndex = document.getElementById('stream-select').value;
    const zoneListDiv = document.getElementById('zone-list');
    
    if (!zoneListDiv) return;
    
    if (streamIndex === "" || !editor) {
        zoneListDiv.innerHTML = '<p class="text-muted small">Select a stream to see zones</p>';
        return;
    }
    
    const config = editor.getValue();
    const zones = config.streams[streamIndex]?.zones || [];
    
    if (zones.length === 0) {
        zoneListDiv.innerHTML = '<p class="text-muted small">No zones defined. Draw one on the video.</p>';
        return;
    }
    
    let html = '';
    zones.forEach((zone, idx) => {
        const color = getZoneColor(idx);
        html += `
            <div class="zone-list-item" data-zone-index="${idx}">
                <span>
                    <span class="zone-color-dot" style="background-color: ${color};"></span>
                    ${zone.name}
                </span>
                <button class="btn btn-outline-danger btn-sm delete-zone-btn" data-zone-index="${idx}" title="Delete zone">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
    });
    
    zoneListDiv.innerHTML = html;
    
    // Attach delete handlers
    document.querySelectorAll('.delete-zone-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const zoneIdx = parseInt(this.dataset.zoneIndex);
            deleteZone(zoneIdx);
        });
    });
}

function deleteZone(zoneIndex) {
    const streamIndex = document.getElementById('stream-select').value;
    if (streamIndex === "" || !editor) return;
    
    const config = editor.getValue();
    const zoneName = config.streams[streamIndex].zones[zoneIndex]?.name || 'this zone';
    
    if (!confirm(`Delete zone "${zoneName}"?`)) return;
    
    config.streams[streamIndex].zones.splice(zoneIndex, 1);
    editor.setValue(config);
    updateZoneList();
    drawOverlay();
}

// Update the door list in the right panel
function updateDoorList() {
    const streamIndex = document.getElementById('stream-select').value;
    const doorListDiv = document.getElementById('door-list');
    
    if (!doorListDiv) return;
    
    if (streamIndex === "" || !editor) {
        doorListDiv.innerHTML = '<p class="text-muted small">Select a stream to see doors</p>';
        return;
    }
    
    const config = editor.getValue();
    const doors = config.streams[streamIndex]?.doors || [];
    
    if (doors.length === 0) {
        doorListDiv.innerHTML = '<p class="text-muted small">No doors defined. Switch to Door mode and draw one.</p>';
        return;
    }
    
    let html = '';
    doors.forEach((door, idx) => {
        html += `
            <div class="zone-list-item" data-door-index="${idx}">
                <span>
                    <span class="zone-color-dot" style="background-color: ${doorColor};"></span>
                    <i class="fas fa-door-open text-warning"></i> ${door.name}
                </span>
                <button class="btn btn-outline-danger btn-sm delete-door-btn" data-door-index="${idx}" title="Delete door">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
    });
    
    doorListDiv.innerHTML = html;
    
    // Attach delete handlers
    document.querySelectorAll('.delete-door-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const doorIdx = parseInt(this.dataset.doorIndex);
            deleteDoor(doorIdx);
        });
    });
}

function deleteDoor(doorIndex) {
    const streamIndex = document.getElementById('stream-select').value;
    if (streamIndex === "" || !editor) return;
    
    const config = editor.getValue();
    const doorName = config.streams[streamIndex].doors[doorIndex]?.name || 'this door';
    
    if (!confirm(`Delete door "${doorName}"?`)) return;
    
    config.streams[streamIndex].doors.splice(doorIndex, 1);
    editor.setValue(config);
    updateDoorList();
    drawOverlay();
}

// Hide zone points in JSON editor and collapse advanced sections
function customizeEditorLayout() {
    // Wait for editor to fully render
    setTimeout(() => {
        const editorEl = document.getElementById('editor_holder');
        if (!editorEl) return;
        
        // Use MutationObserver to handle dynamically added elements
        const observer = new MutationObserver(() => {
            collapseAdvancedSections();
            showPointsSummary();
        });
        
        observer.observe(editorEl, { childList: true, subtree: true });
        
        // Initial run
        collapseAdvancedSections();
        showPointsSummary();
    }, 200);
}

function showPointsSummary() {
    // Find all zones and add a points summary
    document.querySelectorAll('[data-schemapath]').forEach(el => {
        const path = el.getAttribute('data-schemapath');
        if (!path || !path.match(/\.zones\.\d+$/) || !path.includes('root.streams')) return;
        
        // Skip if already has summary
        if (el.querySelector('.points-summary')) return;
        
        // Extract indices
        const streamMatch = path.match(/streams\.(\d+)/);
        const zoneMatch = path.match(/zones\.(\d+)/);
        if (!streamMatch || !zoneMatch) return;
        
        const streamIdx = parseInt(streamMatch[1]);
        const zoneIdx = parseInt(zoneMatch[1]);
        
        // Get points from editor
        const config = editor.getValue();
        const points = config.streams?.[streamIdx]?.zones?.[zoneIdx]?.points || [];
        
        // Create summary div
        const summary = document.createElement('div');
        summary.className = 'points-summary';
        
        let html = `<div class="points-summary-header">Points (${points.length})</div>`;
        points.forEach((p, i) => {
            const x = (p.x || 0).toFixed(3);
            const y = (p.y || 0).toFixed(3);
            html += `<div class="points-summary-row"><span>${i+1}.</span> x: ${x} &nbsp; y: ${y}</div>`;
        });
        html += `<div class="points-summary-hint">Click on video to add/drag points</div>`;
        summary.innerHTML = html;
        
        // Find where to insert - after name field or at end of card body
        const cardBody = el.querySelector('.card-body');
        if (cardBody) {
            cardBody.appendChild(summary);
        }
    });
}

function collapseAdvancedSections() {
    // Collapse sections by title
    document.querySelectorAll('.je-object__title, .card-title').forEach(el => {
        const text = el.textContent?.toLowerCase() || '';
        const shouldCollapse = text.includes('mqtt') || 
                               text.includes('filtered') ||
                               text.includes('triggers');
        
        if (shouldCollapse) {
            const container = el.closest('.je-object__container, .card');
            if (container) {
                const collapseBtn = container.querySelector('.json-editor-btn-collapse');
                // Check if not already collapsed
                if (collapseBtn && !container.querySelector('.je-object__container.collapsed')) {
                    // Try to collapse
                    const content = container.querySelector('.card-body, .je-object__content');
                    if (content && content.style.display !== 'none') {
                        collapseBtn.click();
                    }
                }
            }
        }
    });
}
