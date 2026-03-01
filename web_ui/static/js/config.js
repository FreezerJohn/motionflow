/**
 * MotionFlow Config Page
 * 
 * Modern, dark-themed configuration interface.
 * - Tabbed sections: Streams, General, MQTT
 * - Stream selection with live preview
 * - Zone/Door drawing on video canvas
 * - localStorage for persisting selected camera across pages
 */

// ============================================================================
// State
// ============================================================================

let config = null;          // Current config object
let selectedStreamIndex = null;  // Currently selected stream index
let hasUnsavedChanges = false;

// Drawing state
let drawMode = null;        // null | 'zone' | 'door'
let currentPoints = [];     // Points being drawn
let currentDoorAngle = 0;   // Door normal angle in degrees

// Canvas interaction state
let canvas, ctx;
let draggingPoint = null;   // { type: 'zone'|'door', itemIndex, pointIndex }
let isDraggingArrow = false;
let draggingArrowIndex = null;

// Zone colors
const zoneColors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
];
const doorColor = '#FFA500';

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async function() {
    // Initialize canvas
    canvas = document.getElementById('video-overlay');
    ctx = canvas.getContext('2d');
    
    // Load config
    try {
        const response = await fetch('/api/config');
        config = await response.json();
        console.log('Config loaded:', config);
    } catch (err) {
        console.error('Failed to load config:', err);
        alert('Failed to load configuration');
        return;
    }
    
    // Populate UI
    populateGeneralSettings();
    populateMqttSettings();
    renderStreamList();
    
    // Restore selected camera from localStorage
    const savedCamera = localStorage.getItem('motionflow_selected_camera');
    if (savedCamera) {
        const idx = config.streams.findIndex(s => s.name === savedCamera);
        if (idx >= 0) {
            selectStream(idx);
        }
    }
    
    // Set up event listeners
    setupTabNavigation();
    setupGeneralSettingsListeners();
    setupMqttSettingsListeners();
    setupStreamEditorListeners();
    setupDrawingListeners();
    setupCanvasListeners();
    
    // Apply button
    document.getElementById('apply-btn').addEventListener('click', saveConfig);
    
    // Add stream button
    document.getElementById('add-stream-btn').addEventListener('click', addNewStream);
});

// ============================================================================
// Tab Navigation
// ============================================================================

function setupTabNavigation() {
    document.querySelectorAll('.settings-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            const tabId = this.dataset.tab;
            
            // Update tab active state
            document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // Update pane visibility
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            document.getElementById(`tab-${tabId}`).classList.add('active');
        });
    });
}

// ============================================================================
// General Settings
// ============================================================================

function populateGeneralSettings() {
    const g = config.general || {};
    
    document.getElementById('cfg-max-fps').value = g.max_fps_per_stream || 15;
}

function setupGeneralSettingsListeners() {
    const fields = [
        'cfg-max-fps'
    ];
    
    fields.forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('change', () => {
            updateGeneralFromUI();
            markUnsaved();
        });
    });
}

function updateGeneralFromUI() {
    config.general = config.general || {};
    config.general.max_fps_per_stream = parseInt(document.getElementById('cfg-max-fps').value);
}

// ============================================================================
// MQTT Settings
// ============================================================================

function populateMqttSettings() {
    const m = config.mqtt || {};
    
    document.getElementById('cfg-mqtt-enabled').checked = m.enabled || false;
    document.getElementById('cfg-mqtt-broker').value = m.broker || '';
    document.getElementById('cfg-mqtt-port').value = m.port || 1883;
    document.getElementById('cfg-mqtt-topic').value = m.base_topic || 'motionflow';
    document.getElementById('cfg-mqtt-user').value = m.username || '';
    document.getElementById('cfg-mqtt-pass').value = m.password || '';
    document.getElementById('cfg-mqtt-qos').value = m.qos || 0;
}

function setupMqttSettingsListeners() {
    const fields = [
        'cfg-mqtt-enabled', 'cfg-mqtt-broker', 'cfg-mqtt-port',
        'cfg-mqtt-topic', 'cfg-mqtt-user', 'cfg-mqtt-pass', 'cfg-mqtt-qos'
    ];
    
    fields.forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('change', () => {
            updateMqttFromUI();
            markUnsaved();
        });
    });
}

function updateMqttFromUI() {
    config.mqtt = config.mqtt || {};
    config.mqtt.enabled = document.getElementById('cfg-mqtt-enabled').checked;
    config.mqtt.broker = document.getElementById('cfg-mqtt-broker').value;
    config.mqtt.port = parseInt(document.getElementById('cfg-mqtt-port').value);
    config.mqtt.base_topic = document.getElementById('cfg-mqtt-topic').value;
    config.mqtt.username = document.getElementById('cfg-mqtt-user').value || null;
    config.mqtt.password = document.getElementById('cfg-mqtt-pass').value || null;
    config.mqtt.qos = parseInt(document.getElementById('cfg-mqtt-qos').value);
}

// ============================================================================
// Stream List
// ============================================================================

function renderStreamList() {
    const container = document.getElementById('stream-list');
    container.innerHTML = '';
    
    config.streams.forEach((stream, index) => {
        const card = document.createElement('div');
        card.className = 'stream-card' + 
            (index === selectedStreamIndex ? ' active' : '') +
            (!stream.enabled ? ' disabled-stream' : '');
        card.dataset.index = index;
        
        const zoneCount = (stream.zones || []).length;
        const doorCount = (stream.doors || []).length;
        
        card.innerHTML = `
            <div class="stream-card-header">
                <span class="stream-card-name">${stream.name || 'Unnamed'}</span>
                <div class="stream-toggle" title="${stream.enabled ? 'Processing active' : 'Processing disabled'}">
                    <span class="toggle-label">${stream.enabled ? 'ON' : 'OFF'}</span>
                    <label class="toggle-switch">
                        <input type="checkbox" ${stream.enabled ? 'checked' : ''} data-stream-index="${index}">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>
            <div class="stream-card-url">${stream.rtsp_url || 'No URL'}</div>
            <div class="stream-card-stats">
                <span><i class="fas fa-vector-square"></i> ${zoneCount} zones</span>
                <span><i class="fas fa-door-open"></i> ${doorCount} doors</span>
            </div>
        `;
        
        // Click on card selects stream (but not on toggle)
        card.addEventListener('click', (e) => {
            if (!e.target.closest('.stream-toggle')) {
                selectStream(index);
            }
        });
        
        // Toggle switch changes enabled state
        const toggle = card.querySelector('input[type="checkbox"]');
        toggle.addEventListener('change', (e) => {
            e.stopPropagation();
            toggleStreamEnabled(index, e.target.checked);
        });
        
        container.appendChild(card);
    });
}

function toggleStreamEnabled(index, enabled) {
    config.streams[index].enabled = enabled;
    renderStreamList();
    markUnsaved();
}

function selectStream(index) {
    selectedStreamIndex = index;
    const stream = config.streams[index];
    
    // Save to localStorage for debug page
    localStorage.setItem('motionflow_selected_camera', stream.name);
    
    // Update stream list UI
    document.querySelectorAll('.stream-card').forEach((card, i) => {
        card.classList.toggle('active', i === index);
    });
    
    // Show video and editor
    document.getElementById('no-stream-msg').style.display = 'none';
    document.getElementById('video-container').style.display = 'block';
    document.getElementById('stream-editor').style.display = 'block';
    
    // Load stream into editor
    populateStreamEditor(stream);
    
    // Start video preview
    startVideoPreview(stream.name);
    
    // Cancel any drawing mode
    exitDrawMode();
}

function addNewStream() {
    const newStream = {
        name: `stream_${config.streams.length + 1}`,
        rtsp_url: '',
        enabled: true,
        mqtt_topic_suffix: '',
        confidence_threshold: 0.6,
        zones: [],
        doors: []
    };
    
    config.streams.push(newStream);
    renderStreamList();
    selectStream(config.streams.length - 1);
    markUnsaved();
}

function deleteCurrentStream() {
    if (selectedStreamIndex === null) return;
    
    const streamName = config.streams[selectedStreamIndex].name;
    if (!confirm(`Delete stream "${streamName}"?`)) return;
    
    config.streams.splice(selectedStreamIndex, 1);
    selectedStreamIndex = null;
    
    // Hide editor
    document.getElementById('no-stream-msg').style.display = 'block';
    document.getElementById('video-container').style.display = 'none';
    document.getElementById('stream-editor').style.display = 'none';
    
    renderStreamList();
    markUnsaved();
}

// ============================================================================
// Stream Editor
// ============================================================================

function populateStreamEditor(stream) {
    document.getElementById('editor-stream-name').textContent = stream.name;
    document.getElementById('stream-name').value = stream.name || '';
    document.getElementById('stream-url').value = stream.rtsp_url || '';
    document.getElementById('stream-enabled').checked = stream.enabled !== false;
    document.getElementById('stream-mqtt-suffix').value = stream.mqtt_topic_suffix || '';
    document.getElementById('stream-confidence').value = stream.confidence_threshold || 0.3;
    
    renderZoneList();
    renderDoorList();
}

function setupStreamEditorListeners() {
    // Stream field changes
    ['stream-name', 'stream-url', 'stream-enabled', 'stream-mqtt-suffix', 'stream-confidence'].forEach(id => {
        document.getElementById(id).addEventListener('change', updateStreamFromUI);
    });
    
    // Delete stream button
    document.getElementById('delete-stream-btn').addEventListener('click', deleteCurrentStream);
    
    // Add zone/door buttons
    document.getElementById('add-zone-btn').addEventListener('click', () => enterDrawMode('zone'));
    document.getElementById('add-door-btn').addEventListener('click', () => enterDrawMode('door'));
    
    // Draw mode controls
    document.getElementById('cancel-draw-btn').addEventListener('click', exitDrawMode);
    document.getElementById('confirm-draw-btn').addEventListener('click', confirmDraw);
    
    // Door angle slider
    document.getElementById('door-angle-slider').addEventListener('input', function() {
        currentDoorAngle = parseInt(this.value);
        document.getElementById('door-angle-value').textContent = `${currentDoorAngle}°`;
        drawOverlay();
    });
}

function updateStreamFromUI() {
    if (selectedStreamIndex === null) return;
    
    const stream = config.streams[selectedStreamIndex];
    stream.name = document.getElementById('stream-name').value;
    stream.rtsp_url = document.getElementById('stream-url').value;
    stream.enabled = document.getElementById('stream-enabled').checked;
    stream.mqtt_topic_suffix = document.getElementById('stream-mqtt-suffix').value;
    stream.confidence_threshold = parseFloat(document.getElementById('stream-confidence').value);
    
    // Update stream list display
    renderStreamList();
    document.getElementById('editor-stream-name').textContent = stream.name;
    
    // Update localStorage
    localStorage.setItem('motionflow_selected_camera', stream.name);
    
    markUnsaved();
}

// ============================================================================
// Zone/Door Lists
// ============================================================================

function renderZoneList() {
    const container = document.getElementById('zone-items');
    const stream = config.streams[selectedStreamIndex];
    const zones = stream?.zones || [];
    
    if (zones.length === 0) {
        container.innerHTML = '<div class="empty-list">No zones defined</div>';
        return;
    }
    
    container.innerHTML = zones.map((zone, i) => `
        <div class="zone-item" data-index="${i}">
            <span>
                <span class="zone-color" style="background: ${zoneColors[i % zoneColors.length]}"></span>
                ${zone.name}
            </span>
            <span class="item-actions">
                <button class="delete-btn" onclick="deleteZone(${i})">
                    <i class="fas fa-trash"></i>
                </button>
            </span>
        </div>
    `).join('');
}

function renderDoorList() {
    const container = document.getElementById('door-items');
    const stream = config.streams[selectedStreamIndex];
    const doors = stream?.doors || [];
    
    if (doors.length === 0) {
        container.innerHTML = '<div class="empty-list">No doors defined</div>';
        return;
    }
    
    container.innerHTML = doors.map((door, i) => `
        <div class="door-item" data-index="${i}">
            <span>
                <i class="fas fa-door-open text-warning me-2"></i>
                ${door.name}
            </span>
            <span class="item-actions">
                <button class="delete-btn" onclick="deleteDoor(${i})">
                    <i class="fas fa-trash"></i>
                </button>
            </span>
        </div>
    `).join('');
}

function deleteZone(index) {
    const stream = config.streams[selectedStreamIndex];
    const zoneName = stream.zones[index].name;
    if (!confirm(`Delete zone "${zoneName}"?`)) return;
    
    stream.zones.splice(index, 1);
    renderZoneList();
    drawOverlay();
    markUnsaved();
}

function deleteDoor(index) {
    const stream = config.streams[selectedStreamIndex];
    const doorName = stream.doors[index].name;
    if (!confirm(`Delete door "${doorName}"?`)) return;
    
    stream.doors.splice(index, 1);
    renderDoorList();
    drawOverlay();
    markUnsaved();
}

// ============================================================================
// Drawing Mode
// ============================================================================

function enterDrawMode(mode) {
    if (selectedStreamIndex === null) {
        alert('Please select a stream first');
        return;
    }
    
    drawMode = mode;
    currentPoints = [];
    currentDoorAngle = 0;
    
    // Update UI
    document.getElementById('draw-mode-banner').style.display = 'flex';
    document.getElementById('draw-mode-text').textContent = 
        mode === 'zone' ? 'Drawing new zone...' : 'Drawing new door...';
    document.getElementById('points-count').textContent = '0/4';
    document.getElementById('new-item-name').value = '';
    document.getElementById('confirm-draw-btn').disabled = true;
    
    // Show door angle control only for doors
    document.getElementById('door-angle-control').style.display = 
        mode === 'door' ? 'flex' : 'none';
    document.getElementById('door-angle-slider').value = 0;
    document.getElementById('door-angle-value').textContent = '0°';
    
    canvas.style.cursor = 'crosshair';
    drawOverlay();
}

function exitDrawMode() {
    drawMode = null;
    currentPoints = [];
    
    document.getElementById('draw-mode-banner').style.display = 'none';
    document.getElementById('door-angle-control').style.display = 'none';
    canvas.style.cursor = 'default';
    
    drawOverlay();
}

function confirmDraw() {
    const name = document.getElementById('new-item-name').value.trim();
    if (!name) {
        alert('Please enter a name');
        return;
    }
    
    if (currentPoints.length !== 4) {
        alert('Please draw exactly 4 points');
        return;
    }
    
    const stream = config.streams[selectedStreamIndex];
    const pointsData = currentPoints.map(p => ({ x: p.x, y: p.y }));
    
    if (drawMode === 'zone') {
        stream.zones = stream.zones || [];
        stream.zones.push({
            name: name,
            points: pointsData,
            triggers: ['person']
        });
        renderZoneList();
    } else if (drawMode === 'door') {
        stream.doors = stream.doors || [];
        stream.doors.push({
            name: name,
            points: pointsData,
            normal_angle: currentDoorAngle,
            direction_tolerance: 60,
            triggers: ['person']
        });
        renderDoorList();
    }
    
    exitDrawMode();
    markUnsaved();
}

function setupDrawingListeners() {
    document.getElementById('new-item-name').addEventListener('input', function() {
        const hasName = this.value.trim().length > 0;
        const hasPoints = currentPoints.length === 4;
        document.getElementById('confirm-draw-btn').disabled = !(hasName && hasPoints);
    });
}

// ============================================================================
// Canvas / Video
// ============================================================================

let videoLoaded = false;
let videoNaturalWidth = 0;
let videoNaturalHeight = 0;

let videoConnectTimer = null;

function showVideoStatus(state) {
    // state: 'loading' | 'error' | 'connected'
    const overlay = document.getElementById('video-status-overlay');
    const loading = document.getElementById('video-loading');
    const error = document.getElementById('video-error');
    
    loading.style.display = state === 'loading' ? '' : 'none';
    error.style.display = state === 'error' ? '' : 'none';
    overlay.classList.toggle('hidden', state === 'connected');
}

function startVideoPreview(streamName) {
    const img = document.getElementById('video-stream');
    const stream = config.streams.find(s => s.name === streamName);
    
    // Reset state
    videoLoaded = false;
    canvas.style.display = 'none';
    
    // Clear any pending timeout
    if (videoConnectTimer) {
        clearTimeout(videoConnectTimer);
        videoConnectTimer = null;
    }
    
    if (stream && stream.rtsp_url) {
        // Immediately black out and show loading
        img.src = '';
        showVideoStatus('loading');
        
        // Start the new stream
        img.src = `/video_feed?url=${encodeURIComponent(stream.rtsp_url)}`;
        
        // Timeout: if no frame arrives within 10s, show error
        videoConnectTimer = setTimeout(() => {
            if (!videoLoaded) {
                showVideoStatus('error');
            }
        }, 10000);
        
        img.onload = function() {
            // Store natural dimensions (first frame)
            if (!videoLoaded) {
                videoNaturalWidth = img.naturalWidth;
                videoNaturalHeight = img.naturalHeight;
                // First frame arrived — clear timeout and hide overlay
                if (videoConnectTimer) {
                    clearTimeout(videoConnectTimer);
                    videoConnectTimer = null;
                }
                showVideoStatus('connected');
            }
            videoLoaded = true;
            
            // Calculate and apply sizing
            requestAnimationFrame(() => {
                resizeVideoAndCanvas();
                canvas.style.display = 'block';
                drawOverlay();
            });
        };
        
        img.onerror = function() {
            console.log('No video preview available for', streamName);
            videoLoaded = false;
            if (videoConnectTimer) {
                clearTimeout(videoConnectTimer);
                videoConnectTimer = null;
            }
            showVideoStatus('error');
        };
    } else {
        img.src = '';
        showVideoStatus('loading');
    }
}

function resizeVideoAndCanvas() {
    if (!videoLoaded || !videoNaturalWidth || !videoNaturalHeight) {
        return;
    }
    
    const img = document.getElementById('video-stream');
    const videoSection = document.getElementById('video-section');
    
    // Get available space
    const maxWidth = videoSection.clientWidth;
    const maxHeight = videoSection.clientHeight;
    
    // Calculate display size preserving aspect ratio
    const imgAspect = videoNaturalWidth / videoNaturalHeight;
    const containerAspect = maxWidth / maxHeight;
    
    let displayWidth, displayHeight;
    if (imgAspect > containerAspect) {
        // Image is wider - fit to width
        displayWidth = maxWidth;
        displayHeight = maxWidth / imgAspect;
    } else {
        // Image is taller - fit to height
        displayHeight = maxHeight;
        displayWidth = maxHeight * imgAspect;
    }
    
    // Apply size to image
    img.style.width = displayWidth + 'px';
    img.style.height = displayHeight + 'px';
    
    // Match canvas to image
    canvas.width = Math.round(displayWidth);
    canvas.height = Math.round(displayHeight);
    canvas.style.width = displayWidth + 'px';
    canvas.style.height = displayHeight + 'px';
}

function setupCanvasListeners() {
    canvas.addEventListener('click', onCanvasClick);
    canvas.addEventListener('mousedown', onCanvasMouseDown);
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    canvas.addEventListener('mouseup', onCanvasMouseUp);
    
    // Resize observer - resize video and canvas when container changes
    const resizeObserver = new ResizeObserver(() => {
        if (videoLoaded) {
            resizeVideoAndCanvas();
            drawOverlay();
        }
    });
    resizeObserver.observe(document.getElementById('video-section'));
}

function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) / canvas.width,
        y: (e.clientY - rect.top) / canvas.height
    };
}

function onCanvasClick(e) {
    if (!drawMode) return;
    if (currentPoints.length >= 4) return;
    
    const coords = getCanvasCoords(e);
    currentPoints.push(coords);
    
    document.getElementById('points-count').textContent = `${currentPoints.length}/4`;
    
    // Enable confirm button if we have 4 points and a name
    const hasName = document.getElementById('new-item-name').value.trim().length > 0;
    document.getElementById('confirm-draw-btn').disabled = !(currentPoints.length === 4 && hasName);
    
    drawOverlay();
}

function onCanvasMouseDown(e) {
    if (drawMode) return; // Don't drag while drawing
    
    const coords = getCanvasCoords(e);
    const stream = config.streams[selectedStreamIndex];
    if (!stream) return;
    
    // Check if clicking on an existing point
    const hitRadius = 15 / canvas.width; // 15px hit area
    
    // Check zones
    (stream.zones || []).forEach((zone, zoneIdx) => {
        (zone.points || []).forEach((pt, ptIdx) => {
            const dx = coords.x - pt.x;
            const dy = coords.y - pt.y;
            if (Math.sqrt(dx*dx + dy*dy) < hitRadius) {
                draggingPoint = { type: 'zone', itemIndex: zoneIdx, pointIndex: ptIdx };
            }
        });
    });
    
    // Check doors
    (stream.doors || []).forEach((door, doorIdx) => {
        (door.points || []).forEach((pt, ptIdx) => {
            const dx = coords.x - pt.x;
            const dy = coords.y - pt.y;
            if (Math.sqrt(dx*dx + dy*dy) < hitRadius) {
                draggingPoint = { type: 'door', itemIndex: doorIdx, pointIndex: ptIdx };
            }
        });
    });
}

function onCanvasMouseMove(e) {
    if (!draggingPoint) return;
    
    const coords = getCanvasCoords(e);
    const stream = config.streams[selectedStreamIndex];
    
    if (draggingPoint.type === 'zone') {
        const zone = stream.zones[draggingPoint.itemIndex];
        zone.points[draggingPoint.pointIndex].x = Math.max(0, Math.min(1, coords.x));
        zone.points[draggingPoint.pointIndex].y = Math.max(0, Math.min(1, coords.y));
    } else if (draggingPoint.type === 'door') {
        const door = stream.doors[draggingPoint.itemIndex];
        door.points[draggingPoint.pointIndex].x = Math.max(0, Math.min(1, coords.x));
        door.points[draggingPoint.pointIndex].y = Math.max(0, Math.min(1, coords.y));
    }
    
    drawOverlay();
}

function onCanvasMouseUp(e) {
    if (draggingPoint) {
        markUnsaved();
    }
    draggingPoint = null;
}

// ============================================================================
// Drawing Overlay
// ============================================================================

function drawOverlay() {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const stream = selectedStreamIndex !== null ? config.streams[selectedStreamIndex] : null;
    if (!stream) return;
    
    // Draw existing zones
    (stream.zones || []).forEach((zone, i) => {
        drawPolygon(zone.points, zoneColors[i % zoneColors.length], zone.name);
    });
    
    // Draw existing doors
    (stream.doors || []).forEach((door, i) => {
        drawPolygon(door.points, doorColor, door.name);
        drawDoorTripwire(door.points);
        drawDoorArrow(door.points, door.normal_angle);
    });
    
    // Draw current points being drawn
    if (drawMode && currentPoints.length > 0) {
        const color = drawMode === 'zone' ? '#00FF00' : doorColor;
        drawPolygon(currentPoints, color, '', true);
        
        if (drawMode === 'door' && currentPoints.length === 4) {
            drawDoorTripwire(currentPoints);
            drawDoorArrow(currentPoints, currentDoorAngle);
        }
    }
}

function drawPolygon(points, color, label, isDrawing = false) {
    if (!points || points.length === 0) return;
    
    ctx.beginPath();
    points.forEach((pt, i) => {
        const x = pt.x * canvas.width;
        const y = pt.y * canvas.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    
    if (points.length > 2) {
        ctx.closePath();
        ctx.fillStyle = color + '33'; // 20% opacity
        ctx.fill();
    }
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw points
    points.forEach((pt, i) => {
        const x = pt.x * canvas.width;
        const y = pt.y * canvas.height;
        
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = isDrawing ? '#00FF00' : color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
    
    // Draw label
    if (label && points.length > 0) {
        const cx = points.reduce((s, p) => s + p.x, 0) / points.length * canvas.width;
        const cy = points.reduce((s, p) => s + p.y, 0) / points.length * canvas.height;
        
        ctx.font = '14px sans-serif';
        ctx.fillStyle = '#fff';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 3;
        ctx.textAlign = 'center';
        ctx.strokeText(label, cx, cy);
        ctx.fillText(label, cx, cy);
    }
}

function drawDoorTripwire(points) {
    if (!points || points.length < 4) return;

    // Bottom edge = two points with highest Y (normalized coords)
    const sorted = [...points].sort((a, b) => b.y - a.y);
    const x1 = sorted[0].x * canvas.width;
    const y1 = sorted[0].y * canvas.height;
    const x2 = sorted[1].x * canvas.width;
    const y2 = sorted[1].y * canvas.height;

    // Thick yellow tripwire line
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = '#FFFF00';
    ctx.lineWidth = 4;
    ctx.stroke();

    // Endpoint circles
    for (const [cx, cy] of [[x1, y1], [x2, y2]]) {
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#FFFF00';
        ctx.fill();
    }
}

function drawDoorArrow(points, angle) {
    if (!points || points.length < 4) return;
    
    // Calculate center
    const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
    const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
    
    const centerX = cx * canvas.width;
    const centerY = cy * canvas.height;
    
    // Arrow length
    const arrowLen = 40;
    const rad = angle * Math.PI / 180;
    
    const endX = centerX + Math.cos(rad) * arrowLen;
    const endY = centerY + Math.sin(rad) * arrowLen;
    
    // Draw arrow line
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Draw arrowhead
    const headLen = 12;
    const headAngle = 0.5;
    
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(
        endX - headLen * Math.cos(rad - headAngle),
        endY - headLen * Math.sin(rad - headAngle)
    );
    ctx.moveTo(endX, endY);
    ctx.lineTo(
        endX - headLen * Math.cos(rad + headAngle),
        endY - headLen * Math.sin(rad + headAngle)
    );
    ctx.stroke();
    
    // Circle at center
    ctx.beginPath();
    ctx.arc(centerX, centerY, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#00FF00';
    ctx.fill();
}

// ============================================================================
// Save / Unsaved Changes
// ============================================================================

function markUnsaved() {
    hasUnsavedChanges = true;
    document.getElementById('apply-btn').style.display = 'inline-block';
}

async function saveConfig() {
    // Update all settings from UI first
    updateGeneralFromUI();
    updateMqttFromUI();
    
    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            throw new Error('Save failed');
        }
        
        hasUnsavedChanges = false;
        
        // Show brief success indication
        const btn = document.getElementById('apply-btn');
        btn.innerHTML = '<i class="fas fa-check"></i> Applied!';
        btn.classList.remove('btn-success');
        btn.classList.add('btn-outline-success');
        setTimeout(() => {
            btn.innerHTML = '<i class="fas fa-check"></i> Apply';
            btn.classList.remove('btn-outline-success');
            btn.classList.add('btn-success');
            btn.style.display = 'none';
        }, 1500);
        
    } catch (err) {
        console.error('Save failed:', err);
        alert('Failed to save configuration: ' + err.message);
    }
}

// Warn before leaving with unsaved changes
window.addEventListener('beforeunload', function(e) {
    if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = '';
    }
});

// ============================================================================
// Engine Status Polling
// ============================================================================

let engineRunning = false;

function checkEngineStatus() {
    fetch('/api/engine/status')
        .then(res => res.json())
        .then(data => {
            engineRunning = data.running;
            const dot = document.getElementById('engine-status-dot');
            const label = document.getElementById('engine-status-label');
            const banner = document.getElementById('engine-not-running-msg');
            const noStreamMsg = document.getElementById('no-stream-msg');

            if (data.running) {
                if (dot) { dot.className = 'running'; dot.title = 'Engine is running'; }
                if (label) { label.textContent = 'running'; label.style.color = '#28a745'; }
                if (banner) banner.style.display = 'none';
                // Restore no-stream message visibility if no stream selected
                if (noStreamMsg && selectedStreamIndex === null) {
                    noStreamMsg.style.display = '';
                }
            } else {
                if (dot) { dot.className = 'stopped'; dot.title = 'Engine is not running'; }
                if (label) { label.textContent = 'not running'; label.style.color = '#dc3545'; }
                // Show engine-not-running banner when no stream is selected (default view)
                if (banner && selectedStreamIndex === null) {
                    banner.style.display = '';
                    if (noStreamMsg) noStreamMsg.style.display = 'none';
                }
            }
        })
        .catch(() => {
            const dot = document.getElementById('engine-status-dot');
            const label = document.getElementById('engine-status-label');
            if (dot) { dot.className = 'stopped'; dot.title = 'Status unknown'; }
            if (label) { label.textContent = 'unknown'; label.style.color = '#ffc107'; }
        });
}

// Check immediately, then every 5 seconds
checkEngineStatus();
setInterval(checkEngineStatus, 5000);
