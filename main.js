import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- CONFIGURATION ---
const NEURON_COUNT = 16384;
const WS_URL = `ws://${window.location.hostname}:8003/ws/${Math.random().toString(36).substring(7)}`;

// Color Palette - purple/cyan theme
const COLORS = {
    EMPTY: new THREE.Color(0x111118),
    LOW: new THREE.Color(0x3b1d8e),
    MID: new THREE.Color(0x7c3aed),
    HIGH: new THREE.Color(0x06b6d4),
    PEAK: new THREE.Color(0xffffff)
};

// --- THREE.JS SETUP ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

const container = document.getElementById('canvas-container');

let width = container.clientWidth || window.innerWidth * 0.7;
let height = container.clientHeight || window.innerHeight;

const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
camera.position.z = 25;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(width, height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.3;

// Robust Resizing
const resizeObserver = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
});
resizeObserver.observe(container);

// Hide loading overlay
setTimeout(() => {
    const loader = document.getElementById('loading-overlay');
    if (loader) {
        loader.style.opacity = '0';
        setTimeout(() => loader.remove(), 500);
    }
}, 100);

// --- NEURON SPHERE ---
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(NEURON_COUNT * 3);
const colors = new Float32Array(NEURON_COUNT * 3);
const sizes = new Float32Array(NEURON_COUNT);

const phi = Math.PI * (3 - Math.sqrt(5));

for (let i = 0; i < NEURON_COUNT; i++) {
    const y = 1 - (i / (NEURON_COUNT - 1)) * 2;
    const radiusAtY = Math.sqrt(1 - y * y);
    const theta = phi * i;

    positions[i * 3] = Math.cos(theta) * radiusAtY * 10;
    positions[i * 3 + 1] = y * 10;
    positions[i * 3 + 2] = Math.sin(theta) * radiusAtY * 10;

    colors[i * 3] = COLORS.EMPTY.r;
    colors[i * 3 + 1] = COLORS.EMPTY.g;
    colors[i * 3 + 2] = COLORS.EMPTY.b;

    sizes[i] = 0.15;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

const material = new THREE.PointsMaterial({
    size: 0.15,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending
});

const points = new THREE.Points(geometry, material);
scene.add(points);

// --- INTERACTION ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let selectedId = null;

function onMouseClick(event) {
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(points);

    if (intersects.length > 0) {
        selectedId = intersects[0].index;
        updateDetails(selectedId);
    }
}

container.addEventListener('click', onMouseClick);

// --- MEMORY VISUALIZATION LOGIC ---
let currentMemory = new Float32Array(NEURON_COUNT);
let targetMemory = new Float32Array(NEURON_COUNT);

// Track top tokens per neuron: neuronTokenScores[neuronId] = [{token, score}, ...]
const neuronTokenScores = new Map();

function getColorForStrength(strength) {
    if (strength <= 0) return COLORS.EMPTY;
    if (strength < 0.2) return COLORS.EMPTY.clone().lerp(COLORS.LOW, strength / 0.2);
    if (strength < 0.5) return COLORS.LOW.clone().lerp(COLORS.MID, (strength - 0.2) / 0.3);
    if (strength < 0.8) return COLORS.MID.clone().lerp(COLORS.HIGH, (strength - 0.5) / 0.3);
    return COLORS.HIGH.clone().lerp(COLORS.PEAK, (strength - 0.8) / 0.2);
}

function updateVisuals() {
    const colorAttr = geometry.attributes.color;

    for (let i = 0; i < NEURON_COUNT; i++) {
        currentMemory[i] += (targetMemory[i] - currentMemory[i]) * 0.1;

        const color = getColorForStrength(currentMemory[i]);
        colorAttr.array[i * 3] = color.r;
        colorAttr.array[i * 3 + 1] = color.g;
        colorAttr.array[i * 3 + 2] = color.b;
    }
    colorAttr.needsUpdate = true;

    if (selectedId !== null) {
        updateDetails(selectedId);
    }
}

function updateDetails(id) {
    document.getElementById('no-selection').classList.add('hidden');
    document.getElementById('neuron-details').classList.remove('hidden');

    document.getElementById('detail-id').innerText = `#${id}`;
    const memVal = currentMemory[id].toFixed(4);
    document.getElementById('detail-memory-val').innerText = memVal;
    document.getElementById('detail-memory-bar').style.width = `${Math.min(100, currentMemory[id] * 20)}%`;

    const statusEl = document.getElementById('detail-status');
    const isActive = currentMemory[id] > 0.01;
    statusEl.innerText = isActive ? "Active" : "Dormant";
    statusEl.style.background = isActive ? 'rgba(6, 182, 212, 0.1)' : 'rgba(124, 58, 237, 0.1)';
    statusEl.style.color = isActive ? '#06b6d4' : '#7c3aed';
    statusEl.style.borderColor = isActive ? 'rgba(6, 182, 212, 0.2)' : 'rgba(124, 58, 237, 0.2)';

    // Show top 3 tokens
    const topTokensEl = document.getElementById('detail-top-tokens');
    const scores = neuronTokenScores.get(id);
    if (scores && scores.length > 0) {
        const top3 = scores.slice(0, 3);
        topTokensEl.innerHTML = top3.map(s => {
            const displayToken = s.token === ' ' ? '␣' : s.token === '\n' ? '↵' : s.token;
            return `<span class="top-token-chip">"${displayToken}" <span class="token-score">${s.score.toFixed(2)}</span></span>`;
        }).join('');
    } else {
        topTokensEl.innerHTML = '<span class="top-token-empty">—</span>';
    }
}

// --- CHAT MESSAGE HELPERS ---
const chatHistory = document.getElementById('chat-history');
let currentAiBubble = null;

function getTimeString() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function clearWelcome() {
    const welcome = chatHistory.querySelector('.chat-welcome');
    if (welcome) welcome.remove();
}

function createUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row message-row--user';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar message-avatar--user';
    avatar.textContent = 'U';

    const content = document.createElement('div');

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble message-bubble--user';
    bubble.textContent = text;

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = getTimeString();

    content.appendChild(bubble);
    content.appendChild(time);
    row.appendChild(avatar);
    row.appendChild(content);

    return row;
}

function createAiMessageRow() {
    const row = document.createElement('div');
    row.className = 'message-row message-row--ai';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar message-avatar--ai';
    avatar.innerHTML = `<img src="icon.png" alt="AI" style="width: 100%; height: 100%; object-fit: contain; border-radius: 50%;">`;

    const content = document.createElement('div');
    content.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble message-bubble--ai';

    // Typing indicator
    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    bubble.appendChild(typing);

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = getTimeString();

    content.appendChild(bubble);
    content.appendChild(time);
    row.appendChild(avatar);
    row.appendChild(content);

    return { row, bubble };
}

function appendToken(token) {
    clearWelcome();

    if (!currentAiBubble) {
        const { row, bubble } = createAiMessageRow();
        chatHistory.appendChild(row);
        currentAiBubble = bubble;
    }

    // Remove typing indicator if still present
    const typing = currentAiBubble.querySelector('.typing-indicator');
    if (typing) typing.remove();

    // Get or create the text container span
    let textContainer = currentAiBubble.querySelector('.ai-text-content');
    if (!textContainer) {
        textContainer = document.createElement('span');
        textContainer.className = 'ai-text-content';
        currentAiBubble.appendChild(textContainer);
    }

    // Append token directly as text (no separate span per character)
    textContainer.textContent += token;

    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function appendError(err) {
    const div = document.createElement('div');
    div.className = 'chat-error';
    div.textContent = err;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function resetChatUI() {
    chatHistory.innerHTML = '';
    const welcome = document.createElement('div');
    welcome.className = 'chat-welcome';
    welcome.innerHTML = `
        <div class="welcome-orb">
            <div class="orb-ring"></div>
            <div class="orb-core"></div>
        </div>
        <h3 class="welcome-title">Neural Explorer</h3>
        <p class="welcome-desc">Type a concept or prompt to see how neurons fire and form memory patterns in real-time.</p>
        <div class="welcome-chips">
            <button class="chip" data-prompt="The meaning of life">The meaning of life</button>
            <button class="chip" data-prompt="How does memory work">How does memory work</button>
            <button class="chip" data-prompt="Once upon a time">Once upon a time</button>
        </div>
    `;
    chatHistory.appendChild(welcome);
    bindChips();
    currentAiBubble = null;
}

// --- WEBSOCKET COMMUNICATION ---
let socket;
let reconnectInterval = 1000;
let isGenerating = false;

function connect() {
    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
        console.log('WS connected');
        reconnectInterval = 1000;
        const indicator = document.getElementById('fps-counter');
        if (indicator) indicator.innerText = 'Connected';
        const dot = document.querySelector('.status-dot');
        if (dot) {
            dot.style.background = '#14b8a6';
            dot.style.boxShadow = '0 0 6px rgba(20, 184, 166, 0.5)';
        }
    };

    socket.onmessage = (event) => {
        const data = jsonParseSafe(event.data);
        if (!data) return;

        if (data.error) {
            console.error('Server error:', data.error);
            appendError(data.error);
            isGenerating = false;
            return;
        }

        if (data.status === 'reset') {
            targetMemory.fill(0);
            neuronTokenScores.clear();
            isGenerating = false;
            return;
        }

        if (data.status === 'done') {
            // Generation finished - finalize the current AI bubble
            currentAiBubble = null;
            isGenerating = false;
            return;
        }

        if (data.token) {
            appendToken(data.token);

            // Record which neurons this token fired
            if (data.top_neurons && data.token) {
                for (const n of data.top_neurons) {
                    if (n.value <= 0) continue;
                    if (!neuronTokenScores.has(n.id)) {
                        neuronTokenScores.set(n.id, []);
                    }
                    const scores = neuronTokenScores.get(n.id);
                    // Check if token already recorded, update if higher
                    const existing = scores.find(s => s.token === data.token);
                    if (existing) {
                        if (n.value > existing.score) existing.score = n.value;
                    } else {
                        scores.push({ token: data.token, score: n.value });
                    }
                    // Keep sorted descending, cap at 10
                    scores.sort((a, b) => b.score - a.score);
                    if (scores.length > 10) scores.length = 10;
                }
            }
        }

        if (data.memory) {
            for (let i = 0; i < NEURON_COUNT; i++) {
                targetMemory[i] = data.memory[i];
            }
        }
    };

    socket.onclose = () => {
        const indicator = document.getElementById('fps-counter');
        if (indicator) indicator.innerText = 'Reconnecting...';
        const dot = document.querySelector('.status-dot');
        if (dot) {
            dot.style.background = '#ef4444';
            dot.style.boxShadow = '0 0 6px rgba(239, 68, 68, 0.5)';
        }
        console.log(`WS closed, retrying in ${reconnectInterval}ms...`);
        setTimeout(connect, reconnectInterval);
        reconnectInterval = Math.min(30000, reconnectInterval * 2);
    };

    socket.onerror = (err) => {
        console.error('WS error:', err);
    };
}

function jsonParseSafe(str) {
    try { return JSON.parse(str); }
    catch (e) { return null; }
}

// --- UI EVENTS ---
const input = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const resetBtn = document.getElementById('reset-btn');
const resetCacheBtn = document.getElementById('reset-cache-btn');

function sendMessage() {
    const text = input.value.trim();
    if (text && socket && socket.readyState === WebSocket.OPEN && !isGenerating) {
        clearWelcome();

        // Finalize any previous AI bubble
        currentAiBubble = null;
        isGenerating = true;

        // Add user message bubble
        const userMsg = createUserMessage(text);
        chatHistory.appendChild(userMsg);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        socket.send(JSON.stringify({ type: "prompt", text: text }));
        input.value = '';
    }
}

sendBtn.addEventListener('click', sendMessage);
input.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

resetBtn.addEventListener('click', () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "reset_session" }));
        resetChatUI();
        targetMemory.fill(0);
    }
});

resetCacheBtn.addEventListener('click', () => {
    if (confirm("This will clear the model's global memory cache. Continue?")) {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "reset" }));
            resetChatUI();
            targetMemory.fill(0);
        }
    }
});

// Chip click handling
function bindChips() {
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-prompt');
            if (prompt) {
                input.value = prompt;
                sendMessage();
            }
        });
    });
}
bindChips();

// --- ANIMATION LOOP ---
const fpsText = document.getElementById('fps-counter');
let lastFrameTime = performance.now();
let frames = 0;

function animate() {
    requestAnimationFrame(animate);

    const now = performance.now();
    frames++;
    if (now > lastFrameTime + 1000) {
        const fps = Math.round((frames * 1000) / (now - lastFrameTime));
        if (socket && socket.readyState === WebSocket.OPEN) {
            fpsText.innerText = `${fps} FPS`;
        }
        lastFrameTime = now;
        frames = 0;

        const loader = document.getElementById('loading-overlay');
        if (loader && loader.style.opacity !== '0') {
            loader.style.opacity = '0';
            setTimeout(() => loader.remove(), 500);
        }
    }

    controls.update();
    updateVisuals();
    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
});

connect();
animate();

// --- VIZ CONTROLS ---
const zoomSlider = document.getElementById('zoom-slider');
const zoomInBtn = document.getElementById('zoom-in-btn');
const zoomOutBtn = document.getElementById('zoom-out-btn');
const autoRotateBtn = document.getElementById('auto-rotate-btn');
const resetViewBtn = document.getElementById('reset-view-btn');
const topViewBtn = document.getElementById('top-view-btn');
const sideViewBtn = document.getElementById('side-view-btn');

// Sync slider with camera
function updateZoomSlider() {
    zoomSlider.value = camera.position.length();
}

zoomSlider.addEventListener('input', () => {
    const dist = parseFloat(zoomSlider.value);
    const dir = camera.position.clone().normalize();
    camera.position.copy(dir.multiplyScalar(dist));
});

zoomInBtn.addEventListener('click', () => {
    const dist = Math.max(8, camera.position.length() - 5);
    const dir = camera.position.clone().normalize();
    camera.position.copy(dir.multiplyScalar(dist));
    zoomSlider.value = dist;
});

zoomOutBtn.addEventListener('click', () => {
    const dist = Math.min(80, camera.position.length() + 5);
    const dir = camera.position.clone().normalize();
    camera.position.copy(dir.multiplyScalar(dist));
    zoomSlider.value = dist;
});

// Freeze button - stops all motion
const freezeBtn = document.getElementById('freeze-btn');
let isFrozen = false;
let wasAutoRotating = false;

const PAUSE_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
const PLAY_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;

freezeBtn.addEventListener('click', () => {
    isFrozen = !isFrozen;
    if (isFrozen) {
        // Save auto-rotate state before freezing
        wasAutoRotating = controls.autoRotate;
        controls.autoRotate = false;
        controls.enableRotate = false;
        controls.enableZoom = false;
        controls.enablePan = false;
        controls.enableDamping = false;
        freezeBtn.classList.add('active');
        freezeBtn.innerHTML = PLAY_ICON;
        freezeBtn.title = 'Resume Globe';
        autoRotateBtn.classList.remove('active');
    } else {
        // Restore everything
        controls.enableRotate = true;
        controls.enableZoom = true;
        controls.enablePan = true;
        controls.enableDamping = true;
        controls.autoRotate = wasAutoRotating;
        autoRotateBtn.classList.toggle('active', wasAutoRotating);
        freezeBtn.classList.remove('active');
        freezeBtn.innerHTML = PAUSE_ICON;
        freezeBtn.title = 'Pause Globe';
    }
});

// Auto-rotate toggle
autoRotateBtn.classList.add('active');
autoRotateBtn.addEventListener('click', () => {
    if (isFrozen) return; // Don't toggle while frozen
    controls.autoRotate = !controls.autoRotate;
    autoRotateBtn.classList.toggle('active', controls.autoRotate);
});

// Reset view
resetViewBtn.addEventListener('click', () => {
    // Unfreeze if frozen
    if (isFrozen) {
        isFrozen = false;
        controls.enableRotate = true;
        controls.enableZoom = true;
        controls.enablePan = true;
        freezeBtn.classList.remove('active');
        freezeBtn.innerHTML = PAUSE_ICON;
        freezeBtn.title = 'Pause Globe';
    }
    camera.position.set(0, 0, 25);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    zoomSlider.value = 25;
});

// Top view
topViewBtn.addEventListener('click', () => {
    const dist = camera.position.length();
    camera.position.set(0, dist, 0);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
});

// Side view
sideViewBtn.addEventListener('click', () => {
    const dist = camera.position.length();
    camera.position.set(dist, 0, 0);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
});
