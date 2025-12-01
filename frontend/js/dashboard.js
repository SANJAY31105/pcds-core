// --- Configuration ---
const API_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/dashboard";

// --- State ---
let packetCounter = 0;
let trafficChartInstance = null;
let blockedIPs = new Set();

// --- Initialization ---
function init() {
    // Inject Tailwind Theme Config Programmatically
    tailwind.config = {
        theme: {
            extend: {
                colors: {
                    cyber: {
                        900: '#0a0a0f', 800: '#13131f', 700: '#1c1c2e', neon: '#00f3ff', 
                        danger: '#ff2a2a', warning: '#ffae00', success: '#00ff9d', 
                        text: '#e0e0e0', dim: '#8a8a9b'
                    }
                },
                fontFamily: { sans: ['Inter', 'sans-serif'], mono: ['JetBrains Mono', 'monospace'] },
                boxShadow: { 'neon-blue': '0 0 10px rgba(0, 243, 255, 0.3)' }
            }
        }
    }

    initChart();
    updateClock();
    setInterval(updateClock, 1000);
    connectWebSocket();
    fetchHistory();
    fetchBlockedList();
}

// --- Data Fetching ---
async function fetchHistory() {
    try {
        const res = await fetch(`${API_URL}/api/history`);
        const data = await res.json();
        data.forEach(alert => { addAlertRow(alert); });
    } catch (e) { console.log("Could not fetch history"); }
}

async function fetchBlockedList() {
    try {
        const res = await fetch(`${API_URL}/api/blocked_list`);
        const data = await res.json();
        data.forEach(ip => blockedIPs.add(ip));
    } catch (e) {}
}

async function blockIP(ip, btn) {
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
    try {
        const res = await fetch(`${API_URL}/api/block_ip`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ip: ip})
        });
        const data = await res.json();
        if(data.status === 'blocked') {
            showToast(`IP ${ip} blocked successfully`, "success");
            btn.innerHTML = 'BLOCKED';
            btn.className = "text-xs bg-cyber-danger text-black font-bold px-3 py-1 rounded cursor-not-allowed opacity-50";
            btn.disabled = true;
            blockedIPs.add(ip);
        }
    } catch(e) {
        showToast("Failed to block IP", "danger");
        btn.innerHTML = 'Retry';
    }
}

async function downloadReport() {
    showToast("Generating PDF Report...", "success");
    window.location.href = `${API_URL}/api/report`;
}

// --- Real-time Logic (WebSocket) ---
function connectWebSocket() {
    const socket = new WebSocket(WS_URL);
    socket.onopen = () => showToast("Connected to PCDS Engine", "success");
    
    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "traffic_update") {
            const data = msg.data;
            packetCounter++;
            document.getElementById('packetCount').innerText = packetCounter;

            const normalVal = data.is_anomaly ? 10 : 50;
            const anomalyVal = data.is_anomaly ? 90 : 5;
            updateTrafficChart(normalVal, anomalyVal);

            if (data.is_anomaly) {
                addLog(`[ALERT] ${data.threat_type} from ${data.ip}`, 'CRITICAL');
                addAlertRow({
                    timestamp: new Date().toLocaleTimeString(),
                    type: data.threat_type,
                    ip: data.ip,
                    confidence: data.confidence,
                    status: "Active"
                });
                // Optional: Play Sound
                // new Audio('assets/alert.mp3').play().catch(e=>{}); 
            } else {
                if(Math.random() > 0.9) addLog(`[INFO] Analyzed ${data.ip}: Clean`, 'INFO');
            }
        }
    };
}

// --- UI Components ---
function addAlertRow(alert) {
     const table = document.getElementById('alertsTable');
     const tr = document.createElement('tr');
     tr.className = "hover:bg-cyber-700/20 border-b border-cyber-700/30";
     
     let actionBtn = `<button onclick="blockIP('${alert.ip}', this)" class="text-xs bg-cyber-700 hover:bg-white hover:text-black px-3 py-1 rounded transition-colors text-white">Mitigate</button>`;
     if(blockedIPs.has(alert.ip)) {
         actionBtn = `<span class="text-xs font-bold text-cyber-dim">BLOCKED</span>`;
     }

     tr.innerHTML = `<td class="px-6 py-4 font-mono">${alert.timestamp}</td><td class="px-6 py-4 font-bold text-cyber-danger">${alert.type}</td><td class="px-6 py-4 font-mono">${alert.ip}</td><td class="px-6 py-4 text-cyber-neon">${alert.confidence}%</td><td class="px-6 py-4">${actionBtn}</td>`;
     table.insertBefore(tr, table.firstChild);
     if(table.children.length > 50) table.lastChild.remove();
}

function addLog(msg, type) {
    const term = document.getElementById('logTerminal');
    const div = document.createElement('div');
    const time = new Date().toLocaleTimeString();
    div.className = `whitespace-nowrap ${type === 'CRITICAL' ? 'text-cyber-danger font-bold' : 'text-cyber-dim'}`;
    div.innerText = `[${time}] ${msg}`;
    term.appendChild(div);
    term.scrollTop = term.scrollHeight;
    if(term.children.length > 50) term.firstChild.remove();
}

function initChart() {
    const ctx = document.getElementById('trafficChart').getContext('2d');
    const gradientNeon = ctx.createLinearGradient(0, 0, 0, 400); gradientNeon.addColorStop(0, 'rgba(0, 243, 255, 0.5)'); gradientNeon.addColorStop(1, 'rgba(0, 243, 255, 0)');
    const gradientRed = ctx.createLinearGradient(0, 0, 0, 400); gradientRed.addColorStop(0, 'rgba(255, 42, 42, 0.5)'); gradientRed.addColorStop(1, 'rgba(255, 42, 42, 0)');

    trafficChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 20}, (_, i) => i),
            datasets: [
                { label: 'Normal', data: Array(20).fill(0), borderColor: '#00f3ff', backgroundColor: gradientNeon, fill: true, tension: 0.4 },
                { label: 'Threat', data: Array(20).fill(0), borderColor: '#ff2a2a', backgroundColor: gradientRed, fill: true, tension: 0.1 }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { grid: { color: 'rgba(255,255,255,0.05)' } } } }
    });
}

function updateTrafficChart(val1, val2) {
    if(!trafficChartInstance) return;
    trafficChartInstance.data.datasets[0].data.shift(); trafficChartInstance.data.datasets[0].data.push(val1);
    trafficChartInstance.data.datasets[1].data.shift(); trafficChartInstance.data.datasets[1].data.push(val2);
    trafficChartInstance.update();
}

function showToast(msg, type) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `glass-panel border-l-4 ${type === 'success' ? 'border-cyber-success' : 'border-cyber-neon'} ${type === 'danger' ? 'border-cyber-danger' : ''} p-4 rounded shadow-lg flex items-center gap-3 min-w-[300px] toast`;
    toast.innerHTML = `<i class="fa-solid fa-info-circle text-white"></i><div class="text-sm font-semibold text-white">${msg}</div>`;
    container.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 100);
    setTimeout(() => toast.remove(), 3000);
}

function updateClock() { document.getElementById('clock').innerText = new Date().toLocaleTimeString('en-US', { hour12: false }); }

window.onload = init;