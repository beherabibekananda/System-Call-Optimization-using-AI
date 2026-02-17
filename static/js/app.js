/**
 * SysCall AI - System Call Optimization Platform
 * Frontend JavaScript Application
 */

// API Configuration
const API_BASE = '';  // Same origin
const SOCKET_URL = window.location.origin;

// Global State
let socket = null;
let charts = {};
let liveStats = {
    count: 0,
    highLatency: 0,
    redundant: 0,
    latencies: []
};

// CSRF Helper for Django
function getCsrfToken() {
    return window.CSRF_TOKEN || '';
}

// Fetch helper that adds CSRF token
async function apiFetch(url, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCsrfToken(),
        ...options.headers
    };

    return fetch(url, { ...options, headers });
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initCharts();
    initEventListeners();
    initSocket();
    initStrace();
    loadInitialData();
});

// ============================================
// Tab Navigation
// ============================================

function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.tab;

            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Show target content
            contents.forEach(content => {
                content.classList.remove('active');
                if (content.id === targetId) {
                    content.classList.add('active');
                }
            });

            // Resize charts when switching tabs
            setTimeout(() => {
                Object.values(charts).forEach(chart => {
                    if (chart) chart.resize();
                });
            }, 100);
        });
    });
}

// ============================================
// Chart Initialization
// ============================================

function initCharts() {
    // Chart.js default configuration
    Chart.defaults.color = '#a0a0b0';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
    Chart.defaults.font.family = 'Inter, sans-serif';

    // Latency Distribution Chart
    const latencyCtx = document.getElementById('latencyChart');
    if (latencyCtx) {
        charts.latency = new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: ['read', 'write', 'open', 'close', 'mmap', 'socket', 'connect', 'fork'],
                datasets: [{
                    label: 'Before Optimization',
                    data: [0.8, 1.2, 1.5, 0.4, 3.0, 2.5, 55, 180],
                    borderColor: '#ff006e',
                    backgroundColor: 'rgba(255, 0, 110, 0.1)',
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'After Optimization',
                    data: [0.5, 0.7, 0.9, 0.3, 1.8, 1.5, 35, 100],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Latency (μs)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    }
                }
            }
        });
    }

    // Category Performance Chart
    const categoryCtx = document.getElementById('categoryChart');
    if (categoryCtx) {
        charts.category = new Chart(categoryCtx, {
            type: 'doughnut',
            data: {
                labels: ['I/O', 'File', 'Memory', 'Network', 'Process', 'Sync'],
                datasets: [{
                    data: [25, 20, 15, 18, 12, 10],
                    backgroundColor: [
                        '#00f5ff',
                        '#00ff88',
                        '#bf00ff',
                        '#ff9500',
                        '#ff006e',
                        '#667eea'
                    ],
                    borderWidth: 0,
                    spacing: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'right',
                    }
                }
            }
        });
    }

    // Analysis Chart
    const analysisCtx = document.getElementById('analysisChart');
    if (analysisCtx) {
        charts.analysis = new Chart(analysisCtx, {
            type: 'bar',
            data: {
                labels: ['I/O', 'File', 'Memory', 'Network', 'Process', 'Sync'],
                datasets: [{
                    label: 'Avg Latency (μs)',
                    data: [0, 0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(102, 126, 234, 0.7)',
                    borderRadius: 8
                }, {
                    label: 'Optimization Potential',
                    data: [0, 0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(0, 255, 136, 0.7)',
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    // Resource Chart
    const resourceCtx = document.getElementById('resourceChart');
    if (resourceCtx) {
        charts.resource = new Chart(resourceCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Current CPU',
                    data: [],
                    backgroundColor: 'rgba(255, 0, 110, 0.6)',
                    borderRadius: 4
                }, {
                    label: 'Recommended CPU',
                    data: [],
                    backgroundColor: 'rgba(0, 255, 136, 0.6)',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'CPU %'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    // Benchmark Chart
    const benchmarkCtx = document.getElementById('benchmarkChart');
    if (benchmarkCtx) {
        charts.benchmark = new Chart(benchmarkCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Latency Reduction %',
                    data: [],
                    borderColor: '#00f5ff',
                    backgroundColor: 'rgba(0, 245, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'CPU Savings %',
                    data: [],
                    borderColor: '#bf00ff',
                    backgroundColor: 'rgba(191, 0, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Throughput Increase %',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Improvement %'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    }
                }
            }
        });
    }

    // Initialize real-time bars
    initRealtimeBars();
}

function initRealtimeBars() {
    const container = document.getElementById('latencyBars');
    if (!container) return;

    container.innerHTML = '';
    for (let i = 0; i < 40; i++) {
        const bar = document.createElement('div');
        bar.className = 'realtime-bar';
        bar.style.height = `${Math.random() * 60 + 10}%`;
        container.appendChild(bar);
    }
}

// ============================================
// Event Listeners
// ============================================

function initEventListeners() {
    // Refresh button
    document.getElementById('refreshBtn')?.addEventListener('click', loadInitialData);

    // Run analysis button
    document.getElementById('runAnalysisBtn')?.addEventListener('click', runQuickAnalysis);

    // Generate analysis button
    document.getElementById('generateAnalysis')?.addEventListener('click', runDetailedAnalysis);

    // Generate predictions button
    document.getElementById('generatePredictions')?.addEventListener('click', generatePredictions);

    // Run benchmark button
    document.getElementById('runBenchmark')?.addEventListener('click', runBenchmark);

    // Real-time monitoring
    document.getElementById('startMonitor')?.addEventListener('click', startMonitoring);
    document.getElementById('stopMonitor')?.addEventListener('click', stopMonitoring);
}

// ============================================
// Socket.IO Connection
// ============================================

function initSocket() {
    try {
        socket = io(SOCKET_URL);

        socket.on('connect', () => {
            console.log('Connected to server');
            updateSystemStatus('connected');
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            updateSystemStatus('disconnected');
        });

        socket.on('syscall_event', handleSyscallEvent);

        socket.on('monitoring_status', (data) => {
            console.log('Monitoring status:', data.status);
        });

    } catch (error) {
        console.error('Socket connection failed:', error);
    }
}

function updateSystemStatus(status) {
    const badge = document.getElementById('systemStatus');
    if (!badge) return;

    if (status === 'connected') {
        badge.innerHTML = '<span class="status-dot"></span><span>System Active</span>';
        badge.style.borderColor = 'rgba(0, 255, 136, 0.3)';
    } else {
        badge.innerHTML = '<span class="status-dot" style="background: #ff006e;"></span><span>Disconnected</span>';
        badge.style.borderColor = 'rgba(255, 0, 110, 0.3)';
    }
}

// ============================================
// Data Loading
// ============================================

async function loadInitialData() {
    try {
        // Check API status
        const status = await apiFetch(`${API_BASE}/api/status`).then(r => r.json());
        console.log('API Status:', status);

        // Run a quick analysis for dashboard data
        runQuickAnalysis();

    } catch (error) {
        console.error('Failed to load initial data:', error);
    }
}

// ============================================
// Analysis Functions
// ============================================

async function runQuickAnalysis() {
    const btn = document.getElementById('runAnalysisBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '⏳ Analyzing...';
    }

    try {
        const response = await apiFetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            body: JSON.stringify({
                generate: true,
                num_calls: 1000,
                process_type: 'mixed'
            })
        });

        const data = await response.json();

        if (data.success) {
            updateDashboardMetrics(data.analysis);
            updateRecommendations(data.recommendations);
        }

    } catch (error) {
        console.error('Analysis failed:', error);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = 'Analyze';
        }
    }
}

async function runDetailedAnalysis() {
    const btn = document.getElementById('generateAnalysis');
    const processType = document.getElementById('processType')?.value || 'mixed';

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '⏳ Analyzing...';
    }

    try {
        const response = await apiFetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            body: JSON.stringify({
                generate: true,
                num_calls: 2000,
                process_type: processType
            })
        });

        const data = await response.json();

        if (data.success) {
            // Update analysis metrics
            document.getElementById('totalSyscalls').textContent = data.analysis.total_syscalls.toLocaleString();
            document.getElementById('highLatencyCount').textContent = data.analysis.high_latency_count.toLocaleString();
            document.getElementById('redundantCount').textContent = data.analysis.optimization_candidates.toLocaleString();
            document.getElementById('anomalyCount').textContent = data.analysis.anomalies_detected.toLocaleString();

            // Update analysis chart
            if (data.analysis.category_breakdown) {
                updateAnalysisChart(data.analysis.category_breakdown);
            }

            // Update detailed recommendations
            updateDetailedRecommendations(data.recommendations);
        }

    } catch (error) {
        console.error('Detailed analysis failed:', error);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = 'Generate Analysis';
        }
    }
}

function updateDashboardMetrics(analysis) {
    // Update metric cards
    document.getElementById('avgLatency').textContent = analysis.avg_latency?.toFixed(2) || '--';
    document.getElementById('cpuUsage').textContent = `${analysis.cpu_avg?.toFixed(1) || '--'}%`;
    document.getElementById('throughput').textContent = Math.round(1000000 / (analysis.avg_latency || 1)).toLocaleString();
    document.getElementById('optimizedCalls').textContent = analysis.optimization_candidates?.toLocaleString() || '--';

    // Update change indicators
    const latencyReduction = ((1 - analysis.avg_latency / (analysis.p95_latency || 1)) * 100);
    document.getElementById('latencyChange').innerHTML = `<span>↓</span> <span>${latencyReduction.toFixed(1)}% from p95</span>`;

    document.getElementById('cpuChange').innerHTML = `<span>↓</span> <span>${(100 - analysis.cpu_avg).toFixed(1)}% available</span>`;
    document.getElementById('throughputChange').innerHTML = `<span>↑</span> <span>Optimized</span>`;
}

function updateRecommendations(recommendations) {
    const container = document.getElementById('recommendationsList');
    if (!container || !recommendations || recommendations.length === 0) return;

    container.innerHTML = recommendations.slice(0, 5).map(rec => `
        <div class="recommendation-item ${rec.severity}">
            <div class="recommendation-header">
                <span class="recommendation-type">
                    ${rec.type === 'high_latency' ? '⚠️' : rec.type === 'redundancy' ? '🔄' : '🔍'}
                    ${formatRecType(rec.type)}
                </span>
                <span class="severity-badge ${rec.severity}">${rec.severity}</span>
            </div>
            <p class="recommendation-text">${rec.recommendation}</p>
        </div>
    `).join('');
}

function updateDetailedRecommendations(recommendations) {
    const container = document.getElementById('detailedRecommendations');
    if (!container) return;

    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p class="text-muted">No optimization opportunities detected.</p>';
        return;
    }

    container.innerHTML = recommendations.map(rec => `
        <div class="recommendation-item ${rec.severity}">
            <div class="recommendation-header">
                <span class="recommendation-type">
                    ${rec.type === 'high_latency' ? '⚠️ High Latency' :
            rec.type === 'redundancy' ? '🔄 Redundancy' : '🔍 Anomaly'}
                    ${rec.syscall ? ` - <code>${rec.syscall}</code>` : ''}
                </span>
                <span class="severity-badge ${rec.severity}">${rec.severity}</span>
            </div>
            <p class="recommendation-text">${rec.recommendation}</p>
            ${rec.count ? `<p class="text-muted mt-1">Affected calls: ${rec.count}</p>` : ''}
        </div>
    `).join('');
}

function updateAnalysisChart(categoryBreakdown) {
    if (!charts.analysis) return;

    const categories = Object.keys(categoryBreakdown);
    const avgLatencies = categories.map(c => categoryBreakdown[c].avg_latency || 0);
    const optPotentials = categories.map(c => (categoryBreakdown[c].opt_prob || 0) * 100);

    charts.analysis.data.labels = categories;
    charts.analysis.data.datasets[0].data = avgLatencies;
    charts.analysis.data.datasets[1].data = optPotentials;
    charts.analysis.update();
}

function formatRecType(type) {
    switch (type) {
        case 'high_latency': return 'High Latency Detected';
        case 'redundancy': return 'Redundant Calls';
        case 'anomaly': return 'Anomalous Pattern';
        default: return type;
    }
}

// ============================================
// Predictions Functions
// ============================================

async function generatePredictions() {
    const btn = document.getElementById('generatePredictions');

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '⏳ Generating...';
    }

    try {
        const response = await apiFetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: JSON.stringify({
                generate: true,
                num_processes: 10,
                calls_per_process: 200
            })
        });

        const data = await response.json();

        if (data.success) {
            updatePredictionTable(data.recommendations);
            updateResourceChart(data.recommendations);
        }

    } catch (error) {
        console.error('Prediction failed:', error);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = 'Generate Predictions';
        }
    }
}

function updatePredictionTable(recommendations) {
    const tbody = document.getElementById('predictionTableBody');
    if (!tbody) return;

    tbody.innerHTML = recommendations.slice(0, 10).map(rec => `
        <tr>
            <td><span class="syscall-badge">PID ${rec.pid}</span></td>
            <td>${rec.current_cpu.toFixed(1)}%</td>
            <td style="color: ${rec.recommended_cpu > rec.current_cpu ? '#00ff88' : '#ff006e'}">
                ${rec.recommended_cpu.toFixed(1)}%
                ${rec.recommended_cpu > rec.current_cpu ? '↑' : '↓'}
            </td>
            <td>${rec.current_memory.toFixed(1)}%</td>
            <td style="color: ${rec.recommended_memory > rec.current_memory ? '#00ff88' : '#ff006e'}">
                ${rec.recommended_memory.toFixed(1)}%
                ${rec.recommended_memory > rec.current_memory ? '↑' : '↓'}
            </td>
            <td>
                <div class="progress-bar" style="width: 100px;">
                    <div class="progress-fill" style="width: ${Math.min(rec.priority_score * 10, 100)}%;"></div>
                </div>
            </td>
            <td>
                ${rec.actions.slice(0, 2).map(a => `
                    <span class="category-badge ${a.type.includes('cpu') ? 'process' : 'memory'}">${a.type}</span>
                `).join('')}
            </td>
        </tr>
    `).join('');
}

function updateResourceChart(recommendations) {
    if (!charts.resource) return;

    const pids = recommendations.slice(0, 8).map(r => `PID ${r.pid}`);
    const currentCpu = recommendations.slice(0, 8).map(r => r.current_cpu);
    const recommendedCpu = recommendations.slice(0, 8).map(r => r.recommended_cpu);

    charts.resource.data.labels = pids;
    charts.resource.data.datasets[0].data = currentCpu;
    charts.resource.data.datasets[1].data = recommendedCpu;
    charts.resource.update();
}

// ============================================
// Benchmark Functions
// ============================================

async function runBenchmark() {
    const btn = document.getElementById('runBenchmark');
    const iterations = parseInt(document.getElementById('benchmarkIterations')?.value) || 5;
    const progressDiv = document.getElementById('benchmarkProgress');
    const progressBar = document.getElementById('benchmarkProgressBar');
    const progressText = document.getElementById('benchmarkPercent');

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '⏳ Running...';
    }

    progressDiv?.classList.remove('hidden');

    try {
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress = Math.min(progress + 5, 90);
            if (progressBar) progressBar.style.width = `${progress}%`;
            if (progressText) progressText.textContent = `${progress}%`;
        }, 200);

        const response = await apiFetch(`${API_BASE}/api/benchmark`, {
            method: 'POST',
            body: JSON.stringify({
                iterations: iterations,
                calls_per_iteration: 500
            })
        });

        clearInterval(progressInterval);

        const data = await response.json();

        if (data.success) {
            // Complete progress
            if (progressBar) progressBar.style.width = '100%';
            if (progressText) progressText.textContent = '100%';

            // Update metrics
            document.getElementById('benchLatencyReduction').textContent =
                `${data.average_improvement.latency_reduction.toFixed(1)}%`;
            document.getElementById('benchCpuSavings').textContent =
                `${data.average_improvement.cpu_savings.toFixed(1)}%`;
            document.getElementById('benchMemorySavings').textContent =
                `${data.average_improvement.memory_savings.toFixed(1)}%`;
            document.getElementById('benchThroughputIncrease').textContent =
                `${data.average_improvement.throughput_increase.toFixed(1)}%`;

            // Update chart
            updateBenchmarkChart(data.iterations);

            // Hide progress after delay
            setTimeout(() => progressDiv?.classList.add('hidden'), 1000);
        }

    } catch (error) {
        console.error('Benchmark failed:', error);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = 'Run Benchmark';
        }
    }
}

function updateBenchmarkChart(iterations) {
    if (!charts.benchmark) return;

    const labels = iterations.map(i => `#${i.iteration}`);
    const latencyData = iterations.map(i => i.improvement.latency_reduction);
    const cpuData = iterations.map(i => i.improvement.cpu_savings);
    const throughputData = iterations.map(i => i.improvement.throughput_increase);

    charts.benchmark.data.labels = labels;
    charts.benchmark.data.datasets[0].data = latencyData;
    charts.benchmark.data.datasets[1].data = cpuData;
    charts.benchmark.data.datasets[2].data = throughputData;
    charts.benchmark.update();
}

// ============================================
// Real-time Monitoring
// ============================================

function startMonitoring() {
    if (socket && socket.connected) {
        socket.emit('start_monitoring');
        liveStats = { count: 0, highLatency: 0, redundant: 0, latencies: [] };
        document.getElementById('realtimeFeed').innerHTML = '';
        console.log('Started monitoring');
    }
}

function stopMonitoring() {
    if (socket) {
        socket.emit('stop_monitoring');
        console.log('Stopped monitoring');
    }
}

function handleSyscallEvent(event) {
    liveStats.count++;
    if (event.is_high_latency) liveStats.highLatency++;
    if (event.is_redundant) liveStats.redundant++;
    liveStats.latencies.push(event.latency);

    // Keep only last 40 latencies
    if (liveStats.latencies.length > 40) {
        liveStats.latencies.shift();
    }

    // Update stats display
    document.getElementById('liveCount').textContent = liveStats.count;
    document.getElementById('liveHighLat').textContent = liveStats.highLatency;
    document.getElementById('liveRedundant').textContent = liveStats.redundant;

    // Update feed
    addFeedItem(event);

    // Update realtime bars
    updateRealtimeBars();
}

function addFeedItem(event) {
    const feed = document.getElementById('realtimeFeed');
    if (!feed) return;

    const time = new Date(event.timestamp).toLocaleTimeString();
    const className = event.is_high_latency ? 'high-latency' : event.is_redundant ? 'redundant' : '';

    const item = document.createElement('div');
    item.className = `feed-item ${className}`;
    item.innerHTML = `
        <span class="feed-timestamp">${time}</span>
        <div class="feed-details">
            <span class="feed-syscall">${event.syscall}</span>
            <div class="feed-metrics">
                <span>⏱️ ${event.latency.toFixed(2)}μs</span>
                <span>💻 ${event.cpu.toFixed(1)}%</span>
                <span>💾 ${event.memory.toFixed(1)}%</span>
            </div>
        </div>
        <span class="category-badge ${event.category}">${event.category}</span>
    `;

    feed.insertBefore(item, feed.firstChild);

    // Keep only last 50 items
    while (feed.children.length > 50) {
        feed.removeChild(feed.lastChild);
    }
}

function updateRealtimeBars() {
    const container = document.getElementById('latencyBars');
    if (!container) return;

    const bars = container.querySelectorAll('.realtime-bar');
    const maxLatency = Math.max(...liveStats.latencies, 100);

    bars.forEach((bar, i) => {
        const latency = liveStats.latencies[liveStats.latencies.length - bars.length + i] || Math.random() * 10;
        const height = Math.min((latency / maxLatency) * 100, 100);
        bar.style.height = `${Math.max(height, 5)}%`;

        // Color based on latency
        if (latency > maxLatency * 0.7) {
            bar.style.background = 'linear-gradient(135deg, #eb3349, #f45c43)';
        } else if (latency > maxLatency * 0.4) {
            bar.style.background = 'linear-gradient(135deg, #f7971e, #ffd200)';
        } else {
            bar.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
        }
    });
}

// ============================================
// Utility Functions
// ============================================

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Handle window resize
window.addEventListener('resize', debounce(() => {
    Object.values(charts).forEach(chart => {
        if (chart) chart.resize();
    });
}, 250));

console.log('SysCall AI initialized');

// ============================================
// Strace Integration
// ============================================

let straceLastResults = null;  // Store last analysis results

function initStrace() {
    initStraceUpload();
    initStraceTextParse();
    initStraceSamples();
    initStraceLiveTrace();
    initStraceExport();
    loadTraceToolInfo();
}

// --- File Upload (Drag & Drop) ---
function initStraceUpload() {
    const dropZone = document.getElementById('straceDropZone');
    const fileInput = document.getElementById('straceFileInput');
    if (!dropZone || !fileInput) return;

    // Click to browse
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadStraceFile(files[0]);
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) uploadStraceFile(e.target.files[0]);
    });
}

function uploadStraceFile(file) {
    const dropZone = document.getElementById('straceDropZone');
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadFileName = document.getElementById('uploadFileName');
    const uploadFileSize = document.getElementById('uploadFileSize');

    // Show file info
    uploadStatus.style.display = 'block';
    uploadFileName.textContent = file.name;
    uploadFileSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
    dropZone.classList.add('processing');

    const formData = new FormData();
    formData.append('file', file);

    fetch(`${API_BASE}/api/strace/parse`, {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': getCsrfToken()
        }
    })
        .then(res => res.json())
        .then(data => {
            dropZone.classList.remove('processing');
            if (data.success) {
                showStraceResults(data, `File: ${file.name}`);
            } else {
                showNotification(`Parse error: ${data.error}`, 'error');
            }
        })
        .catch(err => {
            dropZone.classList.remove('processing');
            showNotification(`Upload failed: ${err.message}`, 'error');
        });
}

// --- Paste Text ---
function initStraceTextParse() {
    const btn = document.getElementById('parseStraceText');
    if (!btn) return;

    btn.addEventListener('click', () => {
        const text = document.getElementById('straceTextInput').value.trim();
        if (!text) {
            showNotification('Please paste some strace output first.', 'warning');
            return;
        }
        btn.textContent = '⏳ Parsing...';
        btn.disabled = true;

        apiFetch(`${API_BASE}/api/strace/parse-text`, {
            method: 'POST',
            body: JSON.stringify({ text })
        })
            .then(res => res.json())
            .then(data => {
                btn.textContent = 'Parse Text';
                btn.disabled = false;
                if (data.success) {
                    showStraceResults(data, 'Pasted text');
                } else {
                    showNotification(`Parse error: ${data.error}`, 'error');
                }
            })
            .catch(err => {
                btn.textContent = 'Parse Text';
                btn.disabled = false;
                showNotification(`Request failed: ${err.message}`, 'error');
            });
    });
}

// --- Load Sample Files ---
function initStraceSamples() {
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const sample = btn.dataset.sample;
            btn.classList.add('loading');
            const originalText = btn.innerHTML;
            btn.innerHTML = '⏳ Loading...';

            apiFetch(`${API_BASE}/api/strace/load-sample`, {
                method: 'POST',
                body: JSON.stringify({ filename: sample })
            })
                .then(res => res.json())
                .then(data => {
                    btn.classList.remove('loading');
                    btn.innerHTML = originalText;
                    if (data.success) {
                        showStraceResults(data, `Sample: ${sample}`);
                    } else {
                        showNotification(`Error: ${data.error}`, 'error');
                    }
                })
                .catch(err => {
                    btn.classList.remove('loading');
                    btn.innerHTML = originalText;
                    showNotification(`Failed: ${err.message}`, 'error');
                });
        });
    });
}

// --- Live Trace ---
function initStraceLiveTrace() {
    const btn = document.getElementById('runTraceBtn');
    if (!btn) return;

    btn.addEventListener('click', () => {
        const command = document.getElementById('traceCommand').value.trim();
        if (!command) {
            showNotification('Enter a command to trace.', 'warning');
            return;
        }
        btn.textContent = '⏳ Tracing...';
        btn.disabled = true;

        apiFetch(`${API_BASE}/api/strace/run`, {
            method: 'POST',
            body: JSON.stringify({ command, timeout: 30 })
        })
            .then(res => res.json())
            .then(data => {
                btn.textContent = '▶ Trace';
                btn.disabled = false;
                if (data.success) {
                    showStraceResults(data, `Live trace: ${command}`);
                } else {
                    showNotification(`Trace failed: ${data.error}`, 'error');
                }
            })
            .catch(err => {
                btn.textContent = '▶ Trace';
                btn.disabled = false;
                showNotification(`Trace failed: ${err.message}`, 'error');
            });
    });
}

function loadTraceToolInfo() {
    apiFetch(`${API_BASE}/api/strace/tool-info`)
        .then(res => res.json())
        .then(data => {
            const info = data.tool_info || {};
            const el = document.getElementById('traceToolInfo');
            if (el) {
                if (info.available) {
                    el.innerHTML = `✅ <strong>${info.tool_name}</strong> available on ${info.system}`;
                } else {
                    el.innerHTML = `⚠️ No tracing tool found. ${info.note || 'Upload strace files instead.'}`;
                }
            }
        })
        .catch(() => {
            const el = document.getElementById('traceToolInfo');
            if (el) el.textContent = '⚠️ Could not detect tracing tool.';
        });
}

// --- Display Results ---
function showStraceResults(data, sourceLabel) {
    straceLastResults = data;

    // Show result cards
    document.getElementById('straceResultsCard').style.display = 'block';
    document.getElementById('straceRecsCard').style.display = 'block';

    // Update source label
    document.getElementById('straceSource').textContent = sourceLabel;

    // Update metrics
    const analysis = data.analysis || {};
    document.getElementById('straceTotalCalls').textContent = formatNumber(data.total_syscalls || 0);
    document.getElementById('straceHighLat').textContent = analysis.high_latency_count || 0;
    document.getElementById('straceOptCandidates').textContent = analysis.optimization_candidates || 0;
    document.getElementById('straceAnomalies').textContent = analysis.anomalies_detected || 0;

    // Build syscall distribution chart
    renderStraceSyscallChart(data.syscall_summary || {});

    // Build category chart
    renderStraceCategoryChart(data.category_summary || {});

    // Render recommendations
    renderStraceRecommendations(data.recommendations || []);

    // Scroll to results
    document.getElementById('straceResultsCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderStraceSyscallChart(summary) {
    const ctx = document.getElementById('straceSyscallChart');
    if (!ctx) return;

    // Destroy old chart
    if (charts.straceSyscall) charts.straceSyscall.destroy();

    const labels = Object.keys(summary).slice(0, 12);
    const values = labels.map(l => summary[l]);

    const palette = [
        '#00f5ff', '#bf00ff', '#00ff88', '#ff006e', '#ff9500',
        '#ffd200', '#667eea', '#f5576c', '#11998e', '#38ef7d',
        '#f093fb', '#764ba2'
    ];

    charts.straceSyscall = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Call Count',
                data: values,
                backgroundColor: labels.map((_, i) => palette[i % palette.length] + '80'),
                borderColor: labels.map((_, i) => palette[i % palette.length]),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: { color: '#a0a0b0', maxRotation: 45, font: { size: 10 } },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: '#a0a0b0' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

function renderStraceCategoryChart(summary) {
    const ctx = document.getElementById('straceCategoryChart');
    if (!ctx) return;

    if (charts.straceCategory) charts.straceCategory.destroy();

    const labels = Object.keys(summary);
    const values = labels.map(l => summary[l]);

    const palette = [
        '#00f5ff', '#bf00ff', '#00ff88', '#ff006e', '#ff9500',
        '#ffd200', '#667eea', '#f5576c', '#11998e'
    ];

    charts.straceCategory = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: labels.map((_, i) => palette[i % palette.length] + '99'),
                borderColor: labels.map((_, i) => palette[i % palette.length]),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#e0e0e8', font: { size: 11 }, padding: 12 }
                }
            }
        }
    });
}

function renderStraceRecommendations(recommendations) {
    const container = document.getElementById('straceRecommendations');
    if (!container) return;

    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p class="text-muted">No optimization recommendations at this time. ✅</p>';
        return;
    }

    let html = '';
    recommendations.forEach((rec, i) => {
        const severityClass = rec.severity === 'high' ? 'high' : rec.severity === 'medium' ? 'medium' : 'low';
        const icon = rec.severity === 'high' ? '🔴' : rec.severity === 'medium' ? '🟡' : '🟢';
        const type = (rec.type || 'general').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        html += `
        <div class="recommendation-item" style="margin-bottom: 0.75rem; padding: 0.75rem; border-left: 3px solid ${rec.severity === 'high' ? '#ff006e' : rec.severity === 'medium' ? '#ff9500' : '#00ff88'
            }; background: var(--bg-glass); border-radius: 0 var(--radius-md) var(--radius-md) 0;">
            <div class="flex-between mb-1">
                <span style="font-weight: 600; color: var(--text-primary);">${icon} ${type}</span>
                <span class="severity-badge ${severityClass}">${rec.severity || 'info'}</span>
            </div>
            ${rec.syscall ? `<div style="font-size: 0.8rem; color: var(--neon-cyan);">Syscall: <strong>${rec.syscall}</strong> | Count: ${rec.count || 'N/A'}</div>` : ''}
            ${rec.avg_latency ? `<div style="font-size: 0.8rem; color: var(--text-secondary);">Avg Latency: ${rec.avg_latency.toFixed(2)} μs</div>` : ''}
            <div class="recommendation-text" style="margin-top: 0.25rem;">${rec.recommendation || ''}</div>
        </div>`;
    });

    container.innerHTML = html;
}

// --- Export ---
function initStraceExport() {
    document.querySelectorAll('.export-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const format = btn.dataset.format;
            exportResults(format);
        });
    });

    const closeBtn = document.getElementById('closeExportResult');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            document.getElementById('exportResultCard').style.display = 'none';
        });
    }
}

function exportResults(format) {
    const payload = {
        format,
        recommendations: straceLastResults ? straceLastResults.recommendations : [],
        analysis: straceLastResults ? straceLastResults.analysis : {},
        strace_stats: straceLastResults ? straceLastResults.parse_stats : {}
    };

    showNotification(`Exporting as ${format}...`, 'info');

    apiFetch(`${API_BASE}/api/export`, {
        method: 'POST',
        body: JSON.stringify(payload)
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                // Show preview
                document.getElementById('exportResultCard').style.display = 'block';
                document.getElementById('exportPreview').textContent = data.content || 'Export generated successfully.';
                showNotification(`✅ ${format.toUpperCase()} exported! File: ${data.filename}`, 'success');
            } else {
                showNotification(`Export failed: ${data.error}`, 'error');
            }
        })
        .catch(err => {
            showNotification(`Export failed: ${err.message}`, 'error');
        });
}
