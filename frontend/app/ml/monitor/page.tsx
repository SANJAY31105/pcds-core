'use client';

import { useState, useEffect } from 'react';
import {
    Activity, Shield, AlertTriangle, CheckCircle, XCircle,
    BarChart2, Clock, Zap, Database, RefreshCw, Filter,
    ChevronDown, ExternalLink, Loader2, TrendingUp, Grid3X3
} from 'lucide-react';

// Types
interface DashboardData {
    inference_rate: number;
    total_predictions: number;
    latency: {
        avg: number;
        p50: number;
        p95: number;
        target_met: boolean;
    };
    top_classes: [string, number][];
    top_hosts: [string, number][];
    confidence_distribution: Record<string, number>;
    feedback: {
        tp_count: number;
        fp_count: number;
        fp_rate: number;
        pending_review: number;
    };
    model: {
        version: string;
        accuracy: number;
    };
}

interface PendingPrediction {
    prediction_id: string;
    timestamp: string;
    model_version: string;
    predicted_class: number;
    class_name: string;
    confidence: number;
    severity: 'critical' | 'high' | 'medium' | 'low';
    mitre_technique?: string;
    mitre_tactic?: string;
    top_features?: string[];
    source_ip?: string;
    source_host?: string;
}

interface ValidationData {
    total_reviewed: number;
    time_range_days: number;
    model_version: string;
    confusion_matrix: Record<string, Record<string, number>>;
    class_metrics: Record<string, {
        total: number;
        true_positives: number;
        false_positives: number;
        precision: number;
        recall: number;
        f1_score: number;
    }>;
    overall_metrics: {
        accuracy: number;
        fp_rate: number;
        true_positives: number;
        false_positives: number;
        precision: number;
    };
    daily_trend: {
        date: string;
        tp: number;
        fp: number;
        total: number;
        accuracy: number;
    }[];
}

export default function ModelMonitorPage() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(true);
    const [pendingPredictions, setPendingPredictions] = useState<PendingPrediction[]>([]);
    const [pendingLoading, setPendingLoading] = useState(false);
    const [feedbackLoading, setFeedbackLoading] = useState<string | null>(null);

    // Validation state
    const [validationData, setValidationData] = useState<ValidationData | null>(null);
    const [validationLoading, setValidationLoading] = useState(false);
    const [showValidation, setShowValidation] = useState(false);

    // Real-time pipeline state
    const [pipelineRunning, setPipelineRunning] = useState(false);
    const [pipelineStats, setPipelineStats] = useState<{
        events_processed: number;
        predictions_made: number;
        errors: number;
        queue_size: number;
    } | null>(null);
    const [livePredictions, setLivePredictions] = useState<PendingPrediction[]>([]);
    const [wsConnected, setWsConnected] = useState(false);

    // Filters
    const [filters, setFilters] = useState({
        minConfidence: 0,
        severity: '',
        sourceHost: ''
    });
    const [showFilters, setShowFilters] = useState(false);

    const API_BASE = 'http://localhost:8000/api/v2';

    const fetchData = async () => {
        try {
            const res = await fetch(`${API_BASE}/ml/shadow/dashboard`);
            const jsonData = await res.json();
            setData(jsonData);
        } catch (error) {
            console.error('Error fetching dashboard data:', error);
        } finally {
            setLoading(false);
        }
    };

    const fetchPendingPredictions = async () => {
        setPendingLoading(true);
        try {
            const params = new URLSearchParams();
            params.append('limit', '50');
            if (filters.minConfidence > 0) params.append('min_confidence', filters.minConfidence.toString());
            if (filters.severity) params.append('severity', filters.severity);
            if (filters.sourceHost) params.append('source_host', filters.sourceHost);

            const res = await fetch(`${API_BASE}/ml/shadow/pending?${params}`);
            const jsonData = await res.json();
            setPendingPredictions(jsonData.predictions || []);
        } catch (error) {
            console.error('Error fetching pending predictions:', error);
        } finally {
            setPendingLoading(false);
        }
    };

    const handleFeedback = async (predictionId: string, isCorrect: boolean, escalate: boolean = false) => {
        setFeedbackLoading(predictionId);
        try {
            const res = await fetch(`${API_BASE}/ml/shadow/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Analyst-ID': 'analyst@pcds.com'  // TODO: Get from auth
                },
                body: JSON.stringify({
                    prediction_id: predictionId,
                    is_correct: isCorrect,
                    escalate: escalate
                })
            });

            if (res.ok) {
                // Remove from list and refresh stats
                setPendingPredictions(prev => prev.filter(p => p.prediction_id !== predictionId));
                fetchData();
                // Refresh validation if open
                if (showValidation) fetchValidation();
            } else if (res.status === 429) {
                alert('Rate limit exceeded. Please wait a moment before submitting more feedback.');
            }
        } catch (error) {
            console.error('Error submitting feedback:', error);
        } finally {
            setFeedbackLoading(null);
        }
    };

    const fetchValidation = async () => {
        setValidationLoading(true);
        try {
            const res = await fetch(`${API_BASE}/ml/shadow/validation?days=30`);
            const jsonData = await res.json();
            setValidationData(jsonData);
        } catch (error) {
            console.error('Error fetching validation data:', error);
        } finally {
            setValidationLoading(false);
        }
    };

    const fetchPipelineStatus = async () => {
        try {
            const res = await fetch(`${API_BASE}/realtime/status`);
            const jsonData = await res.json();
            setPipelineRunning(jsonData.status === 'running');
            setPipelineStats(jsonData.stats);
        } catch (error) {
            console.error('Error fetching pipeline status:', error);
        }
    };

    const startSimulation = async () => {
        try {
            await fetch(`${API_BASE}/realtime/simulate/start?events_per_second=2`, {
                method: 'POST'
            });
            setPipelineRunning(true);
            // Fetch updated stats
            fetchPipelineStatus();
        } catch (error) {
            console.error('Error starting simulation:', error);
        }
    };

    const stopSimulation = async () => {
        try {
            await fetch(`${API_BASE}/realtime/simulate/stop`, { method: 'POST' });
            await fetch(`${API_BASE}/realtime/stop`, { method: 'POST' });
            setPipelineRunning(false);
            fetchPipelineStatus();
        } catch (error) {
            console.error('Error stopping simulation:', error);
        }
    };

    useEffect(() => {
        fetchData();
        fetchPendingPredictions();
        fetchPipelineStatus();
        const interval = setInterval(() => {
            fetchData();
            if (pipelineRunning) fetchPipelineStatus();
        }, 5000);
        return () => clearInterval(interval);
    }, [pipelineRunning]);

    useEffect(() => {
        fetchPendingPredictions();
    }, [filters]);

    const getSeverityBadge = (severity: string) => {
        const colors: Record<string, string> = {
            critical: 'bg-[#ef4444] text-white',
            high: 'bg-[#f97316] text-white',
            medium: 'bg-[#eab308] text-black',
            low: 'bg-[#3b82f6] text-white'
        };
        return colors[severity] || colors.medium;
    };

    const getSeverityColor = (name: string) => {
        const n = name.toLowerCase();
        if (n.includes('dos') || n.includes('ddos')) return '#ef4444';
        if (n.includes('botnet')) return '#f97316';
        if (n.includes('scan')) return '#3b82f6';
        if (n.includes('brute')) return '#eab308';
        if (n.includes('normal')) return '#22c55e';
        return '#666666';
    };

    if (loading || !data) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-[#0a0a0a]">
                <div className="flex flex-col items-center gap-4">
                    <div className="w-8 h-8 border-2 border-[#10a37f] border-t-transparent rounded-full animate-spin"></div>
                    <p className="text-[#666]">Loading Monitor...</p>
                </div>
            </div>
        );
    }

    const confidenceBuckets = ['0-50', '50-70', '70-90', '90-100'];

    return (
        <div className="min-h-screen bg-[#0a0a0a] p-6 text-[#e5e5e5]">
            {/* Header */}
            <div className="mb-6 flex justify-between items-start">
                <div>
                    <h1 className="text-2xl font-semibold text-white mb-1 flex items-center gap-3">
                        <Activity className="w-6 h-6 text-[#10a37f]" />
                        ML Shadow Mode Monitor
                    </h1>
                    <p className="text-[#666] text-sm">
                        Live monitoring of inference pipeline, latency, and drift
                    </p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="bg-[#141414] rounded-lg px-4 py-2 border border-[#2a2a2a] flex items-center gap-4">
                        <div className="text-right">
                            <div className="text-xs text-[#666] uppercase tracking-wider">Model Version</div>
                            <code className="text-[#10a37f] text-sm font-mono">{data.model.version}</code>
                        </div>
                        <div className="h-8 w-px bg-[#2a2a2a]"></div>
                        <div className="text-right">
                            <div className="text-xs text-[#666] uppercase tracking-wider">Accuracy</div>
                            <div className="text-white font-bold">{data.model.accuracy}%</div>
                        </div>
                    </div>
                    <button
                        onClick={() => { fetchData(); fetchPendingPredictions(); }}
                        className="p-3 bg-[#141414] hover:bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-[#666] hover:text-white transition-colors"
                    >
                        <RefreshCw className="w-5 h-5" />
                    </button>
                </div>
            </div>

            {/* Real-time Pipeline Control */}
            <div className="mb-6 bg-gradient-to-r from-[#10a37f]/10 to-[#141414] rounded-xl p-4 border border-[#10a37f]/30">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className={`w-3 h-3 rounded-full ${pipelineRunning ? 'bg-[#10a37f] animate-pulse' : 'bg-[#666]'}`} />
                        <div>
                            <h3 className="font-medium text-white">Real-time ML Pipeline</h3>
                            <p className="text-xs text-[#666]">
                                {pipelineRunning ? 'Processing events in real-time' : 'Pipeline stopped'}
                            </p>
                        </div>
                    </div>

                    {pipelineStats && pipelineRunning && (
                        <div className="flex items-center gap-6 text-sm">
                            <div className="text-center">
                                <div className="text-xl font-bold text-[#10a37f]">{pipelineStats.events_processed}</div>
                                <div className="text-xs text-[#666]">Events</div>
                            </div>
                            <div className="text-center">
                                <div className="text-xl font-bold text-[#3b82f6]">{pipelineStats.predictions_made}</div>
                                <div className="text-xs text-[#666]">Predictions</div>
                            </div>
                            <div className="text-center">
                                <div className="text-xl font-bold text-[#ef4444]">{pipelineStats.errors}</div>
                                <div className="text-xs text-[#666]">Errors</div>
                            </div>
                        </div>
                    )}

                    <div className="flex gap-2">
                        {!pipelineRunning ? (
                            <button
                                onClick={startSimulation}
                                className="px-4 py-2 bg-[#10a37f] hover:bg-[#0d8a6a] text-white rounded-lg font-medium flex items-center gap-2 transition-colors"
                            >
                                <Zap className="w-4 h-4" />
                                Start Simulation
                            </button>
                        ) : (
                            <button
                                onClick={stopSimulation}
                                className="px-4 py-2 bg-[#ef4444] hover:bg-[#dc2626] text-white rounded-lg font-medium flex items-center gap-2 transition-colors"
                            >
                                <XCircle className="w-4 h-4" />
                                Stop
                            </button>
                        )}
                    </div>
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-[#141414] rounded-xl p-5 border border-[#2a2a2a]">
                    <div className="flex justify-between items-start mb-2">
                        <div className="text-sm font-medium text-[#666]">Inference Rate</div>
                        <Zap className="w-4 h-4 text-[#10a37f]" />
                    </div>
                    <div className="text-2xl font-semibold text-white">
                        {data.inference_rate.toFixed(1)} <span className="text-sm font-normal text-[#666]">QPS</span>
                    </div>
                    <div className="text-xs text-[#666] mt-1">Global average</div>
                </div>

                <div className="bg-[#141414] rounded-xl p-5 border border-[#2a2a2a]">
                    <div className="flex justify-between items-start mb-2">
                        <div className="text-sm font-medium text-[#666]">P95 Latency</div>
                        <Clock className={`w-4 h-4 ${data.latency.target_met ? 'text-[#22c55e]' : 'text-[#ef4444]'}`} />
                    </div>
                    <div className={`text-2xl font-semibold ${data.latency.target_met ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                        {data.latency.p95.toFixed(1)} <span className="text-sm font-normal text-[#666]">ms</span>
                    </div>
                    <div className="text-xs text-[#666] mt-1">Target: &lt;50ms</div>
                </div>

                <div className="bg-[#141414] rounded-xl p-5 border border-[#2a2a2a]">
                    <div className="flex justify-between items-start mb-2">
                        <div className="text-sm font-medium text-[#666]">Total Predictions</div>
                        <Database className="w-4 h-4 text-[#3b82f6]" />
                    </div>
                    <div className="text-2xl font-semibold text-white">
                        {data.total_predictions.toLocaleString()}
                    </div>
                    <div className="text-xs text-[#666] mt-1">30-day retention</div>
                </div>

                <div className="bg-[#141414] rounded-xl p-5 border border-[#2a2a2a]">
                    <div className="flex justify-between items-start mb-2">
                        <div className="text-sm font-medium text-[#666]">Analyst FP Rate</div>
                        <AlertTriangle className="w-4 h-4 text-[#f97316]" />
                    </div>
                    <div className="text-2xl font-semibold text-[#f97316]">
                        {(data.feedback.fp_rate * 100).toFixed(2)}<span className="text-sm">%</span>
                    </div>
                    <div className="text-xs text-[#666] mt-1">{data.feedback.fp_count} confirmed FPs</div>
                </div>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">

                {/* Confidence Distribution */}
                <div className="bg-[#141414] rounded-xl p-5 border border-[#2a2a2a] lg:col-span-2">
                    <h3 className="text-sm font-medium text-white mb-6 flex items-center gap-2">
                        <BarChart2 className="w-4 h-4 text-[#10a37f]" />
                        Confidence Distribution
                    </h3>
                    <div className="h-64 flex items-end justify-between px-4 gap-4">
                        {confidenceBuckets.map((bucket) => {
                            const count = data.confidence_distribution[bucket] || 0;
                            const max = Math.max(...Object.values(data.confidence_distribution), 1);
                            const height = max > 0 ? (count / max) * 100 : 0;

                            return (
                                <div key={bucket} className="flex flex-col items-center gap-2 flex-1 h-full justify-end group">
                                    <div className="w-full bg-[#1a1a1a] rounded-t-sm relative h-full flex items-end overflow-hidden">
                                        <div
                                            className="w-full transition-all duration-300 hover:opacity-80"
                                            style={{ height: `${height}%`, backgroundColor: '#10a37f' }}
                                        ></div>
                                        {count > 0 && (
                                            <div className="absolute w-full top-2 text-center text-xs text-white pointer-events-none">
                                                {count}
                                            </div>
                                        )}
                                    </div>
                                    <span className="text-xs text-[#666]">{bucket}%</span>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Top Threats */}
                <div className="bg-[#141414] rounded-xl p-5 border border-[#2a2a2a]">
                    <h3 className="text-sm font-medium text-white mb-6 flex items-center gap-2">
                        <Shield className="w-4 h-4 text-[#3b82f6]" />
                        Top Threats
                    </h3>
                    <div className="space-y-4">
                        {data.top_classes.length === 0 ? (
                            <div className="text-center text-[#666] py-12">
                                No threats recorded yet
                            </div>
                        ) : (
                            data.top_classes.map(([name, count]) => {
                                const total = data.total_predictions;
                                const percentage = total > 0 ? (count / total) * 100 : 0;
                                const color = getSeverityColor(name);

                                return (
                                    <div key={name} className="flex flex-col gap-1.5">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-[#e5e5e5]">{name}</span>
                                            <span className="text-[#666] font-mono">{count.toLocaleString()}</span>
                                        </div>
                                        <div className="h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                                            <div
                                                className="h-full rounded-full"
                                                style={{ width: `${percentage}%`, backgroundColor: color }}
                                            ></div>
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>
            </div>

            {/* Feedback Stats */}
            <div className="bg-[#141414] rounded-xl p-5 border border-[#2a2a2a] mb-6">
                <div className="flex flex-row items-center justify-between mb-6">
                    <div>
                        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            <CheckCircle className="w-5 h-5 text-[#22c55e]" />
                            Analyst Feedback Loop
                        </h3>
                        <p className="text-sm text-[#666] mt-1">
                            Validate shadow mode predictions to retrain the model
                        </p>
                    </div>
                    <div className="hidden sm:block">
                        <span className="px-3 py-1 bg-[#eab308]/10 text-[#eab308] text-xs rounded-full border border-[#eab308]/20 font-medium">
                            {data.feedback.pending_review} PENDING REVIEW
                        </span>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="p-4 bg-[#1a1a1a] rounded-lg border border-[#2a2a2a] flex items-center gap-4">
                        <div className="p-3 bg-[#22c55e]/10 rounded-lg">
                            <CheckCircle className="w-6 h-6 text-[#22c55e]" />
                        </div>
                        <div>
                            <div className="text-2xl font-bold text-white">{data.feedback.tp_count}</div>
                            <div className="text-xs text-[#666] uppercase tracking-wider">True Positives</div>
                        </div>
                    </div>

                    <div className="p-4 bg-[#1a1a1a] rounded-lg border border-[#2a2a2a] flex items-center gap-4">
                        <div className="p-3 bg-[#ef4444]/10 rounded-lg">
                            <XCircle className="w-6 h-6 text-[#ef4444]" />
                        </div>
                        <div>
                            <div className="text-2xl font-bold text-white">{data.feedback.fp_count}</div>
                            <div className="text-xs text-[#666] uppercase tracking-wider">False Positives</div>
                        </div>
                    </div>

                    <div className="p-4 bg-[#1a1a1a] rounded-lg border border-[#2a2a2a] flex items-center gap-4">
                        <div className="p-3 bg-[#3b82f6]/10 rounded-lg">
                            <Database className="w-6 h-6 text-[#3b82f6]" />
                        </div>
                        <div>
                            <div className="text-2xl font-bold text-white">{(data.feedback.tp_count + data.feedback.fp_count).toLocaleString()}</div>
                            <div className="text-xs text-[#666] uppercase tracking-wider">Total Validated</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Pending Predictions Table */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] overflow-hidden">
                <div className="p-5 border-b border-[#2a2a2a] flex items-center justify-between">
                    <div>
                        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            <AlertTriangle className="w-5 h-5 text-[#eab308]" />
                            Pending Review
                        </h3>
                        <p className="text-sm text-[#666] mt-1">
                            Review predictions and submit ground truth labels
                        </p>
                    </div>
                    <button
                        onClick={() => setShowFilters(!showFilters)}
                        className={`px-3 py-2 rounded-lg flex items-center gap-2 text-sm transition-colors ${showFilters ? 'bg-[#10a37f] text-white' : 'bg-[#1a1a1a] text-[#a1a1a1] hover:text-white border border-[#2a2a2a]'}`}
                    >
                        <Filter className="w-4 h-4" />
                        Filters
                        <ChevronDown className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                    </button>
                </div>

                {/* Filters Panel */}
                {showFilters && (
                    <div className="p-4 bg-[#0f0f0f] border-b border-[#2a2a2a] flex gap-4 flex-wrap">
                        <div>
                            <label className="text-xs text-[#666] block mb-1">Min Confidence</label>
                            <select
                                value={filters.minConfidence}
                                onChange={(e) => setFilters(prev => ({ ...prev, minConfidence: parseFloat(e.target.value) }))}
                                className="bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white text-sm"
                            >
                                <option value="0">Any</option>
                                <option value="0.5">50%+</option>
                                <option value="0.7">70%+</option>
                                <option value="0.9">90%+</option>
                            </select>
                        </div>
                        <div>
                            <label className="text-xs text-[#666] block mb-1">Severity</label>
                            <select
                                value={filters.severity}
                                onChange={(e) => setFilters(prev => ({ ...prev, severity: e.target.value }))}
                                className="bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white text-sm"
                            >
                                <option value="">Any</option>
                                <option value="critical">Critical</option>
                                <option value="high">High</option>
                                <option value="medium">Medium</option>
                                <option value="low">Low</option>
                            </select>
                        </div>
                        <div>
                            <label className="text-xs text-[#666] block mb-1">Source Host</label>
                            <input
                                type="text"
                                value={filters.sourceHost}
                                onChange={(e) => setFilters(prev => ({ ...prev, sourceHost: e.target.value }))}
                                placeholder="hostname..."
                                className="bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white text-sm placeholder-[#666] w-40"
                            />
                        </div>
                        <button
                            onClick={() => setFilters({ minConfidence: 0, severity: '', sourceHost: '' })}
                            className="self-end px-3 py-2 text-sm text-[#a1a1a1] hover:text-white"
                        >
                            Clear All
                        </button>
                    </div>
                )}

                {/* Table */}
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-[#0f0f0f] border-b border-[#2a2a2a]">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Severity</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Threat Class</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Confidence</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">MITRE</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Source</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Top Features</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Time</th>
                                <th className="px-4 py-3 text-right text-xs font-medium text-[#666] uppercase">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-[#2a2a2a]">
                            {pendingLoading ? (
                                <tr>
                                    <td colSpan={8} className="px-4 py-12 text-center text-[#666]">
                                        <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                                        Loading predictions...
                                    </td>
                                </tr>
                            ) : pendingPredictions.length === 0 ? (
                                <tr>
                                    <td colSpan={8} className="px-4 py-12 text-center text-[#666]">
                                        <CheckCircle className="w-8 h-8 mx-auto mb-2 text-[#22c55e]" />
                                        No predictions pending review
                                    </td>
                                </tr>
                            ) : (
                                pendingPredictions.map((pred) => (
                                    <tr key={pred.prediction_id} className="hover:bg-[#1a1a1a] transition-colors">
                                        <td className="px-4 py-3">
                                            <span className={`text-xs font-medium px-2 py-1 rounded uppercase ${getSeverityBadge(pred.severity)}`}>
                                                {pred.severity}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="text-sm text-white font-medium">{pred.class_name}</div>
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center gap-2">
                                                <div className="w-16 h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full rounded-full bg-[#10a37f]"
                                                        style={{ width: `${pred.confidence * 100}%` }}
                                                    ></div>
                                                </div>
                                                <span className="text-sm text-white font-mono">{(pred.confidence * 100).toFixed(1)}%</span>
                                            </div>
                                        </td>
                                        <td className="px-4 py-3">
                                            {pred.mitre_technique ? (
                                                <div>
                                                    <code className="text-xs text-[#10a37f] font-mono">{pred.mitre_technique}</code>
                                                    {pred.mitre_tactic && (
                                                        <div className="text-xs text-[#666]">{pred.mitre_tactic}</div>
                                                    )}
                                                </div>
                                            ) : (
                                                <span className="text-[#666] text-xs">—</span>
                                            )}
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="text-sm text-white">{pred.source_host || '—'}</div>
                                            {pred.source_ip && (
                                                <div className="text-xs text-[#666] font-mono">{pred.source_ip}</div>
                                            )}
                                        </td>
                                        <td className="px-4 py-3">
                                            {pred.top_features && pred.top_features.length > 0 ? (
                                                <div className="text-xs text-[#a1a1a1] font-mono max-w-[150px] truncate" title={pred.top_features.join(', ')}>
                                                    {pred.top_features[0]}
                                                </div>
                                            ) : (
                                                <span className="text-[#666] text-xs">—</span>
                                            )}
                                        </td>
                                        <td className="px-4 py-3 text-xs text-[#666]">
                                            {new Date(pred.timestamp).toLocaleTimeString()}
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center justify-end gap-2">
                                                <button
                                                    onClick={() => handleFeedback(pred.prediction_id, true)}
                                                    disabled={feedbackLoading === pred.prediction_id}
                                                    className="px-2 py-1 bg-[#22c55e]/10 text-[#22c55e] rounded text-xs font-medium hover:bg-[#22c55e]/20 transition-colors disabled:opacity-50 flex items-center gap-1"
                                                    title="Mark as True Positive"
                                                >
                                                    {feedbackLoading === pred.prediction_id ? (
                                                        <Loader2 className="w-3 h-3 animate-spin" />
                                                    ) : (
                                                        <CheckCircle className="w-3 h-3" />
                                                    )}
                                                    TP
                                                </button>
                                                <button
                                                    onClick={() => handleFeedback(pred.prediction_id, false)}
                                                    disabled={feedbackLoading === pred.prediction_id}
                                                    className="px-2 py-1 bg-[#ef4444]/10 text-[#ef4444] rounded text-xs font-medium hover:bg-[#ef4444]/20 transition-colors disabled:opacity-50 flex items-center gap-1"
                                                    title="Mark as False Positive"
                                                >
                                                    <XCircle className="w-3 h-3" />
                                                    FP
                                                </button>
                                                <button
                                                    onClick={() => handleFeedback(pred.prediction_id, true, true)}
                                                    disabled={feedbackLoading === pred.prediction_id}
                                                    className="px-2 py-1 bg-[#f97316]/10 text-[#f97316] rounded text-xs font-medium hover:bg-[#f97316]/20 transition-colors disabled:opacity-50 flex items-center gap-1"
                                                    title="Escalate to Investigation"
                                                >
                                                    <ExternalLink className="w-3 h-3" />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Model Validation Section */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a]">
                <button
                    onClick={() => {
                        setShowValidation(!showValidation);
                        if (!showValidation && !validationData) fetchValidation();
                    }}
                    className="w-full p-4 flex items-center justify-between hover:bg-[#1a1a1a] transition-colors rounded-xl"
                >
                    <div className="flex items-center gap-3">
                        <Grid3X3 className="w-5 h-5 text-[#10a37f]" />
                        <span className="font-medium text-white">Model Validation</span>
                        <span className="text-xs text-[#666] px-2 py-0.5 bg-[#2a2a2a] rounded">
                            Confusion Matrix & Metrics
                        </span>
                    </div>
                    <ChevronDown className={`w-5 h-5 text-[#666] transition-transform ${showValidation ? 'rotate-180' : ''}`} />
                </button>

                {showValidation && (
                    <div className="p-6 border-t border-[#2a2a2a]">
                        {validationLoading ? (
                            <div className="flex items-center justify-center py-12">
                                <Loader2 className="w-6 h-6 text-[#10a37f] animate-spin" />
                            </div>
                        ) : validationData && validationData.total_reviewed > 0 ? (
                            <div className="space-y-6">
                                {/* Overall Metrics */}
                                <div className="grid grid-cols-5 gap-4">
                                    <div className="bg-[#0a0a0a] rounded-lg p-4 text-center">
                                        <div className="text-2xl font-bold text-white">{validationData.total_reviewed}</div>
                                        <div className="text-xs text-[#666] mt-1">Total Reviewed</div>
                                    </div>
                                    <div className="bg-[#0a0a0a] rounded-lg p-4 text-center">
                                        <div className="text-2xl font-bold text-[#10a37f]">{(validationData.overall_metrics.accuracy * 100).toFixed(1)}%</div>
                                        <div className="text-xs text-[#666] mt-1">Accuracy</div>
                                    </div>
                                    <div className="bg-[#0a0a0a] rounded-lg p-4 text-center">
                                        <div className="text-2xl font-bold text-[#3b82f6]">{(validationData.overall_metrics.precision * 100).toFixed(1)}%</div>
                                        <div className="text-xs text-[#666] mt-1">Precision</div>
                                    </div>
                                    <div className="bg-[#0a0a0a] rounded-lg p-4 text-center">
                                        <div className="text-2xl font-bold text-[#22c55e]">{validationData.overall_metrics.true_positives}</div>
                                        <div className="text-xs text-[#666] mt-1">True Positives</div>
                                    </div>
                                    <div className="bg-[#0a0a0a] rounded-lg p-4 text-center">
                                        <div className="text-2xl font-bold text-[#ef4444]">{validationData.overall_metrics.false_positives}</div>
                                        <div className="text-xs text-[#666] mt-1">False Positives</div>
                                    </div>
                                </div>

                                {/* Confusion Matrix */}
                                {Object.keys(validationData.confusion_matrix).length > 0 && (
                                    <div>
                                        <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                                            <Grid3X3 className="w-4 h-4 text-[#10a37f]" />
                                            Confusion Matrix
                                        </h4>
                                        <div className="overflow-x-auto">
                                            <table className="min-w-full text-xs">
                                                <thead>
                                                    <tr>
                                                        <th className="px-3 py-2 text-left text-[#666] font-normal">Predicted ↓ / Actual →</th>
                                                        {Object.keys(validationData.confusion_matrix).map(cls => (
                                                            <th key={cls} className="px-3 py-2 text-center text-[#a1a1a1] font-normal">{cls}</th>
                                                        ))}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {Object.entries(validationData.confusion_matrix).map(([predicted, actuals]) => (
                                                        <tr key={predicted} className="border-t border-[#2a2a2a]">
                                                            <td className="px-3 py-2 text-[#a1a1a1] font-medium">{predicted}</td>
                                                            {Object.keys(validationData.confusion_matrix).map(actual => {
                                                                const count = actuals[actual] || 0;
                                                                const isMatch = predicted === actual;
                                                                return (
                                                                    <td key={actual} className="px-3 py-2 text-center">
                                                                        <span className={`inline-block px-2 py-1 rounded ${count === 0 ? 'text-[#444]' :
                                                                            isMatch ? 'bg-[#22c55e]/20 text-[#22c55e]' :
                                                                                'bg-[#ef4444]/20 text-[#ef4444]'
                                                                            }`}>
                                                                            {count}
                                                                        </span>
                                                                    </td>
                                                                );
                                                            })}
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}

                                {/* Class Metrics */}
                                {Object.keys(validationData.class_metrics).length > 0 && (
                                    <div>
                                        <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                                            <TrendingUp className="w-4 h-4 text-[#10a37f]" />
                                            Per-Class Performance
                                        </h4>
                                        <div className="overflow-x-auto">
                                            <table className="min-w-full text-xs">
                                                <thead>
                                                    <tr className="text-[#666]">
                                                        <th className="px-4 py-2 text-left font-normal">Class</th>
                                                        <th className="px-4 py-2 text-right font-normal">Total</th>
                                                        <th className="px-4 py-2 text-right font-normal">TP</th>
                                                        <th className="px-4 py-2 text-right font-normal">FP</th>
                                                        <th className="px-4 py-2 text-right font-normal">Precision</th>
                                                        <th className="px-4 py-2 text-right font-normal">Recall</th>
                                                        <th className="px-4 py-2 text-right font-normal">F1 Score</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {Object.entries(validationData.class_metrics).map(([cls, metrics]) => (
                                                        <tr key={cls} className="border-t border-[#2a2a2a] hover:bg-[#1a1a1a]">
                                                            <td className="px-4 py-2 text-[#e5e5e5] font-medium">{cls}</td>
                                                            <td className="px-4 py-2 text-right text-[#a1a1a1]">{metrics.total}</td>
                                                            <td className="px-4 py-2 text-right text-[#22c55e]">{metrics.true_positives}</td>
                                                            <td className="px-4 py-2 text-right text-[#ef4444]">{metrics.false_positives}</td>
                                                            <td className="px-4 py-2 text-right">
                                                                <span className={metrics.precision >= 0.8 ? 'text-[#22c55e]' : metrics.precision >= 0.6 ? 'text-[#eab308]' : 'text-[#ef4444]'}>
                                                                    {(metrics.precision * 100).toFixed(1)}%
                                                                </span>
                                                            </td>
                                                            <td className="px-4 py-2 text-right">
                                                                <span className={metrics.recall >= 0.8 ? 'text-[#22c55e]' : metrics.recall >= 0.6 ? 'text-[#eab308]' : 'text-[#ef4444]'}>
                                                                    {(metrics.recall * 100).toFixed(1)}%
                                                                </span>
                                                            </td>
                                                            <td className="px-4 py-2 text-right">
                                                                <span className={`font-medium ${metrics.f1_score >= 0.8 ? 'text-[#22c55e]' : metrics.f1_score >= 0.6 ? 'text-[#eab308]' : 'text-[#ef4444]'}`}>
                                                                    {(metrics.f1_score * 100).toFixed(1)}%
                                                                </span>
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}

                                {/* Daily Trend */}
                                {validationData.daily_trend.length > 0 && (
                                    <div>
                                        <h4 className="text-sm font-medium text-white mb-3">Daily Accuracy Trend (Last 7 Days)</h4>
                                        <div className="flex items-end gap-2 h-24">
                                            {validationData.daily_trend.reverse().map((day, i) => (
                                                <div key={i} className="flex-1 flex flex-col items-center gap-1">
                                                    <div
                                                        className="w-full bg-[#10a37f]/60 rounded-t"
                                                        style={{ height: `${day.accuracy * 100}%` }}
                                                        title={`${day.date}: ${(day.accuracy * 100).toFixed(0)}% (${day.total} reviewed)`}
                                                    />
                                                    <span className="text-[10px] text-[#666]">{day.date.split('-').slice(1).join('/')}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="text-center py-8">
                                <Grid3X3 className="w-12 h-12 text-[#333] mx-auto mb-3" />
                                <p className="text-[#666]">No reviewed predictions yet</p>
                                <p className="text-xs text-[#444] mt-1">Submit feedback above to build validation data</p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
