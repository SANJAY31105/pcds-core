'use client';

import { useState, useEffect } from 'react';
import { TrendingUp, AlertTriangle, Shield, Clock } from 'lucide-react';

interface RiskDataPoint {
    time: string;
    score: number;
    event?: string;
}

interface PredictionTimelineProps {
    entityId?: string;
}

export default function PredictionTimeline({ entityId }: PredictionTimelineProps) {
    const [data, setData] = useState<RiskDataPoint[]>([]);
    const [trend, setTrend] = useState<'rising' | 'stable' | 'falling'>('stable');
    const [prediction, setPrediction] = useState<string>('');

    useEffect(() => {
        // Generate realistic risk timeline data
        const generateData = () => {
            const now = new Date();
            const points: RiskDataPoint[] = [];

            // Simulate 24-hour risk trend with increasing pattern
            for (let i = 23; i >= 0; i--) {
                const hour = new Date(now.getTime() - i * 3600000);
                const baseScore = 25 + Math.random() * 15;
                const trendBoost = (23 - i) * 1.5; // Increasing trend
                const score = Math.min(95, baseScore + trendBoost + (Math.random() - 0.5) * 10);

                const events = [
                    'Suspicious login attempt',
                    'Unusual network traffic',
                    'Credential access detected',
                    'Lateral movement indicator',
                    'C2 beacon identified',
                    null
                ];

                points.push({
                    time: hour.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
                    score: Math.round(score),
                    event: score > 60 ? events[Math.floor(Math.random() * events.length)] || undefined : undefined
                });
            }

            setData(points);

            // Determine trend
            const recentAvg = points.slice(-6).reduce((a, b) => a + b.score, 0) / 6;
            const olderAvg = points.slice(0, 6).reduce((a, b) => a + b.score, 0) / 6;

            if (recentAvg > olderAvg + 10) {
                setTrend('rising');
                setPrediction('Attack likelihood rising - preemptive action recommended');
            } else if (recentAvg < olderAvg - 10) {
                setTrend('falling');
                setPrediction('Risk decreasing - monitoring normal activity');
            } else {
                setTrend('stable');
                setPrediction('Risk stable - continue standard monitoring');
            }
        };

        generateData();
        const interval = setInterval(generateData, 30000);
        return () => clearInterval(interval);
    }, [entityId]);

    const maxScore = Math.max(...data.map(d => d.score), 100);
    const currentScore = data[data.length - 1]?.score || 0;

    const trendColors = {
        rising: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', icon: '↗️' },
        stable: { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', icon: '→' },
        falling: { bg: 'bg-green-500/10', border: 'border-green-500/30', text: 'text-green-400', icon: '↘️' }
    };

    const colors = trendColors[trend];

    return (
        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-[#10a37f]" />
                    <h3 className="text-sm font-medium text-white">Prediction Timeline</h3>
                </div>
                <div className="flex items-center gap-2">
                    <Clock className="w-3 h-3 text-[#666]" />
                    <span className="text-xs text-[#666]">24h</span>
                </div>
            </div>

            {/* Current Risk Score */}
            <div className={`p-4 rounded-lg ${colors.bg} border ${colors.border} mb-4`}>
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-xs text-[#888] uppercase">Current Risk Score</p>
                        <p className={`text-3xl font-bold ${colors.text}`}>{currentScore}</p>
                    </div>
                    <div className="text-right">
                        <span className={`text-2xl ${colors.text}`}>{colors.icon}</span>
                        <p className={`text-xs ${colors.text} uppercase font-medium`}>{trend}</p>
                    </div>
                </div>
            </div>

            {/* Prediction Alert */}
            <div className={`flex items-center gap-2 p-3 rounded-lg ${trend === 'rising' ? 'bg-red-500/10' : 'bg-[#1a1a1a]'} mb-4`}>
                {trend === 'rising' ? (
                    <AlertTriangle className="w-4 h-4 text-red-400 animate-pulse" />
                ) : (
                    <Shield className="w-4 h-4 text-[#10a37f]" />
                )}
                <p className={`text-sm ${trend === 'rising' ? 'text-red-400 font-medium' : 'text-[#888]'}`}>
                    {prediction}
                </p>
            </div>

            {/* Timeline Chart */}
            <div className="relative h-32">
                {/* Y-axis labels */}
                <div className="absolute left-0 top-0 bottom-0 w-8 flex flex-col justify-between text-xs text-[#666]">
                    <span>100</span>
                    <span>50</span>
                    <span>0</span>
                </div>

                {/* Chart area */}
                <div className="ml-10 h-full flex items-end gap-[2px]">
                    {data.map((point, i) => {
                        const height = (point.score / maxScore) * 100;
                        const barColor = point.score > 70 ? 'bg-red-500' :
                            point.score > 50 ? 'bg-orange-500' :
                                point.score > 30 ? 'bg-yellow-500' : 'bg-green-500';

                        return (
                            <div
                                key={i}
                                className="relative flex-1 group"
                            >
                                <div
                                    className={`w-full ${barColor} rounded-t transition-all hover:opacity-80`}
                                    style={{ height: `${height}%` }}
                                />

                                {/* Tooltip */}
                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10">
                                    <div className="bg-[#1a1a1a] border border-[#333] rounded px-2 py-1 text-xs whitespace-nowrap">
                                        <p className="text-white font-medium">{point.time}</p>
                                        <p className="text-[#888]">Risk: {point.score}</p>
                                        {point.event && <p className="text-red-400 text-[10px]">{point.event}</p>}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* X-axis labels */}
            <div className="ml-10 flex justify-between mt-2 text-xs text-[#666]">
                <span>24h ago</span>
                <span>12h ago</span>
                <span>Now</span>
            </div>

            {/* Key Events */}
            {data.filter(d => d.event).slice(-3).length > 0 && (
                <div className="mt-4 pt-4 border-t border-[#2a2a2a]">
                    <p className="text-xs text-[#666] mb-2">Key Events</p>
                    <div className="space-y-1">
                        {data.filter(d => d.event).slice(-3).map((d, i) => (
                            <div key={i} className="flex items-center gap-2 text-xs">
                                <span className="text-[#666]">{d.time}</span>
                                <span className="text-red-400">{d.event}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
