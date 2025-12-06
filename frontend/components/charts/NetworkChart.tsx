'use client';

import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ThreatDetection } from '@/types';
import { useMemo } from 'react';

interface NetworkChartProps {
    threats: ThreatDetection[];
}

export default function NetworkChart({ threats }: NetworkChartProps) {
    const chartData = useMemo(() => {
        const last12 = threats.slice(0, 12).reverse();
        return last12.map((threat, idx) => ({
            name: `T${idx + 1}`,
            risk: threat.risk_score,
            time: new Date(threat.timestamp).toLocaleTimeString(),
        }));
    }, [threats]);

    return (
        <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={chartData}>
                <defs>
                    <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00f0ff" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#00f0ff" stopOpacity={0} />
                    </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" stroke="rgba(255,255,255,0.3)" style={{ fontSize: 12 }} />
                <YAxis stroke="rgba(255,255,255,0.3)" style={{ fontSize: 12 }} />
                <Tooltip
                    contentStyle={{
                        backgroundColor: 'rgba(10, 10, 15, 0.9)',
                        border: '1px solid rgba(0, 240, 255, 0.3)',
                        borderRadius: '8px',
                    }}
                    labelStyle={{ color: '#00f0ff' }}
                />
                <Area
                    type="monotone"
                    dataKey="risk"
                    stroke="#00f0ff"
                    strokeWidth={2}
                    fill="url(#riskGradient)"
                    animationDuration={1000}
                />
            </AreaChart>
        </ResponsiveContainer>
    );
}
