'use client';

import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';

interface StatsCardProps {
    title: string;
    value: number | string;
    icon: React.ReactElement<LucideIcon>;
    color: 'blue' | 'red' | 'green' | 'yellow';
    trend?: string;
    trendUp?: boolean;
    suffix?: string;
}

export default function StatsCard({ title, value, icon, color, trend, trendUp = true, suffix = '' }: StatsCardProps) {
    const colorMap = {
        blue: 'from-cyber-blue/20 to-cyber-blue/5 border-cyber-blue/30',
        red: 'from-threat-critical/20 to-threat-critical/5 border-threat-critical/30',
        green: 'from-cyber-green/20 to-cyber-green/5 border-cyber-green/30',
        yellow: 'from-threat-medium/20 to-threat-medium/5 border-threat-medium/30',
    };

    const iconColorMap = {
        blue: 'text-cyber-blue',
        red: 'text-threat-critical',
        green: 'text-cyber-green',
        yellow: 'text-threat-medium',
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`glass p-6 rounded-2xl border bg-gradient-to-br ${colorMap[color]} hover:scale-105 transition-transform duration-300`}
        >
            <div className="flex items-start justify-between mb-4">
                <div className={`p-3 glass rounded-xl ${iconColorMap[color]}`}>
                    {icon}
                </div>
                {trend && (
                    <span className={`text-sm font-semibold ${trendUp ? 'text-cyber-green' : 'text-threat-high'}`}>
                        {trend}
                    </span>
                )}
            </div>
            <div>
                <p className="text-gray-400 text-sm mb-1">{title}</p>
                <p className="text-3xl font-bold">
                    {value}{suffix}
                </p>
            </div>
        </motion.div>
    );
}
