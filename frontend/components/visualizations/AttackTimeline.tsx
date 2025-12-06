import React from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';
import { format } from 'date-fns';

interface AttackTimelineProps {
  detections: any[];
}

const KILL_CHAIN_STAGES = [
  "Reconnaissance",
  "Weaponization",
  "Delivery",
  "Exploitation",
  "Installation",
  "Command and Control",
  "Actions on Objectives"
];

export function AttackTimeline({ detections }: AttackTimelineProps) {
  // Process data: Map detections to Kill Chain stages
  const data = detections.map(d => {
    // Determine stage based on tactic or severity (mock logic if tactic missing)
    let stageIndex = 0;
    const tactic = d.tactic?.toLowerCase() || "";
    
    if (tactic.includes("recon")) stageIndex = 1;
    else if (tactic.includes("weapon")) stageIndex = 2;
    else if (tactic.includes("deliver")) stageIndex = 3;
    else if (tactic.includes("exploit")) stageIndex = 4;
    else if (tactic.includes("install")) stageIndex = 5;
    else if (tactic.includes("command") || tactic.includes("c2")) stageIndex = 6;
    else if (tactic.includes("action") || tactic.includes("exfil")) stageIndex = 7;
    else {
        // Fallback based on severity if tactic not mapped
        stageIndex = Math.min(Math.max(Math.floor((d.severity || 0) / 15), 1), 7);
    }

    return {
      id: d.id,
      name: d.title || "Unknown Detection",
      timestamp: new Date(d.timestamp).getTime(),
      stage: stageIndex,
      severity: d.severity,
      tactic: d.tactic
    };
  }).sort((a, b) => a.timestamp - b.timestamp);

  const formatXAxis = (tickItem: number) => {
    return format(new Date(tickItem), 'HH:mm:ss');
  };

  const formatYAxis = (tickItem: number) => {
    return KILL_CHAIN_STAGES[tickItem - 1] || "";
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-slate-900 border border-slate-700 p-3 rounded shadow-lg text-xs">
          <p className="font-bold text-slate-100">{data.name}</p>
          <p className="text-slate-300">Time: {format(new Date(data.timestamp), 'HH:mm:ss')}</p>
          <p className="text-slate-300">Stage: {KILL_CHAIN_STAGES[data.stage - 1]}</p>
          <p className="text-slate-300">Severity: {data.severity}</p>
          <p className="text-slate-300">Tactic: {data.tactic}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-lg shadow-sm">
      <div className="p-6 pb-2">
        <h3 className="text-lg font-medium text-slate-100">
          Attack Progression (Cyber Kill Chain)
        </h3>
      </div>
      <div className="p-6 pt-0">
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis 
                type="number" 
                dataKey="timestamp" 
                name="Time" 
                domain={['auto', 'auto']} 
                tickFormatter={formatXAxis} 
                stroke="#94a3b8"
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                type="number" 
                dataKey="stage" 
                name="Stage" 
                domain={[0, 8]} 
                tickCount={9}
                tickFormatter={formatYAxis} 
                stroke="#94a3b8"
                width={120}
                tick={{ fontSize: 11 }}
              />
              <ZAxis type="number" dataKey="severity" range={[50, 400]} name="Severity" />
              <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Detections" data={data} fill="#f43f5e" shape="circle" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
