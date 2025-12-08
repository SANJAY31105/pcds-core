'use client';

import { useState, useEffect } from 'react';
import { Shield, Clock, AlertTriangle, Target, Zap, CheckCircle, ChevronRight } from 'lucide-react';

interface TimelineEvent {
  id: string;
  timestamp: string;
  type: 'detection' | 'action' | 'approval' | 'escalation';
  title: string;
  description: string;
  severity?: string;
  entity?: string;
  technique_id?: string;
  status?: 'pending' | 'completed' | 'failed';
}

interface AttackTimelineProps {
  events?: TimelineEvent[];
  detectionId?: string;
}

export default function AttackTimeline({ events: propEvents, detectionId }: AttackTimelineProps) {
  const [events, setEvents] = useState<TimelineEvent[]>(propEvents || []);
  const [loading, setLoading] = useState(!propEvents);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  useEffect(() => {
    if (propEvents && propEvents.length > 0) {
      setEvents(propEvents);
      setLoading(false);
    } else if (!propEvents) {
      setEvents(getDemoEvents());
      setLoading(false);
    }
  }, [propEvents]);

  const getDemoEvents = (): TimelineEvent[] => [
    {
      id: '1',
      timestamp: new Date(Date.now() - 360000).toISOString(),
      type: 'detection',
      title: 'Phishing Email Opened',
      description: 'User clicked suspicious link in email from external sender',
      severity: 'medium',
      entity: 'user-faculty-42',
      technique_id: 'T1566.002',
      status: 'completed'
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 300000).toISOString(),
      type: 'detection',
      title: 'Malware Download Initiated',
      description: 'Suspicious executable downloaded from untrusted domain',
      severity: 'high',
      entity: 'workstation-15',
      technique_id: 'T1204.002',
      status: 'completed'
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 240000).toISOString(),
      type: 'detection',
      title: 'Process Injection Attempt',
      description: 'Malicious code injection into svchost.exe detected',
      severity: 'critical',
      entity: 'workstation-15',
      technique_id: 'T1055.001',
      status: 'completed'
    },
    {
      id: '4',
      timestamp: new Date(Date.now() - 180000).toISOString(),
      type: 'escalation',
      title: 'ML Engine Alert: 94% Confidence',
      description: 'Ensemble model detected ransomware attack pattern',
      severity: 'critical',
      status: 'completed'
    },
    {
      id: '5',
      timestamp: new Date(Date.now() - 120000).toISOString(),
      type: 'approval',
      title: 'Isolation Queued for Approval',
      description: 'Decision Engine queued host isolation for analyst review',
      entity: 'workstation-15',
      status: 'pending'
    }
  ];

  const getEventConfig = (type: string, severity?: string) => {
    if (severity === 'critical') return {
      gradient: 'from-red-500 to-rose-600',
      glow: 'shadow-red-500/50',
      border: 'border-red-500',
      bg: 'bg-red-500/10',
      lineColor: '#ef4444',
      icon: AlertTriangle,
      pulse: true
    };
    if (severity === 'high') return {
      gradient: 'from-orange-500 to-amber-600',
      glow: 'shadow-orange-500/50',
      border: 'border-orange-500',
      bg: 'bg-orange-500/10',
      lineColor: '#f97316',
      icon: AlertTriangle,
      pulse: false
    };
    if (type === 'action') return {
      gradient: 'from-green-500 to-emerald-600',
      glow: 'shadow-green-500/50',
      border: 'border-green-500',
      bg: 'bg-green-500/10',
      lineColor: '#22c55e',
      icon: Shield,
      pulse: false
    };
    if (type === 'approval') return {
      gradient: 'from-yellow-500 to-amber-500',
      glow: 'shadow-yellow-500/50',
      border: 'border-yellow-500',
      bg: 'bg-yellow-500/10',
      lineColor: '#eab308',
      icon: Clock,
      pulse: true
    };
    if (type === 'escalation') return {
      gradient: 'from-purple-500 to-pink-600',
      glow: 'shadow-purple-500/50',
      border: 'border-purple-500',
      bg: 'bg-purple-500/10',
      lineColor: '#a855f7',
      icon: Zap,
      pulse: false
    };
    return {
      gradient: 'from-cyan-500 to-blue-600',
      glow: 'shadow-cyan-500/50',
      border: 'border-cyan-500',
      bg: 'bg-cyan-500/10',
      lineColor: '#06b6d4',
      icon: Target,
      pulse: false
    };
  };

  const formatTime = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const getRelativeTime = (isoString: string) => {
    const diff = Date.now() - new Date(isoString).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 60) return `${mins}m ago`;
    return `${Math.floor(mins / 60)}h ${mins % 60}m ago`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-12">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-cyan-500/30 rounded-full"></div>
          <div className="absolute top-0 left-0 w-16 h-16 border-4 border-transparent border-t-cyan-500 rounded-full animate-spin"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative py-4 pl-2">
      {/* SVG Background Line - Premium Glowing Effect */}
      <svg
        className="absolute left-0 top-0 w-20 h-full pointer-events-none"
        style={{ overflow: 'visible' }}
      >
        <defs>
          {/* Gradient for the line */}
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#06b6d4" />
            <stop offset="25%" stopColor="#8b5cf6" />
            <stop offset="50%" stopColor="#ec4899" />
            <stop offset="75%" stopColor="#ef4444" />
            <stop offset="100%" stopColor="#f59e0b" />
          </linearGradient>

          {/* Glow filter */}
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Animated glow pulse */}
          <filter id="glowPulse" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Outer glow line (thicker, more blur) */}
        <line
          x1="32" y1="32"
          x2="32" y2={`calc(100% - 32px)`}
          stroke="url(#lineGradient)"
          strokeWidth="8"
          strokeLinecap="round"
          opacity="0.3"
          filter="url(#glowPulse)"
        />

        {/* Main gradient line */}
        <line
          x1="32" y1="32"
          x2="32" y2={`calc(100% - 32px)`}
          stroke="url(#lineGradient)"
          strokeWidth="4"
          strokeLinecap="round"
          filter="url(#glow)"
        />

        {/* Inner bright core */}
        <line
          x1="32" y1="32"
          x2="32" y2={`calc(100% - 32px)`}
          stroke="white"
          strokeWidth="1"
          strokeLinecap="round"
          opacity="0.5"
        />

        {/* Animated traveling dot */}
        <circle r="6" fill="white" filter="url(#glowPulse)">
          <animate
            attributeName="opacity"
            values="0.8;1;0.8"
            dur="2s"
            repeatCount="indefinite"
          />
          <animateMotion
            dur="4s"
            repeatCount="indefinite"
            path="M32,32 L32,500"
          />
        </circle>
      </svg>

      {/* Events */}
      <div className="space-y-6 ml-16">
        {events.map((event, index) => {
          const config = getEventConfig(event.type, event.severity);
          const Icon = config.icon;
          const isHovered = hoveredId === event.id;

          return (
            <div
              key={event.id}
              className={`relative flex items-start gap-4 transition-all duration-300 ${isHovered ? 'translate-x-2' : ''}`}
              onMouseEnter={() => setHoveredId(event.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              {/* Node */}
              <div className="absolute -left-16 top-0 z-10">
                {/* Glow effect */}
                <div
                  className={`absolute inset-0 w-14 h-14 rounded-full blur-xl opacity-60 ${config.pulse ? 'animate-pulse' : ''}`}
                  style={{ background: `linear-gradient(135deg, ${config.lineColor}, ${config.lineColor}88)` }}
                ></div>

                {/* Main circle */}
                <div className={`relative w-14 h-14 rounded-full bg-gradient-to-br ${config.gradient} flex items-center justify-center shadow-2xl border-2 border-white/30`}>
                  <Icon className="w-6 h-6 text-white drop-shadow-lg" />

                  {/* Status indicator */}
                  {event.status === 'completed' && (
                    <div className="absolute -bottom-1 -right-1 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center border-2 border-slate-900 shadow-lg">
                      <CheckCircle className="w-3 h-3 text-white" />
                    </div>
                  )}
                  {event.status === 'pending' && (
                    <div className="absolute -bottom-1 -right-1 w-5 h-5 bg-yellow-500 rounded-full flex items-center justify-center border-2 border-slate-900 animate-pulse shadow-lg">
                      <Clock className="w-3 h-3 text-white" />
                    </div>
                  )}
                </div>
              </div>

              {/* Content Card */}
              <div className={`flex-1 relative`}>
                {/* Glassmorphism card */}
                <div className={`
                  relative overflow-hidden rounded-xl 
                  backdrop-blur-md bg-slate-900/70
                  border ${config.border}/40
                  p-5 
                  transition-all duration-300
                  hover:bg-slate-800/80 hover:border-opacity-70
                  ${isHovered ? `shadow-2xl ${config.glow}` : 'shadow-lg'}
                `}>
                  {/* Gradient accent line at top */}
                  <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${config.gradient}`}></div>

                  {/* Header */}
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1 pr-4">
                      <h4 className="text-lg font-bold text-white flex items-center gap-2 flex-wrap">
                        {event.title}
                        {event.severity && (
                          <span className={`px-2 py-0.5 text-xs font-bold uppercase rounded ${config.bg} ${config.border}/50 border`}>
                            {event.severity}
                          </span>
                        )}
                      </h4>
                      <p className="text-sm text-slate-400 mt-1">{event.description}</p>
                    </div>

                    {/* Time badge */}
                    <div className="text-right flex-shrink-0">
                      <div className="text-sm font-mono text-cyan-400 font-semibold">{formatTime(event.timestamp)}</div>
                      <div className="text-xs text-slate-500">{getRelativeTime(event.timestamp)}</div>
                    </div>
                  </div>

                  {/* Tags */}
                  <div className="flex flex-wrap gap-2 mt-3">
                    {event.entity && (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 text-slate-300 text-sm border border-slate-700/50">
                        <Target className="w-3.5 h-3.5 text-cyan-400" />
                        {event.entity}
                      </span>
                    )}
                    {event.technique_id && (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 text-sm font-mono border border-cyan-500/30 font-semibold">
                        <Shield className="w-3.5 h-3.5" />
                        {event.technique_id}
                      </span>
                    )}
                  </div>

                  {/* Hover arrow */}
                  <ChevronRight className={`absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-600 transition-all duration-300 ${isHovered ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-2'}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
