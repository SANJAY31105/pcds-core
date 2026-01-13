'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard, Shield, Target, FileText,
    Activity, AlertTriangle, ClipboardList, Crosshair, Bot, Bell, Clock,
    Brain, Share2, Sparkles, BarChart3, CheckCircle, DollarSign
} from 'lucide-react';
const navigation = [
    // Key demo pages at top (in demo order)
    { name: 'Overview', href: '/', icon: LayoutDashboard },
    { name: 'Live Feed', href: '/live', icon: Activity },
    { name: 'AI Copilot', href: '/copilot', icon: Sparkles },
    { name: 'Timeline', href: '/timeline', icon: Clock },
    { name: 'Playbooks', href: '/playbooks', icon: Bot },
    { name: 'MITRE', href: '/mitre', icon: Shield },
    { name: 'Reports', href: '/reports', icon: FileText },
    { name: 'Detections', href: '/detections', icon: AlertTriangle },
    // Other pages
    { name: 'Entities', href: '/entities', icon: Target },
    { name: 'Alerts', href: '/alerts', icon: Bell },
    { name: 'Investigations', href: '/investigations', icon: ClipboardList },
    { name: 'Hunt', href: '/hunt', icon: Crosshair },
    { name: 'ML Metrics', href: '/ml-metrics', icon: BarChart3 },
    { name: 'Validation', href: '/validation', icon: CheckCircle },
    { name: 'Pricing', href: '/pricing', icon: DollarSign },
    { name: 'ML Hub', href: '/ml', icon: Brain },
    { name: 'SIEM', href: '/siem', icon: Share2 },
];

interface NavigationProps {
    collapsed?: boolean;
}

export default function Navigation({ collapsed = false }: NavigationProps) {
    const pathname = usePathname();

    return (
        <nav className={`fixed left-0 top-0 h-screen bg-[#0a0a0a] border-r border-[#2a2a2a] transition-all duration-300 ${collapsed ? 'w-16' : 'w-64'}`}>
            {/* Logo */}
            <div className="p-4 border-b border-[#2a2a2a]">
                <div className={`flex items-center ${collapsed ? 'justify-center' : 'gap-3'}`}>
                    <div className="w-9 h-9 bg-[#10a37f] rounded-lg flex items-center justify-center flex-shrink-0">
                        <Shield className="w-5 h-5 text-white" />
                    </div>
                    {!collapsed && (
                        <div>
                            <h1 className="text-lg font-semibold text-white">PCDS</h1>
                            <p className="text-xs text-[#666]">Enterprise NDR</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Navigation Links */}
            <div className={`p-2 space-y-0.5 pb-24 overflow-y-auto max-h-[calc(100vh-160px)] ${collapsed ? 'px-2' : 'p-3'}`}>
                {navigation.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.href;

                    return (
                        <Link
                            key={item.name}
                            href={item.href}
                            title={collapsed ? item.name : undefined}
                            className={`
                                flex items-center gap-3 rounded-lg transition-colors text-sm
                                ${collapsed ? 'justify-center p-2.5' : 'px-3 py-2.5'}
                                ${isActive
                                    ? 'bg-[#1a1a1a] text-white'
                                    : 'text-[#a1a1a1] hover:text-white hover:bg-[#141414]'
                                }
                            `}
                        >
                            <Icon className={`w-[18px] h-[18px] flex-shrink-0 ${isActive ? 'text-[#10a37f]' : ''}`} />
                            {!collapsed && <span className="font-medium">{item.name}</span>}
                        </Link>
                    );
                })}
            </div>

            {/* Status Indicator */}
            <div className="absolute bottom-0 left-0 right-0 p-3 border-t border-[#2a2a2a] bg-[#0a0a0a]">
                <div className={`flex items-center gap-2 rounded-lg bg-[#141414] ${collapsed ? 'justify-center p-2' : 'px-3 py-2'}`}>
                    <div className="w-2 h-2 rounded-full bg-[#22c55e]"></div>
                    {!collapsed && <span className="text-sm text-[#a1a1a1]">Online</span>}
                </div>
            </div>
        </nav>
    );
}
