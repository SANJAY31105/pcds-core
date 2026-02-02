'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard, Shield, Target, FileText,
    Activity, AlertTriangle, ClipboardList, Crosshair, Bot, Bell, Clock,
    Brain, Share2, Sparkles, BarChart3, CheckCircle, DollarSign, Download, Key
} from 'lucide-react';

const navigation = [
    // Key demo pages at top (in demo order)
    { name: 'Overview', href: '/dashboard', icon: LayoutDashboard },
    { name: 'Live Feed', href: '/live', icon: Activity },
    { name: 'AI Copilot', href: '/copilot', icon: Sparkles },
    { name: 'Timeline', href: '/timeline', icon: Clock },
    { name: 'Playbooks', href: '/playbooks', icon: Bot },
    { name: 'MITRE', href: '/mitre', icon: Shield },
    { name: 'Reports', href: '/reports', icon: FileText },
    { name: 'Detections', href: '/detections', icon: AlertTriangle },
    { name: 'Download Agent', href: '/download', icon: Download },
    { name: 'API Keys', href: '/settings/api-keys', icon: Key },
    // Other pages
    { name: 'Entities', href: '/entities', icon: Target },
    { name: 'Alerts', href: '/alerts', icon: Bell },
    { name: 'Investigations', href: '/investigations', icon: ClipboardList },
    { name: 'Hunt', href: '/hunt', icon: Crosshair },
    { name: 'ML Metrics', href: '/ml-metrics', icon: BarChart3 },
    { name: 'ML Hub', href: '/ml', icon: Brain },
    { name: 'SIEM', href: '/siem', icon: Share2 },
];

interface NavigationProps {
    collapsed?: boolean;
    mobileOpen?: boolean;
    setMobileOpen?: (open: boolean) => void;
}

export default function Navigation({ collapsed = false, mobileOpen = false, setMobileOpen }: NavigationProps) {
    const pathname = usePathname();

    return (
        <>
            {/* Mobile Backdrop */}
            {mobileOpen && (
                <div
                    className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden transition-opacity duration-300"
                    onClick={() => setMobileOpen?.(false)}
                />
            )}

            <nav className={`
                fixed left-0 top-0 h-screen bg-[#0f0f0f] transition-all duration-300 ease-out z-50
                ${collapsed ? 'md:w-[72px]' : 'md:w-60'}
                w-60
                ${mobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
            `}>
                {/* Logo */}
                <div className="h-14 flex items-center px-4">
                    <div className={`flex items-center ${collapsed ? 'md:justify-center' : 'gap-3'}`}>
                        <div className="w-8 h-8 bg-[#10a37f] rounded-lg flex items-center justify-center flex-shrink-0">
                            <Shield className="w-4 h-4 text-white" />
                        </div>
                        {(!collapsed || mobileOpen) && (
                            <div className={`${collapsed ? 'md:hidden' : 'block'}`}>
                                <h1 className="text-base font-semibold text-white tracking-tight">PCDS</h1>
                                <p className="text-[10px] text-[#717171] -mt-0.5">Enterprise NDR</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Navigation Links */}
                <div className={`px-3 py-2 overflow-y-auto max-h-[calc(100vh-120px)] scrollbar-thin scrollbar-thumb-[#333] scrollbar-track-transparent`}>
                    {navigation.map((item) => {
                        const Icon = item.icon;
                        const isActive = pathname === item.href;

                        return (
                            <Link
                                key={item.name}
                                href={item.href}
                                onClick={() => setMobileOpen?.(false)}
                                title={collapsed ? item.name : undefined}
                                className={`
                                    flex items-center gap-3 rounded-lg transition-all duration-200 ease-out
                                    ${collapsed ? 'md:justify-center md:px-0 md:py-2 px-3 py-2' : 'px-3 py-2'}
                                    ${isActive
                                        ? 'bg-[#272727] text-white'
                                        : 'text-[#aaa] hover:bg-[#1a1a1a] hover:text-white'
                                    }
                                    mb-0.5
                                `}
                            >
                                <Icon className={`w-5 h-5 flex-shrink-0 transition-colors duration-200 ${isActive ? 'text-[#10a37f]' : ''}`} />
                                {(!collapsed || mobileOpen) && (
                                    <span className={`text-sm font-normal ${collapsed ? 'md:hidden' : 'block'}`}>{item.name}</span>
                                )}
                            </Link>
                        );
                    })}
                </div>

                {/* Status Indicator */}
                <div className="absolute bottom-0 left-0 right-0 p-3 bg-[#0f0f0f]">
                    <div className={`flex items-center gap-2 ${collapsed ? 'md:justify-center' : ''}`}>
                        <div className="w-2 h-2 rounded-full bg-[#22c55e] animate-pulse"></div>
                        {(!collapsed || mobileOpen) && (
                            <span className={`text-xs text-[#717171] ${collapsed ? 'md:hidden' : 'block'}`}>Online</span>
                        )}
                    </div>
                </div>
            </nav>
        </>
    );
}

