'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard, Shield, Target, Search, FileText,
    Settings, Activity, AlertTriangle, ClipboardList, Crosshair, Bot, Bell, Clock
} from 'lucide-react';

const navigation = [
    { name: 'Overview', href: '/', icon: LayoutDashboard },
    { name: 'Entities', href: '/entities', icon: Target },
    { name: 'Detections', href: '/detections', icon: AlertTriangle },
    { name: 'Approvals', href: '/approvals', icon: Bell },
    { name: 'Timeline', href: '/timeline', icon: Clock },
    { name: 'Investigations', href: '/investigations', icon: ClipboardList },
    { name: 'Playbooks', href: '/playbooks', icon: Bot },
    { name: 'Hunt', href: '/hunt', icon: Crosshair },
    { name: 'MITRE', href: '/mitre', icon: Shield },
    { name: 'Live Feed', href: '/live', icon: Activity },
    { name: 'Reports', href: '/reports', icon: FileText },
];

export default function Navigation() {
    const pathname = usePathname();

    return (
        <nav className="fixed left-0 top-0 h-screen w-64 bg-[#0a0a0a] border-r border-[#2a2a2a]">
            {/* Logo */}
            <div className="p-5 border-b border-[#2a2a2a]">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 bg-[#10a37f] rounded-lg flex items-center justify-center">
                        <Shield className="w-5 h-5 text-white" />
                    </div>
                    <div>
                        <h1 className="text-lg font-semibold text-white">PCDS</h1>
                        <p className="text-xs text-[#666]">Enterprise NDR</p>
                    </div>
                </div>
            </div>

            {/* Navigation Links */}
            <div className="p-3 space-y-0.5 pb-24 overflow-y-auto max-h-[calc(100vh-160px)]">
                {navigation.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.href;

                    return (
                        <Link
                            key={item.name}
                            href={item.href}
                            className={`
                                flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-sm
                                ${isActive
                                    ? 'bg-[#1a1a1a] text-white'
                                    : 'text-[#a1a1a1] hover:text-white hover:bg-[#141414]'
                                }
                            `}
                        >
                            <Icon className={`w-[18px] h-[18px] ${isActive ? 'text-[#10a37f]' : ''}`} />
                            <span className="font-medium">{item.name}</span>
                        </Link>
                    );
                })}
            </div>

            {/* Status Indicator */}
            <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-[#2a2a2a] bg-[#0a0a0a]">
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#141414]">
                    <div className="w-2 h-2 rounded-full bg-[#22c55e]"></div>
                    <span className="text-sm text-[#a1a1a1]">System Operational</span>
                </div>
            </div>
        </nav>
    );
}
