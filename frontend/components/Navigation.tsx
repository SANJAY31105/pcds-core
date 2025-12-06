'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard, Shield, Target, Search, FileText,
    Grid3x3, Settings, Activity, AlertTriangle, ClipboardList, Crosshair, ShieldAlert, Bot
} from 'lucide-react';

const navigation = [
    { name: 'Overview', href: '/', icon: LayoutDashboard },
    { name: 'Entities', href: '/entities', icon: Target },
    { name: 'Detections', href: '/detections', icon: AlertTriangle },
    { name: 'Investigations', href: '/investigations', icon: ClipboardList },
    { name: 'Playbooks', href: '/playbooks', icon: Bot },
    { name: 'Hunt', href: '/hunt', icon: Crosshair },
    { name: 'MITRE', href: '/mitre', icon: ShieldAlert },
    { name: 'Live Feed', href: '/live', icon: Activity },
    { name: 'Reports', href: '/reports', icon: FileText },
];

export default function Navigation() {
    const pathname = usePathname();

    return (
        <nav className="fixed left-0 top-0 h-screen w-64 bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 border-r border-cyan-500/20 shadow-2xl shadow-cyan-500/10">
            {/* Logo */}
            <div className="p-6 border-b border-cyan-500/20">
                <div className="flex items-center space-x-3">
                    <div className="relative">
                        <Shield className="w-10 h-10 text-cyan-400" strokeWidth={1.5} />
                        <div className="absolute inset-0 blur-xl bg-cyan-400/30"></div>
                    </div>
                    <div>
                        <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                            PCDS
                        </h1>
                        <p className="text-xs text-slate-400">Enterprise NDR</p>
                    </div>
                </div>
            </div>

            {/* Navigation Links */}
            <div className="p-4 space-y-1">
                {navigation.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.href;

                    return (
                        <Link
                            key={item.name}
                            href={item.href}
                            className={`
                                group flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200
                                ${isActive
                                    ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 shadow-lg shadow-cyan-500/20'
                                    : 'text-slate-400 hover:text-cyan-400 hover:bg-slate-800/50'
                                }
                            `}
                        >
                            <Icon className={`w-5 h-5 ${isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-cyan-400'}`} />
                            <span className="font-medium">{item.name}</span>
                            {isActive && (
                                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50"></div>
                            )}
                        </Link>
                    );
                })}
            </div>

            {/* Status Indicator */}
            <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-cyan-500/20">
                <div className="flex items-center space-x-2 px-4 py-2 bg-green-500/10 rounded-lg border border-green-500/20">
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse shadow-lg shadow-green-400/50"></div>
                    <span className="text-sm text-green-400 font-medium">System Operational</span>
                </div>
            </div>
        </nav>
    );
}
