'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { User, LogOut, Shield } from 'lucide-react';

interface UserData {
    email: string;
    role: string;
    full_name: string;
}

export default function UserMenu() {
    const router = useRouter();
    const [user, setUser] = useState<UserData | null>(null);
    const [open, setOpen] = useState(false);

    useEffect(() => {
        loadUser();
    }, []);

    const loadUser = async () => {
        try {
            const token = localStorage.getItem('access_token');
            if (!token) {
                console.log('No access token found');
                return;
            }

            const response = await fetch('http://localhost:8000/api/auth/me', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                setUser(data);
            } else {
                console.error('Failed to load user profile:', response.status);
                // Don't redirect here - let api.ts handle it
            }
        } catch (error) {
            console.error('Failed to load user:', error);
        }
    };

    const handleLogout = () => {
        localStorage.removeItem('access_token');
        router.push('/login');
    };

    if (!user) return null;

    const getRoleBadgeColor = (role: string) => {
        const colors = {
            super_admin: 'bg-purple-500/20 text-purple-400 border-purple-500/50',
            tenant_admin: 'bg-blue-500/20 text-blue-400 border-blue-500/50',
            analyst: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50',
            viewer: 'bg-slate-500/20 text-slate-400 border-slate-500/50'
        };
        return colors[role as keyof typeof colors] || colors.viewer;
    };

    return (
        <div className="relative">
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center space-x-3 px-4 py-2 bg-slate-800/50 border border-cyan-500/20 rounded-lg hover:bg-slate-800 transition-all"
            >
                <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center">
                    <User className="w-4 h-4 text-white" />
                </div>
                <div className="text-left hidden md:block">
                    <p className="text-sm font-medium text-white">{user.full_name || user.email}</p>
                    <p className="text-xs text-slate-400 capitalize">{user.role.replace('_', ' ')}</p>
                </div>
            </button>

            {open && (
                <>
                    {/* Backdrop */}
                    <div
                        className="fixed inset-0 z-40"
                        onClick={() => setOpen(false)}
                    />

                    {/* Dropdown */}
                    <div className="absolute right-0 mt-2 w-64 bg-slate-800 border border-cyan-500/20 rounded-lg shadow-2xl z-50 overflow-hidden">
                        {/* User Info */}
                        <div className="p-4 border-b border-slate-700">
                            <p className="font-medium text-white">{user.full_name || 'User'}</p>
                            <p className="text-sm text-slate-400 mt-1">{user.email}</p>
                            <div className="mt-3">
                                <span className={`inline-flex items-center px-2 py-1 text-xs font-medium rounded border ${getRoleBadgeColor(user.role)}`}>
                                    <Shield className="w-3 h-3 mr-1" />
                                    {user.role.replace('_', ' ').toUpperCase()}
                                </span>
                            </div>
                        </div>

                        {/* Actions */}
                        <div className="p-2">
                            <button
                                onClick={handleLogout}
                                className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-all"
                            >
                                <LogOut className="w-4 h-4" />
                                <span>Sign Out</span>
                            </button>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
