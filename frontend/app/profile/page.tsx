'use client';

import { useEffect, useState } from 'react';
import { getUser, logout, isAuthenticated } from '@/lib/auth';
import { useRouter } from 'next/navigation';
import { User, Shield, LogOut, Settings } from 'lucide-react';

export default function ProfilePage() {
    const router = useRouter();
    const [user, setUser] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!isAuthenticated()) {
            router.push('/login');
            return;
        }

        const currentUser = getUser();
        setUser(currentUser);
        setLoading(false);
    }, [router]);

    const handleLogout = () => {
        logout();
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="text-white">Loading...</div>
            </div>
        );
    }

    return (
        <div className="p-8 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                        Profile Settings
                    </h1>
                    <p className="text-slate-400 mt-2">Manage your account and preferences</p>
                </div>
                <button
                    onClick={handleLogout}
                    className="flex items-center space-x-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 transition-colors"
                >
                    <LogOut size={20} />
                    <span>Logout</span>
                </button>
            </div>

            {/* Profile Card */}
            <div className="bg-slate-800/50 border border-cyan-500/20 rounded-xl p-6 space-y-6">
                {/* User Info */}
                <div className="flex items-center space-x-4">
                    <div className="w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-full flex items-center justify-center">
                        <User size={40} className="text-white" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold text-white">{user?.username}</h2>
                        <p className="text-slate-400">{user?.email}</p>
                        <div className="mt-2 inline-flex items-center space-x-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/50 rounded-full">
                            <Shield size={16} className="text-cyan-400" />
                            <span className="text-sm text-cyan-400 capitalize">{user?.role}</span>
                        </div>
                    </div>
                </div>

                {/* Account Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-6 border-t border-slate-700/50">
                    <div>
                        <label className="text-sm text-slate-400">Username</label>
                        <p className="text-white font-medium mt-1">{user?.username}</p>
                    </div>
                    <div>
                        <label className="text-sm text-slate-400">Email</label>
                        <p className="text-white font-medium mt-1">{user?.email}</p>
                    </div>
                    <div>
                        <label className="text-sm text-slate-400">Role</label>
                        <p className="text-white font-medium mt-1 capitalize">{user?.role}</p>
                    </div>
                    <div>
                        <label className="text-sm text-slate-400">Status</label>
                        <p className={`font-medium mt-1 ${user?.is_active ? 'text-green-400' : 'text-red-400'}`}>
                            {user?.is_active ? 'Active' : 'Inactive'}
                        </p>
                    </div>
                    <div>
                        <label className="text-sm text-slate-400">Account Created</label>
                        <p className="text-white font-medium mt-1">
                            {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}
                        </p>
                    </div>
                    <div>
                        <label className="text-sm text-slate-400">Last Login</label>
                        <p className="text-white font-medium mt-1">
                            {user?.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
                        </p>
                    </div>
                </div>

                {/* Actions */}
                <div className="pt-6 border-t border-slate-700/50 flex space-x-4">
                    <button className="flex items-center space-x-2 px-4 py-2 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/50 rounded-lg text-cyan-400 transition-colors">
                        <Settings size={20} />
                        <span>Edit Profile</span>
                    </button>
                    <button className="px-4 py-2 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 rounded-lg text-slate-300 transition-colors">
                        Change Password
                    </button>
                </div>
            </div>

            {/* Security Info */}
            <div className="bg-blue-500/10 border border-blue-500/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-blue-400 mb-2">Security Information</h3>
                <p className="text-slate-300 text-sm">
                    Your account is secured with JWT authentication. Sessions automatically expire after 15 minutes of inactivity.
                    Refresh tokens are valid for 7 days.
                </p>
            </div>
        </div>
    );
}
