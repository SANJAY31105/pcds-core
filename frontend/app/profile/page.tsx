'use client';

import { User, Mail, Shield, Key, Bell, Save } from 'lucide-react';

export default function ProfilePage() {
    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white">Profile Settings</h1>
                <p className="text-[#666] text-sm mt-1">Manage your account preferences</p>
            </div>

            {/* Profile Info */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <div className="flex items-center gap-4 mb-6">
                    <div className="w-16 h-16 bg-[#10a37f] rounded-full flex items-center justify-center">
                        <User className="w-8 h-8 text-white" />
                    </div>
                    <div>
                        <h2 className="text-lg font-medium text-white">Admin User</h2>
                        <p className="text-sm text-[#666]">Security Administrator</p>
                    </div>
                </div>

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm text-[#a1a1a1] mb-1.5">Full Name</label>
                        <input
                            type="text"
                            defaultValue="Admin User"
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f] text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-[#a1a1a1] mb-1.5">Email</label>
                        <input
                            type="email"
                            defaultValue="admin@pcds.com"
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f] text-sm"
                        />
                    </div>
                </div>
            </div>

            {/* Security */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <h3 className="text-base font-medium text-white mb-4 flex items-center gap-2">
                    <Shield className="w-4 h-4 text-[#10a37f]" /> Security
                </h3>
                <div className="space-y-4">
                    <div>
                        <label className="block text-sm text-[#a1a1a1] mb-1.5">Current Password</label>
                        <input
                            type="password"
                            placeholder="••••••••"
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-white placeholder-[#666] focus:outline-none focus:border-[#10a37f] text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-[#a1a1a1] mb-1.5">New Password</label>
                        <input
                            type="password"
                            placeholder="••••••••"
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-white placeholder-[#666] focus:outline-none focus:border-[#10a37f] text-sm"
                        />
                    </div>
                </div>
            </div>

            {/* Notifications */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <h3 className="text-base font-medium text-white mb-4 flex items-center gap-2">
                    <Bell className="w-4 h-4 text-[#10a37f]" /> Notifications
                </h3>
                <div className="space-y-3">
                    {['Critical alerts', 'High severity detections', 'Daily summary', 'Weekly reports'].map((item, i) => (
                        <label key={i} className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a] cursor-pointer">
                            <span className="text-sm text-[#a1a1a1]">{item}</span>
                            <input type="checkbox" defaultChecked={i < 2} className="w-4 h-4 accent-[#10a37f]" />
                        </label>
                    ))}
                </div>
            </div>

            {/* Save */}
            <button className="w-full py-2.5 bg-[#10a37f] text-white text-sm font-medium rounded-lg hover:bg-[#0d8a6a] transition-colors flex items-center justify-center gap-2">
                <Save className="w-4 h-4" /> Save Changes
            </button>
        </div>
    );
}
