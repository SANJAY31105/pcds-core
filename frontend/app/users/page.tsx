'use client';

import { useState } from 'react';
import { Users, UserPlus, Shield, Mail, Clock, MoreVertical, Search, Filter } from 'lucide-react';

interface User {
    id: string;
    name: string;
    email: string;
    role: 'admin' | 'analyst' | 'viewer';
    status: 'active' | 'inactive';
    lastActive: string;
    avatar?: string;
}

const mockUsers: User[] = [
    { id: '1', name: 'Sanjay Kumar', email: 'sanjay@pcds.local', role: 'admin', status: 'active', lastActive: '2024-12-08T10:30:00Z' },
    { id: '2', name: 'Alex Chen', email: 'alex.chen@pcds.local', role: 'analyst', status: 'active', lastActive: '2024-12-08T09:45:00Z' },
    { id: '3', name: 'Sarah Miller', email: 'sarah.m@pcds.local', role: 'analyst', status: 'active', lastActive: '2024-12-07T16:20:00Z' },
    { id: '4', name: 'James Wilson', email: 'j.wilson@pcds.local', role: 'viewer', status: 'inactive', lastActive: '2024-12-01T11:00:00Z' },
    { id: '5', name: 'Emily Davis', email: 'emily.d@pcds.local', role: 'analyst', status: 'active', lastActive: '2024-12-08T08:15:00Z' },
];

export default function UsersPage() {
    const [users] = useState<User[]>(mockUsers);
    const [searchQuery, setSearchQuery] = useState('');
    const [showAddModal, setShowAddModal] = useState(false);
    const [newUser, setNewUser] = useState({ name: '', email: '', role: 'analyst' });

    const filteredUsers = users.filter(user =>
        user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        user.email.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const getRoleBadge = (role: string) => {
        const colors = {
            admin: 'bg-purple-500/20 text-purple-400',
            analyst: 'bg-blue-500/20 text-blue-400',
            viewer: 'bg-gray-500/20 text-gray-400'
        };
        return colors[role as keyof typeof colors] || colors.viewer;
    };

    const getStatusBadge = (status: string) => {
        return status === 'active'
            ? 'bg-green-500/20 text-green-400'
            : 'bg-red-500/20 text-red-400';
    };

    const handleAddUser = () => {
        // In production, this would call an API
        console.log('Adding user:', newUser);
        setShowAddModal(false);
        setNewUser({ name: '', email: '', role: 'analyst' });
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-semibold text-white">User Management</h1>
                    <p className="text-[#666] mt-1">Manage analyst access and permissions</p>
                </div>
                <button
                    onClick={() => setShowAddModal(true)}
                    className="flex items-center gap-2 px-5 py-2.5 bg-[#10a37f] text-white rounded-lg text-sm font-medium hover:bg-[#0d8a6a] transition-colors"
                >
                    <UserPlus className="w-4 h-4" />
                    Add User
                </button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Users className="w-5 h-5 text-[#10a37f]" />
                        <span className="text-sm text-[#666]">Total Users</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{users.length}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Shield className="w-5 h-5 text-purple-400" />
                        <span className="text-sm text-[#666]">Admins</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{users.filter(u => u.role === 'admin').length}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Users className="w-5 h-5 text-blue-400" />
                        <span className="text-sm text-[#666]">Analysts</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{users.filter(u => u.role === 'analyst').length}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Clock className="w-5 h-5 text-green-400" />
                        <span className="text-sm text-[#666]">Active Now</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{users.filter(u => u.status === 'active').length}</p>
                </div>
            </div>

            {/* Search & Filter */}
            <div className="flex items-center gap-4">
                <div className="flex-1 relative">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-[#666]" />
                    <input
                        type="text"
                        placeholder="Search users..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-12 pr-4 py-3 bg-[#141414] border border-[#2a2a2a] rounded-lg text-white placeholder-[#666] focus:outline-none focus:border-[#10a37f]"
                    />
                </div>
                <button className="flex items-center gap-2 px-4 py-3 bg-[#141414] border border-[#2a2a2a] rounded-lg text-[#a1a1a1] hover:text-white transition-colors">
                    <Filter className="w-5 h-5" />
                    Filter
                </button>
            </div>

            {/* Users Table */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] overflow-hidden">
                <table className="w-full">
                    <thead>
                        <tr className="border-b border-[#2a2a2a]">
                            <th className="text-left px-6 py-4 text-xs text-[#666] font-medium uppercase tracking-wider">User</th>
                            <th className="text-left px-6 py-4 text-xs text-[#666] font-medium uppercase tracking-wider">Role</th>
                            <th className="text-left px-6 py-4 text-xs text-[#666] font-medium uppercase tracking-wider">Status</th>
                            <th className="text-left px-6 py-4 text-xs text-[#666] font-medium uppercase tracking-wider">Last Active</th>
                            <th className="text-right px-6 py-4 text-xs text-[#666] font-medium uppercase tracking-wider">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredUsers.map((user) => (
                            <tr key={user.id} className="border-b border-[#1a1a1a] hover:bg-[#1a1a1a] transition-colors">
                                <td className="px-6 py-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#10a37f] to-[#0d8a6a] flex items-center justify-center text-white font-medium">
                                            {user.name.split(' ').map(n => n[0]).join('')}
                                        </div>
                                        <div>
                                            <p className="text-white font-medium">{user.name}</p>
                                            <p className="text-sm text-[#666]">{user.email}</p>
                                        </div>
                                    </div>
                                </td>
                                <td className="px-6 py-4">
                                    <span className={`px-2.5 py-1 rounded-full text-xs font-medium capitalize ${getRoleBadge(user.role)}`}>
                                        {user.role}
                                    </span>
                                </td>
                                <td className="px-6 py-4">
                                    <span className={`px-2.5 py-1 rounded-full text-xs font-medium capitalize ${getStatusBadge(user.status)}`}>
                                        {user.status}
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-sm text-[#a1a1a1]">
                                    {new Date(user.lastActive).toLocaleString()}
                                </td>
                                <td className="px-6 py-4 text-right">
                                    <button className="p-2 hover:bg-[#2a2a2a] rounded-lg transition-colors">
                                        <MoreVertical className="w-5 h-5 text-[#666]" />
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Add User Modal */}
            {showAddModal && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6 w-full max-w-md">
                        <h2 className="text-xl font-semibold text-white mb-4">Add New User</h2>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-[#666] mb-2">Full Name</label>
                                <input
                                    type="text"
                                    value={newUser.name}
                                    onChange={(e) => setNewUser({ ...newUser, name: e.target.value })}
                                    className="w-full px-4 py-3 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f]"
                                    placeholder="John Doe"
                                />
                            </div>
                            <div>
                                <label className="block text-sm text-[#666] mb-2">Email</label>
                                <input
                                    type="email"
                                    value={newUser.email}
                                    onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                                    className="w-full px-4 py-3 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f]"
                                    placeholder="john@company.com"
                                />
                            </div>
                            <div>
                                <label className="block text-sm text-[#666] mb-2">Role</label>
                                <select
                                    value={newUser.role}
                                    onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
                                    className="w-full px-4 py-3 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f]"
                                >
                                    <option value="viewer">Viewer</option>
                                    <option value="analyst">Analyst</option>
                                    <option value="admin">Admin</option>
                                </select>
                            </div>
                        </div>

                        <div className="flex gap-3 mt-6">
                            <button
                                onClick={() => setShowAddModal(false)}
                                className="flex-1 px-4 py-2.5 bg-[#2a2a2a] text-white rounded-lg hover:bg-[#3a3a3a] transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleAddUser}
                                className="flex-1 px-4 py-2.5 bg-[#10a37f] text-white rounded-lg hover:bg-[#0d8a6a] transition-colors"
                            >
                                Add User
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
