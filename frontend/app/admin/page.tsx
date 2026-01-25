'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/lib/AuthContext';
import { useRouter } from 'next/navigation';

export default function AdminLeadsPage() {
    const { isAuthenticated, isLoading, getAccessToken } = useAuth();
    const [leads, setLeads] = useState<any[]>([]);
    const router = useRouter();

    useEffect(() => {
        if (!isLoading && !isAuthenticated) {
            router.push('/login');
        }
    }, [isLoading, isAuthenticated, router]);

    useEffect(() => {
        if (isAuthenticated) {
            fetchLeads();
        }
    }, [isAuthenticated]);

    const fetchLeads = async () => {
        const token = getAccessToken();
        const apiBase = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2`;
        try {
            const res = await fetch(`${apiBase}/leads`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            const data = await res.json();
            if (Array.isArray(data)) {
                setLeads(data);
            }
        } catch (err) {
            console.error(err);
        }
    };

    if (isLoading || !isAuthenticated) return <div className="p-10 text-white">Loading...</div>;

    return (
        <div className="min-h-screen bg-[#0a0a0a] text-white p-10 font-sans">
            <div className="max-w-4xl mx-auto">
                <div className="flex justify-between items-center mb-8">
                    <h1 className="text-3xl font-bold">Admin: Captured Leads</h1>
                    <button
                        onClick={() => router.push('/dashboard')}
                        className="px-4 py-2 bg-[#2a2a2a] rounded-lg hover:bg-[#333] transition-colors"
                    >
                        ← Back to Dashboard
                    </button>
                </div>

                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] overflow-hidden">
                    <table className="w-full">
                        <thead className="bg-[#1a1a1a]">
                            <tr>
                                <th className="p-4 text-left font-medium text-gray-400">Email</th>
                                <th className="p-4 text-left font-medium text-gray-400">Timestamp</th>
                            </tr>
                        </thead>
                        <tbody>
                            {leads.length === 0 ? (
                                <tr><td colSpan={2} className="p-8 text-center text-gray-500">No leads captured yet (Session Storage).</td></tr>
                            ) : (
                                leads.map((lead, i) => (
                                    <tr key={i} className="border-t border-[#2a2a2a] hover:bg-[#1a1a1a] transition-colors">
                                        <td className="p-4 font-mono text-[#f5c16c]">{lead.email}</td>
                                        <td className="p-4 text-gray-500 text-sm">{lead.timestamp}</td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
