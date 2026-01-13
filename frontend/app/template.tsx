'use client';

import { useState, createContext, useContext } from 'react';
import { usePathname } from 'next/navigation';
import Navigation from "@/components/Navigation";
import GlobalSearch from "@/components/GlobalSearch";
import { User, PanelLeftClose, PanelLeft } from 'lucide-react';

// Sidebar context
export const SidebarContext = createContext({
    collapsed: false,
    toggle: () => { },
});

export default function AppLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const pathname = usePathname();
    const isLoginPage = pathname === '/login';
    const [collapsed, setCollapsed] = useState(false);

    if (isLoginPage) {
        return <>{children}</>;
    }

    return (
        <SidebarContext.Provider value={{ collapsed, toggle: () => setCollapsed(!collapsed) }}>
            <div className="flex min-h-screen bg-[#0a0a0a]">
                <Navigation collapsed={collapsed} />
                <div className={`flex-1 transition-all duration-300 ${collapsed ? 'ml-16' : 'ml-64'}`}>
                    {/* Clean Header */}
                    <header className="sticky top-0 z-30 bg-[#0a0a0a]/95 backdrop-blur-sm border-b border-[#2a2a2a] px-6 py-3">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                {/* Sidebar Toggle Button */}
                                <button
                                    onClick={() => setCollapsed(!collapsed)}
                                    className="p-2 rounded-lg hover:bg-[#141414] text-[#666] hover:text-white transition-colors"
                                    title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                                >
                                    {collapsed ? (
                                        <PanelLeft className="w-5 h-5" />
                                    ) : (
                                        <PanelLeftClose className="w-5 h-5" />
                                    )}
                                </button>
                                <div>
                                    <h2 className="text-base font-medium text-white">PCDS Enterprise</h2>
                                    <p className="text-xs text-[#666]">Network Detection & Response</p>
                                </div>
                                <GlobalSearch />
                            </div>

                            {/* Right side */}
                            <div className="flex items-center gap-3">
                                <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#141414] border border-[#2a2a2a]">
                                    <User className="w-4 h-4 text-[#666]" />
                                    <span className="text-sm text-[#a1a1a1]">Admin</span>
                                </div>
                            </div>
                        </div>
                    </header>

                    {/* Main Content */}
                    <main className="p-6">
                        {children}
                    </main>
                </div>
            </div>
        </SidebarContext.Provider>
    );
}
