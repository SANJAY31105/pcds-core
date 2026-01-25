'use client';

import { useState, createContext } from 'react';
import { usePathname } from 'next/navigation';
import Navigation from "@/components/Navigation";
import GlobalSearch from "@/components/GlobalSearch";
import { User, PanelLeftClose, PanelLeft, Menu } from 'lucide-react';
import { RequireAuth, useAuth } from '@/lib/AuthContext';

// Sidebar context
export const SidebarContext = createContext({
    collapsed: false,
    toggle: () => { },
});

// Public pages that don't require authentication
const PUBLIC_PAGES = ['/login', '/landing', '/get-started', '/pricing', '/'];

export default function AppLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const pathname = usePathname();
    const isPublicPage = PUBLIC_PAGES.includes(pathname);
    const [collapsed, setCollapsed] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);

    // Skip dashboard layout for public pages
    if (isPublicPage) {
        return <>{children}</>;
    }

    // All other pages require authentication
    return (
        <RequireAuth>
            <SidebarContext.Provider value={{ collapsed, toggle: () => setCollapsed(!collapsed) }}>
                <div className="flex min-h-screen bg-[#0a0a0a]">
                    <Navigation collapsed={collapsed} mobileOpen={mobileOpen} setMobileOpen={setMobileOpen} />
                    <div className={`flex-1 transition-all duration-300 ${collapsed ? 'md:ml-16' : 'md:ml-64'} ml-0`}>
                        {/* Clean Header */}
                        <header className="sticky top-0 z-30 bg-[#0a0a0a]/95 backdrop-blur-sm border-b border-[#2a2a2a] px-6 py-3">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    {/* Mobile Toggle */}
                                    <button
                                        onClick={() => setMobileOpen(true)}
                                        className="md:hidden p-2 rounded-lg hover:bg-[#141414] text-[#666] hover:text-white transition-colors"
                                    >
                                        <Menu className="w-5 h-5" />
                                    </button>

                                    {/* Sidebar Toggle Button (Desktop) */}
                                    <button
                                        onClick={() => setCollapsed(!collapsed)}
                                        className="hidden md:block p-2 rounded-lg hover:bg-[#141414] text-[#666] hover:text-white transition-colors"
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

                                {/* Right side - User info */}
                                <UserBadge />
                            </div>
                        </header>

                        {/* Main Content */}
                        <main className="p-6">
                            {children}
                        </main>
                    </div>
                </div>
            </SidebarContext.Provider>
        </RequireAuth>
    );
}

// Separate component to use useAuth hook
function UserBadge() {
    const { user, logout } = useAuth();

    // Get avatar URL or generate initials fallback
    const userData = user as any;
    const avatarUrl = userData?.user_metadata?.avatar_url ||
        `https://ui-avatars.com/api/?name=${encodeURIComponent(user?.name || 'User')}&background=random&color=fff`;

    return (
        <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#141414] border border-[#2a2a2a]">
                <img
                    src={avatarUrl}
                    alt="Profile"
                    className="w-5 h-5 rounded-full object-cover"
                />
                <span className="text-sm text-[#a1a1a1]">{user?.name || 'User'}</span>
            </div>
            <button
                onClick={logout}
                className="text-xs text-[#666] hover:text-red-400 transition-colors"
            >
                Logout
            </button>
        </div>
    );
}
