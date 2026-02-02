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
                <div className="flex min-h-screen bg-[#0f0f0f]">
                    <Navigation collapsed={collapsed} mobileOpen={mobileOpen} setMobileOpen={setMobileOpen} />
                    <div className={`flex-1 transition-all duration-300 ease-out ${collapsed ? 'md:ml-[72px]' : 'md:ml-60'} ml-0`}>
                        {/* Clean Header - YouTube-like */}
                        <header className="sticky top-0 z-30 bg-[#0f0f0f] h-14 flex items-center px-4">
                            <div className="flex items-center justify-between w-full">
                                <div className="flex items-center gap-3">
                                    {/* Mobile Toggle */}
                                    <button
                                        onClick={() => setMobileOpen(true)}
                                        className="md:hidden p-2 rounded-full hover:bg-[#272727] text-[#aaa] hover:text-white transition-colors duration-200"
                                    >
                                        <Menu className="w-5 h-5" />
                                    </button>

                                    {/* Sidebar Toggle Button (Desktop) */}
                                    <button
                                        onClick={() => setCollapsed(!collapsed)}
                                        className="hidden md:block p-2 rounded-full hover:bg-[#272727] text-[#aaa] hover:text-white transition-colors duration-200"
                                        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                                    >
                                        {collapsed ? (
                                            <PanelLeft className="w-5 h-5" />
                                        ) : (
                                            <PanelLeftClose className="w-5 h-5" />
                                        )}
                                    </button>
                                    <div>
                                        <h2 className="text-sm font-medium text-white">PCDS Enterprise</h2>
                                        <p className="text-[10px] text-[#717171]">Network Detection & Response</p>
                                    </div>
                                    <GlobalSearch />
                                </div>

                                {/* Right side - User info */}
                                <UserBadge />
                            </div>
                        </header>

                        {/* Main Content */}
                        <main className="p-4 md:p-6">
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
