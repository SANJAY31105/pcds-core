'use client';

import { usePathname } from 'next/navigation';
import Navigation from "@/components/Navigation";
import UserMenu from "@/components/UserMenu";

export default function AppLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const pathname = usePathname();
    const isLoginPage = pathname === '/login';

    if (isLoginPage) {
        return <>{children}</>;
    }

    return (
        <div className="flex min-h-screen">
            <Navigation />
            <div className="flex-1 ml-64">
                {/* Header with User Menu */}
                <header className="sticky top-0 z-30 bg-slate-900/80 backdrop-blur-xl border-b border-cyan-500/20 px-8 py-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <h2 className="text-lg font-semibold text-white">PCDS Enterprise</h2>
                            <p className="text-sm text-slate-400">Network Detection & Response</p>
                        </div>
                        {/* <UserMenu /> */}
                        <div className="text-slate-400 text-sm">Logged In</div>
                    </div>
                </header>

                {/* Main Content */}
                <main className="p-8">
                    {children}
                </main>
            </div>
        </div>
    );
}
