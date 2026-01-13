'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';

interface User {
    user_id: string;
    email: string;
    name: string;
    role: string;
}

interface AuthContextType {
    user: User | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    login: (email: string, password: string) => Promise<boolean>;
    logout: () => Promise<void>;
    getAccessToken: () => string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_BASE = 'http://localhost:8000/api/v2';
const TOKEN_REFRESH_INTERVAL = 13 * 60 * 1000; // 13 minutes (before 15 min expiry)

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const router = useRouter();

    const getAccessToken = useCallback(() => {
        if (typeof window !== 'undefined') {
            return localStorage.getItem('access_token');
        }
        return null;
    }, []);

    const refreshToken = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/auth/refresh`, {
                method: 'POST',
                credentials: 'include'
            });

            if (res.ok) {
                const data = await res.json();
                localStorage.setItem('access_token', data.access_token);
                return true;
            }
            return false;
        } catch {
            return false;
        }
    }, []);

    const checkAuth = useCallback(async () => {
        const token = getAccessToken();
        if (!token) {
            setIsLoading(false);
            return;
        }

        try {
            const res = await fetch(`${API_BASE}/auth/me`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.ok) {
                const userData = await res.json();
                setUser(userData);
            } else if (res.status === 401) {
                // Try to refresh
                const refreshed = await refreshToken();
                if (refreshed) {
                    await checkAuth();
                    return;
                }
                localStorage.removeItem('access_token');
                localStorage.removeItem('user');
            }
        } catch (error) {
            console.error('Auth check failed:', error);
        } finally {
            setIsLoading(false);
        }
    }, [getAccessToken, refreshToken]);

    // Check auth on mount
    useEffect(() => {
        checkAuth();
    }, [checkAuth]);

    // Silent refresh timer
    useEffect(() => {
        if (!user) return;

        const interval = setInterval(async () => {
            await refreshToken();
        }, TOKEN_REFRESH_INTERVAL);

        return () => clearInterval(interval);
    }, [user, refreshToken]);

    const login = async (email: string, password: string): Promise<boolean> => {
        try {
            const res = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ email, password })
            });

            if (res.ok) {
                const data = await res.json();
                localStorage.setItem('access_token', data.access_token);
                localStorage.setItem('user', JSON.stringify(data.user));
                setUser(data.user);
                return true;
            }
            return false;
        } catch {
            return false;
        }
    };

    const logout = async () => {
        const token = getAccessToken();
        try {
            await fetch(`${API_BASE}/auth/logout`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                credentials: 'include'
            });
        } catch {
            // Ignore errors
        }

        localStorage.removeItem('access_token');
        localStorage.removeItem('user');
        setUser(null);
        router.push('/login');
    };

    return (
        <AuthContext.Provider value={{
            user,
            isAuthenticated: !!user,
            isLoading,
            login,
            logout,
            getAccessToken
        }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within AuthProvider');
    }
    return context;
}

// Protected route wrapper
export function RequireAuth({ children }: { children: React.ReactNode }) {
    const { isAuthenticated, isLoading } = useAuth();
    const router = useRouter();

    useEffect(() => {
        if (!isLoading && !isAuthenticated) {
            router.push('/login');
        }
    }, [isAuthenticated, isLoading, router]);

    if (isLoading) {
        return (
            <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
                <div className="text-[#10a37f]">Loading...</div>
            </div>
        );
    }

    if (!isAuthenticated) {
        return null;
    }

    return <>{children}</>;
}
