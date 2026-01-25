'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { supabase } from './supabase';
import type { User as SupabaseUser, Session } from '@supabase/supabase-js';

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
    signup: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
    logout: () => Promise<void>;
    getAccessToken: () => string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [session, setSession] = useState<Session | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const router = useRouter();

    // Convert Supabase user to our User format
    const mapSupabaseUser = (supabaseUser: SupabaseUser | null): User | null => {
        if (!supabaseUser) return null;
        return {
            user_id: supabaseUser.id,
            email: supabaseUser.email || '',
            name: supabaseUser.email?.split('@')[0] || 'User',
            role: 'user'
        };
    };

    const getAccessToken = useCallback(() => {
        return session?.access_token || null;
    }, [session]);

    // Check auth on mount
    useEffect(() => {
        // Get initial session
        supabase.auth.getSession().then(({ data: { session } }) => {
            setSession(session);
            setUser(mapSupabaseUser(session?.user || null));
            setIsLoading(false);
        });

        // Listen for auth changes
        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            setSession(session);
            setUser(mapSupabaseUser(session?.user || null));
            setIsLoading(false);
        });

        return () => subscription.unsubscribe();
    }, []);

    const login = async (email: string, password: string): Promise<boolean> => {
        try {
            const { data, error } = await supabase.auth.signInWithPassword({
                email,
                password
            });

            if (error) {
                console.error('Login error:', error.message);
                return false;
            }

            if (data.user) {
                setUser(mapSupabaseUser(data.user));
                setSession(data.session);
                return true;
            }
            return false;
        } catch (e) {
            console.error('Login exception:', e);
            return false;
        }
    };

    const signup = async (email: string, password: string): Promise<{ success: boolean; error?: string }> => {
        try {
            const { data, error } = await supabase.auth.signUp({
                email,
                password
            });

            if (error) {
                return { success: false, error: error.message };
            }

            if (data.user) {
                return { success: true };
            }
            return { success: false, error: 'Unknown error' };
        } catch (e: any) {
            return { success: false, error: e.message };
        }
    };

    const logout = async () => {
        await supabase.auth.signOut();
        setUser(null);
        setSession(null);
        router.push('/login');
    };

    return (
        <AuthContext.Provider value={{
            user,
            isAuthenticated: !!user,
            isLoading,
            login,
            signup,
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
