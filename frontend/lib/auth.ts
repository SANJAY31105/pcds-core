// Auth utility functions for frontend
// Handles token storage, login, logout, and authentication checks

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface User {
    id: string;
    username: string;
    email: string;
    role: string;
    is_active: boolean;
    created_at: string;
    last_login?: string;
}

export interface LoginResponse {
    access_token: string;
    refresh_token: string;
    token_type: string;
    user: User;
}

// Token management
export const setTokens = (accessToken: string, refreshToken: string) => {
    localStorage.setItem('access_token', accessToken);
    localStorage.setItem('refresh_token', refreshToken);
};

export const getAccessToken = (): string | null => {
    return localStorage.getItem('access_token');
};

export const getRefreshToken = (): string | null => {
    return localStorage.getItem('refresh_token');
};

export const clearTokens = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
};

// User management
export const setUser = (user: User) => {
    localStorage.setItem('user', JSON.stringify(user));
};

export const getUser = (): User | null => {
    const userStr = localStorage.getItem('user');
    return userStr ? JSON.parse(userStr) : null;
};

// Authentication functions
export const login = async (username: string, password: string): Promise<LoginResponse> => {
    const response = await fetch(`${API_URL}/api/v2/auth/login`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password, remember_me: true }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Login failed');
    }

    const data: LoginResponse = await response.json();

    // Store tokens and user
    setTokens(data.access_token, data.refresh_token);
    setUser(data.user);

    return data;
};

export const register = async (username: string, email: string, password: string, role: string = 'analyst'): Promise<LoginResponse> => {
    const response = await fetch(`${API_URL}/api/v2/auth/register`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, email, password, role }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Registration failed');
    }

    const data: LoginResponse = await response.json();

    // Store tokens and user
    setTokens(data.access_token, data.refresh_token);
    setUser(data.user);

    return data;
};

export const logout = () => {
    clearTokens();
    window.location.href = '/login';
};

export const isAuthenticated = (): boolean => {
    return !!getAccessToken();
};

export const refreshAccessToken = async (): Promise<string | null> => {
    const refreshToken = getRefreshToken();

    if (!refreshToken) {
        return null;
    }

    try {
        const response = await fetch(`${API_URL}/api/v2/auth/refresh`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${refreshToken}`,
            },
        });

        if (!response.ok) {
            clearTokens();
            return null;
        }

        const data = await response.json();
        localStorage.setItem('access_token', data.access_token);

        return data.access_token;
    } catch (error) {
        clearTokens();
        return null;
    }
};

export const getCurrentUser = async (): Promise<User | null> => {
    const token = getAccessToken();

    if (!token) {
        return null;
    }

    try {
        const response = await fetch(`${API_URL}/api/v2/auth/me`, {
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            if (response.status === 401) {
                // Try to refresh token
                const newToken = await refreshAccessToken();
                if (newToken) {
                    // Retry with new token
                    const retryResponse = await fetch(`${API_URL}/api/v2/auth/me`, {
                        headers: {
                            'Authorization': `Bearer ${newToken}`,
                        },
                    });

                    if (retryResponse.ok) {
                        const user = await retryResponse.json();
                        setUser(user);
                        return user;
                    }
                }
            }
            return null;
        }

        const user = await response.json();
        setUser(user);
        return user;
    } catch (error) {
        return null;
    }
};
