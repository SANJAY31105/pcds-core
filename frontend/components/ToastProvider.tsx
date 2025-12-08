'use client';

import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { X, AlertTriangle, CheckCircle, Info, AlertCircle } from 'lucide-react';

interface Toast {
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    title: string;
    message?: string;
    duration?: number;
}

interface ToastContextType {
    toasts: Toast[];
    addToast: (toast: Omit<Toast, 'id'>) => void;
    removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function useToast() {
    const context = useContext(ToastContext);
    if (!context) throw new Error('useToast must be used within ToastProvider');
    return context;
}

export function ToastProvider({ children }: { children: ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
        const id = Date.now().toString();
        const newToast = { ...toast, id };
        setToasts(prev => [...prev, newToast]);

        // Auto remove after duration
        setTimeout(() => {
            removeToast(id);
        }, toast.duration || 5000);
    }, []);

    const removeToast = useCallback((id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    }, []);

    return (
        <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
            {children}
            <ToastContainer toasts={toasts} removeToast={removeToast} />
        </ToastContext.Provider>
    );
}

function ToastContainer({ toasts, removeToast }: { toasts: Toast[]; removeToast: (id: string) => void }) {
    if (toasts.length === 0) return null;

    const getIcon = (type: string) => {
        switch (type) {
            case 'success': return <CheckCircle className="w-5 h-5 text-[#22c55e]" />;
            case 'error': return <AlertCircle className="w-5 h-5 text-[#ef4444]" />;
            case 'warning': return <AlertTriangle className="w-5 h-5 text-[#f97316]" />;
            default: return <Info className="w-5 h-5 text-[#3b82f6]" />;
        }
    };

    const getBorderColor = (type: string) => {
        switch (type) {
            case 'success': return 'border-[#22c55e]/30';
            case 'error': return 'border-[#ef4444]/30';
            case 'warning': return 'border-[#f97316]/30';
            default: return 'border-[#3b82f6]/30';
        }
    };

    return (
        <div className="fixed bottom-4 right-4 z-50 space-y-2">
            {toasts.map((toast) => (
                <div
                    key={toast.id}
                    className={`flex items-start gap-3 p-4 bg-[#141414] rounded-xl border ${getBorderColor(toast.type)} shadow-xl animate-slideIn min-w-[320px] max-w-md`}
                >
                    {getIcon(toast.type)}
                    <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-white">{toast.title}</p>
                        {toast.message && (
                            <p className="text-xs text-[#666] mt-0.5">{toast.message}</p>
                        )}
                    </div>
                    <button onClick={() => removeToast(toast.id)} className="text-[#666] hover:text-white">
                        <X className="w-4 h-4" />
                    </button>
                </div>
            ))}
        </div>
    );
}

// Convenience hooks
export function useSuccessToast() {
    const { addToast } = useToast();
    return (title: string, message?: string) => addToast({ type: 'success', title, message });
}

export function useErrorToast() {
    const { addToast } = useToast();
    return (title: string, message?: string) => addToast({ type: 'error', title, message });
}

export function useWarningToast() {
    const { addToast } = useToast();
    return (title: string, message?: string) => addToast({ type: 'warning', title, message });
}
