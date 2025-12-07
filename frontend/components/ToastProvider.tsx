'use client';

import { createContext, useContext, useState, ReactNode } from 'react';
import { AlertTriangle, CheckCircle, XCircle, Info, X } from 'lucide-react';

interface Toast {
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    title: string;
    message: string;
}

interface ToastContextType {
    toasts: Toast[];
    addToast: (type: Toast['type'], title: string, message: string) => void;
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

    const addToast = (type: Toast['type'], title: string, message: string) => {
        const id = Date.now().toString();
        setToasts(prev => [...prev, { id, type, title, message }]);
        setTimeout(() => removeToast(id), 5000); // Auto-remove after 5s
    };

    const removeToast = (id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    };

    return (
        <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
            {children}
            <ToastContainer toasts={toasts} removeToast={removeToast} />
        </ToastContext.Provider>
    );
}

function ToastContainer({ toasts, removeToast }: { toasts: Toast[], removeToast: (id: string) => void }) {
    const getIcon = (type: Toast['type']) => {
        switch (type) {
            case 'success': return <CheckCircle className="w-5 h-5 text-green-400" />;
            case 'error': return <XCircle className="w-5 h-5 text-red-400" />;
            case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
            default: return <Info className="w-5 h-5 text-blue-400" />;
        }
    };

    const getColors = (type: Toast['type']) => {
        switch (type) {
            case 'success': return 'border-green-500/50 bg-green-500/10';
            case 'error': return 'border-red-500/50 bg-red-500/10';
            case 'warning': return 'border-yellow-500/50 bg-yellow-500/10';
            default: return 'border-blue-500/50 bg-blue-500/10';
        }
    };

    return (
        <div className="fixed top-4 right-4 z-50 space-y-2 max-w-md">
            {toasts.map(toast => (
                <div
                    key={toast.id}
                    className={`flex items-start gap-3 p-4 rounded-lg border backdrop-blur-sm shadow-lg animate-slide-in ${getColors(toast.type)}`}
                >
                    {getIcon(toast.type)}
                    <div className="flex-1">
                        <h4 className="font-semibold text-white">{toast.title}</h4>
                        <p className="text-sm text-gray-300">{toast.message}</p>
                    </div>
                    <button onClick={() => removeToast(toast.id)} className="text-gray-400 hover:text-white">
                        <X className="w-4 h-4" />
                    </button>
                </div>
            ))}
        </div>
    );
}
