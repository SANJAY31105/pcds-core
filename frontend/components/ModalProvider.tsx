'use client';

import { useState, useEffect, createContext, useContext, ReactNode, useRef } from 'react';
import { X } from 'lucide-react';

interface ModalContextType {
    openModal: (content: ReactNode, title?: string) => void;
    closeModal: () => void;
    openPrompt: (title: string, placeholder: string, onSubmit: (value: string) => void) => void;
    openConfirm: (title: string, message: string, onConfirm: () => void) => void;
}

const ModalContext = createContext<ModalContextType | undefined>(undefined);

export function useModal() {
    const context = useContext(ModalContext);
    if (!context) throw new Error('useModal must be used within ModalProvider');
    return context;
}

export function ModalProvider({ children }: { children: ReactNode }) {
    const [isOpen, setIsOpen] = useState(false);
    const [title, setTitle] = useState('');
    const [modalType, setModalType] = useState<'custom' | 'prompt' | 'confirm'>('custom');
    const [content, setContent] = useState<ReactNode>(null);
    const [promptPlaceholder, setPromptPlaceholder] = useState('');
    const [confirmMessage, setConfirmMessage] = useState('');
    const onSubmitRef = useRef<((val: string) => void) | null>(null);
    const onConfirmRef = useRef<(() => void) | null>(null);

    const openModal = (content: ReactNode, title?: string) => {
        setModalType('custom');
        setContent(content);
        setTitle(title || '');
        setIsOpen(true);
    };

    const closeModal = () => {
        setIsOpen(false);
        setContent(null);
        setTitle('');
        onSubmitRef.current = null;
        onConfirmRef.current = null;
    };

    const openPrompt = (title: string, placeholder: string, onSubmit: (value: string) => void) => {
        setModalType('prompt');
        setPromptPlaceholder(placeholder);
        onSubmitRef.current = onSubmit;
        setTitle(title);
        setIsOpen(true);
    };

    const openConfirm = (title: string, message: string, onConfirm: () => void) => {
        setModalType('confirm');
        setConfirmMessage(message);
        onConfirmRef.current = onConfirm;
        setTitle(title);
        setIsOpen(true);
    };

    // ESC to close
    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') closeModal();
        };
        document.addEventListener('keydown', handleEsc);
        return () => document.removeEventListener('keydown', handleEsc);
    }, []);

    const handlePromptSubmit = (value: string) => {
        if (onSubmitRef.current) {
            onSubmitRef.current(value);
        }
        closeModal();
    };

    const handleConfirm = () => {
        if (onConfirmRef.current) {
            onConfirmRef.current();
        }
        closeModal();
    };

    return (
        <ModalContext.Provider value={{ openModal, closeModal, openPrompt, openConfirm }}>
            {children}
            {isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center">
                    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={closeModal} />
                    <div className="relative bg-[#141414] border border-[#2a2a2a] rounded-xl shadow-2xl w-full max-w-md overflow-hidden">
                        {title && (
                            <div className="flex items-center justify-between px-5 py-4 border-b border-[#2a2a2a]">
                                <h3 className="text-base font-medium text-white">{title}</h3>
                                <button onClick={closeModal} className="text-[#666] hover:text-white">
                                    <X className="w-5 h-5" />
                                </button>
                            </div>
                        )}
                        <div className="p-5">
                            {modalType === 'prompt' && (
                                <PromptContent
                                    placeholder={promptPlaceholder}
                                    onSubmit={handlePromptSubmit}
                                    onCancel={closeModal}
                                />
                            )}
                            {modalType === 'confirm' && (
                                <ConfirmContent
                                    message={confirmMessage}
                                    onConfirm={handleConfirm}
                                    onCancel={closeModal}
                                />
                            )}
                            {modalType === 'custom' && content}
                        </div>
                    </div>
                </div>
            )}
        </ModalContext.Provider>
    );
}

// Prompt content component
function PromptContent({ placeholder, onSubmit, onCancel }: { placeholder: string; onSubmit: (val: string) => void; onCancel: () => void }) {
    const [value, setValue] = useState('');
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        // Focus input on mount
        setTimeout(() => inputRef.current?.focus(), 100);
    }, []);

    return (
        <div>
            <input
                ref={inputRef}
                type="text"
                value={value}
                onChange={(e) => setValue(e.target.value)}
                placeholder={placeholder}
                className="w-full px-4 py-2.5 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-white placeholder-[#666] focus:outline-none focus:border-[#10a37f] text-sm"
                onKeyDown={(e) => { if (e.key === 'Enter' && value.trim()) onSubmit(value); }}
            />
            <div className="flex gap-3 mt-4">
                <button onClick={onCancel} className="flex-1 py-2 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a] text-sm text-[#a1a1a1] hover:text-white transition-colors">
                    Cancel
                </button>
                <button
                    onClick={() => value.trim() && onSubmit(value)}
                    className="flex-1 py-2 rounded-lg bg-[#10a37f] text-white text-sm font-medium hover:bg-[#0d8a6a] transition-colors"
                >
                    Create
                </button>
            </div>
        </div>
    );
}

// Confirm content component
function ConfirmContent({ message, onConfirm, onCancel }: { message: string; onConfirm: () => void; onCancel: () => void }) {
    return (
        <div>
            <p className="text-sm text-[#a1a1a1] mb-4">{message}</p>
            <div className="flex gap-3">
                <button onClick={onCancel} className="flex-1 py-2 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a] text-sm text-[#a1a1a1] hover:text-white transition-colors">
                    Cancel
                </button>
                <button onClick={onConfirm} className="flex-1 py-2 rounded-lg bg-[#10a37f] text-white text-sm font-medium hover:bg-[#0d8a6a] transition-colors">
                    Confirm
                </button>
            </div>
        </div>
    );
}
