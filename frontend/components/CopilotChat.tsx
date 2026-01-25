'use client';

import { useState, useRef, useEffect } from 'react';
import { Bot, Send, User, Loader2, Sparkles } from 'lucide-react';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

interface CopilotChatProps {
    context?: Record<string, any>;
    placeholder?: string;
}

export default function CopilotChat({ context, placeholder = "Ask about threats, investigations, or security..." }: CopilotChatProps) {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '0',
            role: 'assistant',
            content: "⚠️ The Security Co-pilot is currently disabled by the administrator to conserve API credits. Please contact your system admin to enable it.",
            timestamp: new Date()
        }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async () => {
        // Disabled
        return;
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const suggestedQueries = [
        "What's the most critical threat right now?",
        "How should I respond to ransomware?",
        "Explain the latest detection",
        "What MITRE techniques are most common?"
    ];

    return (
        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] flex flex-col h-[500px]">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-[#2a2a2a]">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#10a37f] to-[#0d8a6a] flex items-center justify-center">
                        <Sparkles className="w-4 h-4 text-white" />
                    </div>
                    <div>
                        <h3 className="text-sm font-medium text-white">Security Co-pilot</h3>
                        <p className="text-xs text-[#10a37f]">Powered by Azure OpenAI</p>
                    </div>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
                    >
                        <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${message.role === 'assistant'
                            ? 'bg-[#10a37f]/20'
                            : 'bg-[#3b82f6]/20'
                            }`}>
                            {message.role === 'assistant'
                                ? <Bot className="w-4 h-4 text-[#10a37f]" />
                                : <User className="w-4 h-4 text-[#3b82f6]" />
                            }
                        </div>
                        <div className={`max-w-[80%] rounded-lg p-3 ${message.role === 'assistant'
                            ? 'bg-[#1a1a1a] text-[#e5e5e5]'
                            : 'bg-[#10a37f] text-white'
                            }`}>
                            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                        </div>
                    </div>
                ))}

                {loading && (
                    <div className="flex gap-3">
                        <div className="w-7 h-7 rounded-full bg-[#10a37f]/20 flex items-center justify-center">
                            <Loader2 className="w-4 h-4 text-[#10a37f] animate-spin" />
                        </div>
                        <div className="bg-[#1a1a1a] rounded-lg p-3">
                            <p className="text-sm text-[#888]">Thinking...</p>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Suggested Queries (show only if few messages) */}
            {messages.length <= 2 && (
                <div className="px-4 pb-2">
                    <p className="text-xs text-[#666] mb-2">Try asking:</p>
                    <div className="flex flex-wrap gap-2">
                        {suggestedQueries.map((query, i) => (
                            <button
                                key={i}
                                onClick={() => setInput(query)}
                                className="text-xs px-2 py-1 rounded-full bg-[#1a1a1a] text-[#a1a1a1] hover:bg-[#222] hover:text-white transition-colors"
                            >
                                {query}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* Input */}
            <div className="p-4 border-t border-[#2a2a2a]">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder={"Chat is disabled"}
                        disabled={true}
                        className="flex-1 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-4 py-2.5 text-sm text-white placeholder-[#666] focus:outline-none focus:border-[#10a37f] disabled:opacity-50 disabled:cursor-not-allowed"
                    />
                    <button
                        onClick={sendMessage}
                        disabled={true}
                        className="px-4 py-2.5 bg-[#10a37f] hover:bg-[#0d8a6a] disabled:bg-[#333] disabled:cursor-not-allowed rounded-lg text-white transition-colors"
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </div>
            </div>
        </div>
    );
}
