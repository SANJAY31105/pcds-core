'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { WebSocketMessage } from '@/types';

const getWsUrl = () => {
    if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
    const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    return apiBase.replace(/^http/, 'ws') + '/ws';
};

const WS_URL = getWsUrl();

export function useWebSocket(onMessage?: (message: WebSocketMessage) => void) {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const connect = useCallback(() => {
        try {
            const ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                console.log('✅ WebSocket connected');
                setIsConnected(true);
            };

            ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);
                    setLastMessage(message);
                    onMessage?.(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onclose = () => {
                console.log('❌ WebSocket disconnected');
                setIsConnected(false);

                // Auto-reconnect after 3 seconds
                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log('🔄 Attempting to reconnect...');
                    connect();
                }, 3000);
            };

            wsRef.current = ws;
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }, [onMessage]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
    }, []);

    const sendMessage = useCallback((message: any) => {
        if (wsRef.current && isConnected) {
            wsRef.current.send(JSON.stringify(message));
        }
    }, [isConnected]);

    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    return {
        isConnected,
        lastMessage,
        sendMessage,
        disconnect,
        reconnect: connect,
    };
}
