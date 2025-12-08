// PCDS Enterprise API Client v2
// Updated to connect with backend API v2 endpoints

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_V2_PREFIX = '/api/v2';

export class PCDSClient {
    private baseURL: string;
    private ws: WebSocket | null = null;

    constructor(baseURL: string = API_BASE_URL) {
        this.baseURL = baseURL;
    }

    private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
        // Get token from localStorage
        const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;

        const response = await fetch(`${this.baseURL}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...(token && { 'Authorization': `Bearer ${token}` }),
                ...options?.headers,
            },
        });

        if (response.status === 401) {
            // Token expired or invalid, redirect to login
            if (typeof window !== 'undefined') {
                localStorage.removeItem('access_token');
                window.location.href = '/login';
            }
            throw new Error('Authentication required');
        }

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || `API error: ${response.statusText}`);
        }

        return response.json();
    }

    // Generic GET helper for simple API calls
    async get<T = any>(endpoint: string): Promise<T> {
        return this.request<T>(endpoint);
    }

    // ============================================
    // Dashboard API
    // ============================================

    async getDashboardOverview(hours: number = 24) {
        return this.request(`${API_V2_PREFIX}/dashboard/overview?hours=${hours}`);
    }

    // ============================================
    // Entities API
    // ============================================

    async getEntities(params?: {
        limit?: number;
        urgency_level?: 'critical' | 'high' | 'medium' | 'low';
        entity_type?: string;
    }) {
        const query = new URLSearchParams();
        if (params?.limit) query.append('limit', params.limit.toString());
        if (params?.urgency_level) query.append('urgency_level', params.urgency_level);
        if (params?.entity_type) query.append('entity_type', params.entity_type);

        return this.request(`${API_V2_PREFIX}/entities?${query}`);
    }

    async getEntity(entityId: string) {
        return this.request(`${API_V2_PREFIX}/entities/${entityId}`);
    }

    async getEntityTimeline(entityId: string, hours: number = 24) {
        return this.request(`${API_V2_PREFIX}/entities/${entityId}/timeline?hours=${hours}`);
    }

    async getEntityGraph(entityId: string) {
        return this.request(`${API_V2_PREFIX}/entities/${entityId}/graph`);
    }

    async getEntityStats() {
        return this.request(`${API_V2_PREFIX}/entities/stats/overview`);
    }

    async recalculateEntityScore(entityId: string, assetValue: number = 50) {
        return this.request(`${API_V2_PREFIX}/entities/${entityId}/recalculate-score?asset_value=${assetValue}`, {
            method: 'POST',
        });
    }

    // ============================================
    // Detections API
    // ============================================

    async getDetections(params?: {
        limit?: number;
        severity?: 'critical' | 'high' | 'medium' | 'low';
        entity_id?: string;
        technique_id?: string;
        hours?: number;
    }) {
        const query = new URLSearchParams();
        if (params?.limit) query.append('limit', params.limit.toString());
        if (params?.severity) query.append('severity', params.severity);
        if (params?.entity_id) query.append('entity_id', params.entity_id);
        if (params?.technique_id) query.append('technique_id', params.technique_id);
        if (params?.hours) query.append('hours', params.hours.toString());

        return this.request(`${API_V2_PREFIX}/detections?${query}`);
    }

    async getDetection(detectionId: string) {
        return this.request(`${API_V2_PREFIX}/detections/${detectionId}`);
    }

    async createDetection(data: any) {
        return this.request(`${API_V2_PREFIX}/detections`, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    async updateDetectionStatus(detectionId: string, status: string, assigned_to?: string) {
        const query = new URLSearchParams({ status });
        if (assigned_to) query.append('assigned_to', assigned_to);

        return this.request(`${API_V2_PREFIX}/detections/${detectionId}/status?${query}`, {
            method: 'PATCH',
        });
    }

    async getDetectionStats(hours: number = 24) {
        return this.request(`${API_V2_PREFIX}/detections/stats/severity-breakdown?hours=${hours}`);
    }

    async getTechniqueFrequency(limit: number = 10, hours: number = 24) {
        return this.request(`${API_V2_PREFIX}/detections/stats/technique-frequency?limit=${limit}&hours=${hours}`);
    }

    // ============================================
    // Campaigns API
    // ============================================

    async getCampaigns(status?: 'active' | 'contained' | 'resolved', limit: number = 50) {
        const query = new URLSearchParams({ limit: limit.toString() });
        if (status) query.append('status', status);

        return this.request(`${API_V2_PREFIX}/campaigns?${query}`);
    }

    async getCampaign(campaignId: string) {
        return this.request(`${API_V2_PREFIX}/campaigns/${campaignId}`);
    }

    async updateCampaignStatus(campaignId: string, status: 'active' | 'contained' | 'resolved') {
        return this.request(`${API_V2_PREFIX}/campaigns/${campaignId}/status?status=${status}`, {
            method: 'PATCH',
        });
    }

    // ============================================
    // Investigations API
    // ============================================

    async getInvestigations(params?: {
        status?: 'open' | 'investigating' | 'resolved' | 'closed';
        assigned_to?: string;
        limit?: number;
    }) {
        const query = new URLSearchParams();
        if (params?.status) query.append('status', params.status);
        if (params?.assigned_to) query.append('assigned_to', params.assigned_to);
        if (params?.limit) query.append('limit', params.limit.toString());

        return this.request(`${API_V2_PREFIX}/investigations?${query}`);
    }

    async getInvestigation(investigationId: string) {
        return this.request(`${API_V2_PREFIX}/investigations/${investigationId}`);
    }

    async createInvestigation(data: any) {
        return this.request(`${API_V2_PREFIX}/investigations`, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    async addInvestigationNote(investigationId: string, author: string, content: string) {
        return this.request(`${API_V2_PREFIX}/investigations/${investigationId}/notes`, {
            method: 'POST',
            body: JSON.stringify({ author, content }),
        });
    }

    async updateInvestigationStatus(
        investigationId: string,
        status: 'open' | 'investigating' | 'resolved' | 'closed',
        resolution?: 'true_positive' | 'false_positive' | 'benign',
        resolution_notes?: string
    ) {
        const query = new URLSearchParams({ status });
        if (resolution) query.append('resolution', resolution);
        if (resolution_notes) query.append('resolution_notes', resolution_notes);

        return this.request(`${API_V2_PREFIX}/investigations/${investigationId}/status?${query}`, {
            method: 'PATCH',
        });
    }

    // ============================================
    // Threat Hunting API
    // ============================================

    async getHuntQueries(query_type?: 'saved' | 'scheduled' | 'template', is_public?: boolean) {
        const query = new URLSearchParams();
        if (query_type) query.append('query_type', query_type);
        if (is_public !== undefined) query.append('is_public', is_public.toString());

        return this.request(`${API_V2_PREFIX}/hunt/queries?${query}`);
    }

    async getHuntQuery(queryId: string) {
        return this.request(`${API_V2_PREFIX}/hunt/queries/${queryId}`);
    }

    async runHuntQuery(queryId: string) {
        return this.request(`${API_V2_PREFIX}/hunt/queries/${queryId}/run`, {
            method: 'POST',
        });
    }

    // ============================================
    // MITRE ATT&CK API
    // ============================================

    async getMITRETactics() {
        return this.request(`${API_V2_PREFIX}/mitre/tactics`);
    }

    async getTacticTechniques(tacticId: string) {
        return this.request(`${API_V2_PREFIX}/mitre/tactics/${tacticId}/techniques`);
    }

    async getMITRETechnique(techniqueId: string) {
        return this.request(`${API_V2_PREFIX}/mitre/techniques/${techniqueId}`);
    }

    async getMITREHeatmap(hours: number = 24) {
        return this.request(`${API_V2_PREFIX}/mitre/matrix/heatmap?hours=${hours}`);
    }

    async getMITRECoverage() {
        return this.request(`${API_V2_PREFIX}/mitre/stats/coverage`);
    }

    private wsReconnectAttempts = 0;
    private maxReconnectAttempts = 3;

    connectWebSocket(onMessage: (data: any) => void, onError?: (error: Event) => void) {
        if (this.ws) {
            this.ws.close();
        }

        const wsURL = this.baseURL.replace('http://', 'ws://').replace('https://', 'wss://');

        try {
            this.ws = new WebSocket(`${wsURL}/ws`);

            this.ws.onopen = () => {
                console.log('✅ WebSocket connected to PCDS Enterprise');
                this.wsReconnectAttempts = 0; // Reset on successful connection
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    onMessage(data);
                } catch (error) {
                    console.error('WebSocket message parse error:', error);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                if (onError) onError(error);
            };

            this.ws.onclose = () => {
                console.log('❌ WebSocket disconnected');
                // Limited auto-reconnect (max 3 attempts)
                if (this.wsReconnectAttempts < this.maxReconnectAttempts) {
                    this.wsReconnectAttempts++;
                    console.log(`🔄 Attempting WebSocket reconnection (${this.wsReconnectAttempts}/${this.maxReconnectAttempts})...`);
                    setTimeout(() => {
                        this.connectWebSocket(onMessage, onError);
                    }, 5000);
                } else {
                    console.log('⚠️ Max WebSocket reconnection attempts reached. Use manual refresh.');
                }
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
        }

        return this.ws;
    }

    disconnectWebSocket() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    sendWebSocketMessage(message: any) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected');
        }
    }

    // Ping server
    ping() {
        this.sendWebSocketMessage({ action: 'ping' });
    }

    // Subscribe to channels
    subscribe(channels: string[]) {
        this.sendWebSocketMessage({ action: 'subscribe', channels });
    }
}

// Export singleton instance
export const apiClient = new PCDSClient();
export default apiClient;
