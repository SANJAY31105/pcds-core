'use client';

import { useState, useEffect, useRef } from 'react';
import { Search, X, Target, AlertTriangle, ClipboardList, ArrowRight } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface SearchResult {
    id: string;
    type: 'entity' | 'detection' | 'investigation';
    title: string;
    subtitle: string;
    severity?: string;
}

export default function GlobalSearch() {
    const [isOpen, setIsOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);
    const router = useRouter();

    // Keyboard shortcut: Ctrl+K or Cmd+K
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                setIsOpen(true);
            }
            if (e.key === 'Escape') {
                setIsOpen(false);
            }
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, []);

    // Focus input when opened
    useEffect(() => {
        if (isOpen && inputRef.current) {
            inputRef.current.focus();
        }
    }, [isOpen]);

    // Search functionality
    useEffect(() => {
        if (!query.trim()) {
            setResults([]);
            return;
        }

        setLoading(true);
        const searchTimeout = setTimeout(async () => {
            try {
                // Search entities
                const entityRes = await fetch(`http://localhost:8000/api/v2/entities?search=${query}&limit=5`);
                const entityData = await entityRes.json();

                // Search detections
                const detectionRes = await fetch(`http://localhost:8000/api/v2/detections?limit=5`);
                const detectionData = await detectionRes.json();

                const searchResults: SearchResult[] = [];

                // Add entities
                (entityData.entities || []).forEach((e: any) => {
                    if (e.identifier?.toLowerCase().includes(query.toLowerCase())) {
                        searchResults.push({
                            id: e.id,
                            type: 'entity',
                            title: e.identifier,
                            subtitle: e.entity_type || e.type || 'Entity',
                            severity: e.urgency_level
                        });
                    }
                });

                // Add detections
                (detectionData.detections || []).forEach((d: any) => {
                    const title = d.detection_type || d.title || 'Detection';
                    if (title.toLowerCase().includes(query.toLowerCase())) {
                        searchResults.push({
                            id: d.id,
                            type: 'detection',
                            title: title,
                            subtitle: d.source_ip || 'Detection',
                            severity: d.severity
                        });
                    }
                });

                setResults(searchResults.slice(0, 8));
            } catch (error) {
                console.error('Search failed:', error);
                // Mock results for demo
                setResults([
                    { id: '1', type: 'entity', title: '192.168.1.100', subtitle: 'IP Address', severity: 'high' },
                    { id: '2', type: 'entity', title: 'workstation-15', subtitle: 'Hostname', severity: 'critical' },
                    { id: '3', type: 'detection', title: 'Phishing Attempt', subtitle: 'Email Security', severity: 'high' }
                ].filter(r => r.title.toLowerCase().includes(query.toLowerCase())));
            } finally {
                setLoading(false);
            }
        }, 300);

        return () => clearTimeout(searchTimeout);
    }, [query]);

    // Navigate with keyboard
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!isOpen) return;
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                setSelectedIndex(i => Math.min(i + 1, results.length - 1));
            }
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                setSelectedIndex(i => Math.max(i - 1, 0));
            }
            if (e.key === 'Enter' && results[selectedIndex]) {
                navigateToResult(results[selectedIndex]);
            }
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, results, selectedIndex]);

    const navigateToResult = (result: SearchResult) => {
        setIsOpen(false);
        setQuery('');
        if (result.type === 'entity') router.push(`/entities/${result.id}`);
        else if (result.type === 'detection') router.push('/detections');
        else if (result.type === 'investigation') router.push('/investigations');
    };

    const getIcon = (type: string) => {
        switch (type) {
            case 'entity': return Target;
            case 'detection': return AlertTriangle;
            case 'investigation': return ClipboardList;
            default: return Search;
        }
    };

    const getSeverityColor = (severity?: string) => {
        const colors: Record<string, string> = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#3b82f6' };
        return severity ? colors[severity] : '#666';
    };

    if (!isOpen) {
        return (
            <button
                onClick={() => setIsOpen(true)}
                className="flex items-center gap-2 px-3 py-1.5 bg-[#141414] border border-[#2a2a2a] rounded-lg text-sm text-[#666] hover:text-[#a1a1a1] hover:border-[#333] transition-colors"
            >
                <Search className="w-4 h-4" />
                <span>Search</span>
                <kbd className="ml-2 px-1.5 py-0.5 bg-[#0a0a0a] rounded text-xs">⌘K</kbd>
            </button>
        );
    }

    return (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setIsOpen(false)} />

            {/* Search Modal */}
            <div className="relative w-full max-w-xl bg-[#141414] border border-[#2a2a2a] rounded-xl shadow-2xl overflow-hidden">
                {/* Input */}
                <div className="flex items-center gap-3 px-4 py-3 border-b border-[#2a2a2a]">
                    <Search className="w-5 h-5 text-[#666]" />
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={(e) => { setQuery(e.target.value); setSelectedIndex(0); }}
                        placeholder="Search entities, detections, investigations..."
                        className="flex-1 bg-transparent text-white placeholder-[#666] outline-none text-sm"
                    />
                    {query && (
                        <button onClick={() => setQuery('')} className="text-[#666] hover:text-white">
                            <X className="w-4 h-4" />
                        </button>
                    )}
                    <kbd className="px-1.5 py-0.5 bg-[#0a0a0a] rounded text-xs text-[#666]">ESC</kbd>
                </div>

                {/* Results */}
                {query && (
                    <div className="max-h-80 overflow-y-auto">
                        {loading ? (
                            <div className="p-4 text-center text-[#666] text-sm">Searching...</div>
                        ) : results.length === 0 ? (
                            <div className="p-4 text-center text-[#666] text-sm">No results found</div>
                        ) : (
                            <div className="py-2">
                                {results.map((result, i) => {
                                    const Icon = getIcon(result.type);
                                    return (
                                        <button
                                            key={`${result.type}-${result.id}`}
                                            onClick={() => navigateToResult(result)}
                                            className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${i === selectedIndex ? 'bg-[#1a1a1a]' : 'hover:bg-[#1a1a1a]'
                                                }`}
                                        >
                                            <Icon className="w-4 h-4" style={{ color: getSeverityColor(result.severity) }} />
                                            <div className="flex-1 min-w-0">
                                                <p className="text-sm text-white truncate">{result.title}</p>
                                                <p className="text-xs text-[#666]">{result.subtitle}</p>
                                            </div>
                                            {result.severity && (
                                                <span className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: `${getSeverityColor(result.severity)}20`, color: getSeverityColor(result.severity) }}>
                                                    {result.severity}
                                                </span>
                                            )}
                                            <ArrowRight className="w-4 h-4 text-[#444]" />
                                        </button>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}

                {/* Footer */}
                <div className="px-4 py-2 border-t border-[#2a2a2a] flex items-center justify-between text-xs text-[#666]">
                    <div className="flex items-center gap-4">
                        <span>↑↓ Navigate</span>
                        <span>↵ Open</span>
                    </div>
                    <span>ESC to close</span>
                </div>
            </div>
        </div>
    );
}
