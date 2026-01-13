'use client';

import { useState, useEffect } from 'react';

interface FeatureImportance {
    feature: string;
    importance: number;
    direction: 'positive' | 'negative';
}

interface Explanation {
    prediction: string;
    confidence: number;
    features: FeatureImportance[];
    text_explanation: string;
}

export default function XAIPage() {
    const [status, setStatus] = useState<any>(null);
    const [explanation, setExplanation] = useState<Explanation | null>(null);
    const [loading, setLoading] = useState(false);
    const [featureValues, setFeatureValues] = useState<number[]>(Array(10).fill(0));
    const [globalImportance, setGlobalImportance] = useState<FeatureImportance[]>([]);

    const API_BASE = 'http://localhost:8000/api/v2';

    const featureNames = [
        'Flow Duration', 'Total Fwd Packets', 'Total Bwd Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Fwd IAT Mean',
        'Bwd IAT Mean', 'Packet Length Mean', 'FIN Flag Count', 'SYN Flag Count'
    ];

    useEffect(() => {
        fetchStatus();
        fetchGlobalImportance();
    }, []);

    const fetchStatus = async () => {
        try {
            const res = await fetch(`${API_BASE}/xai/status`);
            const data = await res.json();
            setStatus(data);
        } catch (error) {
            console.error('Failed to fetch XAI status:', error);
        }
    };

    const fetchGlobalImportance = async () => {
        try {
            const res = await fetch(`${API_BASE}/xai/feature-importance`);
            if (res.ok) {
                const data = await res.json();
                if (data.importance) {
                    const importance = Object.entries(data.importance).map(([feature, value]) => ({
                        feature,
                        importance: Math.abs(value as number),
                        direction: ((value as number) >= 0 ? 'positive' : 'negative') as const
                    })).sort((a, b) => b.importance - a.importance).slice(0, 15);
                    setGlobalImportance(importance);
                }
            }
        } catch (error) {
            console.error('Failed to fetch global importance:', error);
        }
    };

    const explainPrediction = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/xai/explain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: featureValues, method: 'shap' })
            });
            const data = await res.json();

            if (data.explanation) {
                setExplanation({
                    prediction: data.prediction || 'Unknown',
                    confidence: data.confidence || 0,
                    features: data.explanation.top_features?.map((f: any) => ({
                        feature: f.feature,
                        importance: Math.abs(f.contribution),
                        direction: f.contribution >= 0 ? 'positive' : 'negative'
                    })) || [],
                    text_explanation: data.explanation.text_explanation || ''
                });
            }
        } catch (error) {
            console.error('Failed to get explanation:', error);
        }
        setLoading(false);
    };

    const generateRandomSample = () => {
        setFeatureValues(featureNames.map(() => Math.random() * 1000));
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white flex items-center gap-3">
                    🧠 Explainable AI Dashboard
                </h1>
                <p className="text-[#666] text-sm mt-1">
                    Understand why ML models make their predictions using SHAP and LIME
                </p>
            </div>

            {/* Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">SHAP Available</div>
                    <div className={`text-2xl font-bold ${status?.shap_available ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                        {status?.shap_available ? '✓ Yes' : '✗ No'}
                    </div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">LIME Available</div>
                    <div className={`text-2xl font-bold ${status?.lime_available ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                        {status?.lime_available ? '✓ Yes' : '✗ No'}
                    </div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Model Loaded</div>
                    <div className={`text-2xl font-bold ${status?.model_loaded ? 'text-[#22c55e]' : 'text-[#eab308]'}`}>
                        {status?.model_loaded ? '✓ Yes' : '⏳ Pending'}
                    </div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Explainers Ready</div>
                    <div className={`text-2xl font-bold ${status?.shap_explainer_ready && status?.lime_explainer_ready ? 'text-[#22c55e]' : 'text-[#eab308]'}`}>
                        {status?.shap_explainer_ready && status?.lime_explainer_ready ? '✓ Yes' : '⏳ Init'}
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Input Panel */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h2 className="text-lg font-medium text-white mb-4">📊 Input Features</h2>
                    <div className="space-y-3 mb-6">
                        {featureNames.map((name, idx) => (
                            <div key={idx} className="flex items-center gap-3">
                                <label className="text-[#888] text-sm w-40 truncate">{name}</label>
                                <input
                                    type="number"
                                    value={featureValues[idx].toFixed(2)}
                                    onChange={(e) => {
                                        const newValues = [...featureValues];
                                        newValues[idx] = parseFloat(e.target.value) || 0;
                                        setFeatureValues(newValues);
                                    }}
                                    className="flex-1 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-white text-sm focus:border-[#10a37f] outline-none"
                                />
                            </div>
                        ))}
                    </div>
                    <div className="flex gap-3">
                        <button
                            onClick={generateRandomSample}
                            className="flex-1 bg-[#1a1a1a] hover:bg-[#252525] text-white py-2 px-4 rounded-lg transition border border-[#2a2a2a]"
                        >
                            🎲 Random Sample
                        </button>
                        <button
                            onClick={explainPrediction}
                            disabled={loading}
                            className="flex-1 bg-[#10a37f] hover:bg-[#0d8c6d] text-white py-2 px-4 rounded-lg transition disabled:opacity-50"
                        >
                            {loading ? '⏳ Analyzing...' : '🔍 Explain Prediction'}
                        </button>
                    </div>
                </div>

                {/* Explanation Panel */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h2 className="text-lg font-medium text-white mb-4">💡 Explanation</h2>
                    {explanation ? (
                        <div className="space-y-4">
                            <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                <div className="flex justify-between items-center">
                                    <span className="text-[#666]">Prediction</span>
                                    <span className={`text-xl font-bold ${explanation.prediction === 'BENIGN' ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                                        {explanation.prediction}
                                    </span>
                                </div>
                                <div className="mt-2">
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-[#666]">Confidence</span>
                                        <span className="text-white">{(explanation.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="w-full bg-[#1a1a1a] rounded-full h-2">
                                        <div
                                            className="bg-[#10a37f] h-2 rounded-full transition-all"
                                            style={{ width: `${explanation.confidence * 100}%` }}
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-2">
                                <h3 className="text-[#10a37f] font-medium text-sm">Top Contributing Features</h3>
                                {explanation.features.slice(0, 5).map((feat, idx) => (
                                    <div key={idx} className="flex items-center gap-2">
                                        <span className="text-white text-sm w-32 truncate">{feat.feature}</span>
                                        <div className="flex-1 h-4 bg-[#1a1a1a] rounded overflow-hidden">
                                            <div
                                                className={`h-full ${feat.direction === 'positive' ? 'bg-[#22c55e]' : 'bg-[#ef4444]'}`}
                                                style={{ width: `${Math.min(feat.importance * 100, 100)}%` }}
                                            />
                                        </div>
                                        <span className={`text-xs ${feat.direction === 'positive' ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                                            {feat.direction === 'positive' ? '↑' : '↓'}
                                        </span>
                                    </div>
                                ))}
                            </div>

                            {explanation.text_explanation && (
                                <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                    <h3 className="text-[#10a37f] font-medium text-sm mb-2">Natural Language Explanation</h3>
                                    <p className="text-white text-sm leading-relaxed">{explanation.text_explanation}</p>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-center text-[#666] py-12">
                            <div className="text-6xl mb-4">🤔</div>
                            <p>Enter feature values and click "Explain Prediction"</p>
                            <p className="text-sm mt-2">to see why the model made its decision</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Global Feature Importance */}
            <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                <h2 className="text-lg font-medium text-white mb-4">📈 Global Feature Importance</h2>
                {globalImportance.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {globalImportance.map((feat, idx) => (
                            <div key={idx} className="flex items-center gap-3">
                                <span className="text-[#10a37f] text-sm font-mono w-6">{idx + 1}.</span>
                                <span className="text-white text-sm flex-1 truncate">{feat.feature}</span>
                                <div className="w-24 h-3 bg-[#1a1a1a] rounded overflow-hidden">
                                    <div
                                        className="h-full bg-[#10a37f]"
                                        style={{ width: `${Math.min(feat.importance * 100, 100)}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-[#666] text-center py-8">
                        Initialize XAI module with training data to see global feature importance
                    </p>
                )}
            </div>
        </div>
    );
}
