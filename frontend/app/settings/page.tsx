'use client';

import { useState } from 'react';
import { Settings, Shield, Bell, Database, Cpu, Globe, Mail, Save, RotateCcw } from 'lucide-react';

interface SettingsSection {
    id: string;
    title: string;
    icon: any;
}

export default function SettingsPage() {
    const [activeSection, setActiveSection] = useState('general');
    const [saved, setSaved] = useState(false);

    // Settings state
    const [settings, setSettings] = useState({
        // General
        companyName: 'PCDS Enterprise',
        timezone: 'Asia/Kolkata',
        dateFormat: 'DD/MM/YYYY',

        // Detection
        detectionThreshold: 0.7,
        autoIsolate: true,
        autoIsolateThreshold: 0.9,
        realTimeMonitoring: true,

        // Notifications
        emailAlerts: true,
        slackIntegration: false,
        alertOnCritical: true,
        alertOnHigh: true,
        alertOnMedium: false,

        // ML Engine
        modelVersion: 'v2.1.0-lstm',
        autoRetrain: false,
        retrainInterval: 7,

        // API
        apiRateLimit: 1000,
        enableCORS: true,
        apiKey: 'pcds_live_*****************************'
    });

    const sections: SettingsSection[] = [
        { id: 'general', title: 'General', icon: Settings },
        { id: 'detection', title: 'Detection', icon: Shield },
        { id: 'notifications', title: 'Notifications', icon: Bell },
        { id: 'ml', title: 'ML Engine', icon: Cpu },
        { id: 'api', title: 'API', icon: Globe },
    ];

    const handleSave = () => {
        // In production, this would save to backend
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
    };

    const handleReset = () => {
        // Reset to defaults
        setSettings({
            ...settings,
            detectionThreshold: 0.7,
            autoIsolate: true,
            autoIsolateThreshold: 0.9,
        });
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-semibold text-white">Settings</h1>
                    <p className="text-[#666] mt-1">Configure PCDS Enterprise</p>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={handleReset}
                        className="flex items-center gap-2 px-4 py-2.5 bg-[#141414] border border-[#2a2a2a] text-[#a1a1a1] rounded-lg hover:text-white transition-colors"
                    >
                        <RotateCcw className="w-4 h-4" />
                        Reset
                    </button>
                    <button
                        onClick={handleSave}
                        className="flex items-center gap-2 px-5 py-2.5 bg-[#10a37f] text-white rounded-lg hover:bg-[#0d8a6a] transition-colors"
                    >
                        <Save className="w-4 h-4" />
                        {saved ? 'Saved!' : 'Save Changes'}
                    </button>
                </div>
            </div>

            <div className="flex gap-6">
                {/* Sidebar */}
                <div className="w-64 space-y-1">
                    {sections.map((section) => (
                        <button
                            key={section.id}
                            onClick={() => setActiveSection(section.id)}
                            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${activeSection === section.id
                                    ? 'bg-[#10a37f]/20 text-[#10a37f]'
                                    : 'text-[#a1a1a1] hover:bg-[#141414] hover:text-white'
                                }`}
                        >
                            <section.icon className="w-5 h-5" />
                            {section.title}
                        </button>
                    ))}
                </div>

                {/* Content */}
                <div className="flex-1 bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                    {/* General Settings */}
                    {activeSection === 'general' && (
                        <div className="space-y-6">
                            <h2 className="text-xl font-semibold text-white mb-4">General Settings</h2>

                            <div>
                                <label className="block text-sm text-[#666] mb-2">Company Name</label>
                                <input
                                    type="text"
                                    value={settings.companyName}
                                    onChange={(e) => setSettings({ ...settings, companyName: e.target.value })}
                                    className="w-full px-4 py-3 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f]"
                                />
                            </div>

                            <div>
                                <label className="block text-sm text-[#666] mb-2">Timezone</label>
                                <select
                                    value={settings.timezone}
                                    onChange={(e) => setSettings({ ...settings, timezone: e.target.value })}
                                    className="w-full px-4 py-3 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f]"
                                >
                                    <option value="Asia/Kolkata">Asia/Kolkata (IST)</option>
                                    <option value="UTC">UTC</option>
                                    <option value="America/New_York">America/New_York (EST)</option>
                                    <option value="Europe/London">Europe/London (GMT)</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm text-[#666] mb-2">Date Format</label>
                                <select
                                    value={settings.dateFormat}
                                    onChange={(e) => setSettings({ ...settings, dateFormat: e.target.value })}
                                    className="w-full px-4 py-3 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f]"
                                >
                                    <option value="DD/MM/YYYY">DD/MM/YYYY</option>
                                    <option value="MM/DD/YYYY">MM/DD/YYYY</option>
                                    <option value="YYYY-MM-DD">YYYY-MM-DD</option>
                                </select>
                            </div>
                        </div>
                    )}

                    {/* Detection Settings */}
                    {activeSection === 'detection' && (
                        <div className="space-y-6">
                            <h2 className="text-xl font-semibold text-white mb-4">Detection Settings</h2>

                            <div>
                                <label className="block text-sm text-[#666] mb-2">
                                    Detection Threshold: {(settings.detectionThreshold * 100).toFixed(0)}%
                                </label>
                                <input
                                    type="range"
                                    min="0.5"
                                    max="0.95"
                                    step="0.05"
                                    value={settings.detectionThreshold}
                                    onChange={(e) => setSettings({ ...settings, detectionThreshold: parseFloat(e.target.value) })}
                                    className="w-full h-2 bg-[#2a2a2a] rounded-lg appearance-none cursor-pointer accent-[#10a37f]"
                                />
                                <p className="text-xs text-[#666] mt-1">Minimum confidence score to trigger a detection</p>
                            </div>

                            <div className="flex items-center justify-between py-3 border-b border-[#2a2a2a]">
                                <div>
                                    <p className="text-white">Real-time Monitoring</p>
                                    <p className="text-sm text-[#666]">Enable live network monitoring</p>
                                </div>
                                <button
                                    onClick={() => setSettings({ ...settings, realTimeMonitoring: !settings.realTimeMonitoring })}
                                    className={`w-12 h-6 rounded-full transition-colors ${settings.realTimeMonitoring ? 'bg-[#10a37f]' : 'bg-[#2a2a2a]'}`}
                                >
                                    <div className={`w-5 h-5 bg-white rounded-full transition-transform ${settings.realTimeMonitoring ? 'translate-x-6' : 'translate-x-0.5'}`} />
                                </button>
                            </div>

                            <div className="flex items-center justify-between py-3 border-b border-[#2a2a2a]">
                                <div>
                                    <p className="text-white">Auto-Isolate Threats</p>
                                    <p className="text-sm text-[#666]">Automatically isolate hosts on critical detection</p>
                                </div>
                                <button
                                    onClick={() => setSettings({ ...settings, autoIsolate: !settings.autoIsolate })}
                                    className={`w-12 h-6 rounded-full transition-colors ${settings.autoIsolate ? 'bg-[#10a37f]' : 'bg-[#2a2a2a]'}`}
                                >
                                    <div className={`w-5 h-5 bg-white rounded-full transition-transform ${settings.autoIsolate ? 'translate-x-6' : 'translate-x-0.5'}`} />
                                </button>
                            </div>

                            {settings.autoIsolate && (
                                <div>
                                    <label className="block text-sm text-[#666] mb-2">
                                        Auto-Isolate Threshold: {(settings.autoIsolateThreshold * 100).toFixed(0)}%
                                    </label>
                                    <input
                                        type="range"
                                        min="0.8"
                                        max="0.99"
                                        step="0.01"
                                        value={settings.autoIsolateThreshold}
                                        onChange={(e) => setSettings({ ...settings, autoIsolateThreshold: parseFloat(e.target.value) })}
                                        className="w-full h-2 bg-[#2a2a2a] rounded-lg appearance-none cursor-pointer accent-[#10a37f]"
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {/* Notifications Settings */}
                    {activeSection === 'notifications' && (
                        <div className="space-y-6">
                            <h2 className="text-xl font-semibold text-white mb-4">Notification Settings</h2>

                            <div className="flex items-center justify-between py-3 border-b border-[#2a2a2a]">
                                <div>
                                    <p className="text-white">Email Alerts</p>
                                    <p className="text-sm text-[#666]">Receive alerts via email</p>
                                </div>
                                <button
                                    onClick={() => setSettings({ ...settings, emailAlerts: !settings.emailAlerts })}
                                    className={`w-12 h-6 rounded-full transition-colors ${settings.emailAlerts ? 'bg-[#10a37f]' : 'bg-[#2a2a2a]'}`}
                                >
                                    <div className={`w-5 h-5 bg-white rounded-full transition-transform ${settings.emailAlerts ? 'translate-x-6' : 'translate-x-0.5'}`} />
                                </button>
                            </div>

                            <div className="flex items-center justify-between py-3 border-b border-[#2a2a2a]">
                                <div>
                                    <p className="text-white">Slack Integration</p>
                                    <p className="text-sm text-[#666]">Post alerts to Slack channel</p>
                                </div>
                                <button
                                    onClick={() => setSettings({ ...settings, slackIntegration: !settings.slackIntegration })}
                                    className={`w-12 h-6 rounded-full transition-colors ${settings.slackIntegration ? 'bg-[#10a37f]' : 'bg-[#2a2a2a]'}`}
                                >
                                    <div className={`w-5 h-5 bg-white rounded-full transition-transform ${settings.slackIntegration ? 'translate-x-6' : 'translate-x-0.5'}`} />
                                </button>
                            </div>

                            <div className="space-y-3">
                                <p className="text-white">Alert on Severity</p>
                                {(['Critical', 'High', 'Medium'] as const).map((level) => {
                                    const key = `alertOn${level}` as keyof typeof settings;
                                    return (
                                        <label key={level} className="flex items-center gap-3 cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={settings[key] as boolean}
                                                onChange={() => setSettings({ ...settings, [key]: !settings[key] })}
                                                className="w-4 h-4 rounded border-[#2a2a2a] bg-[#0a0a0a] text-[#10a37f] focus:ring-[#10a37f]"
                                            />
                                            <span className="text-[#a1a1a1]">{level}</span>
                                        </label>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* ML Engine Settings */}
                    {activeSection === 'ml' && (
                        <div className="space-y-6">
                            <h2 className="text-xl font-semibold text-white mb-4">ML Engine Settings</h2>

                            <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-[#666]">Current Model</span>
                                    <span className="text-[#10a37f] font-mono">{settings.modelVersion}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-[#666]">Status</span>
                                    <span className="text-green-400">Active</span>
                                </div>
                            </div>

                            <div className="flex items-center justify-between py-3 border-b border-[#2a2a2a]">
                                <div>
                                    <p className="text-white">Auto Retrain</p>
                                    <p className="text-sm text-[#666]">Automatically retrain model on new data</p>
                                </div>
                                <button
                                    onClick={() => setSettings({ ...settings, autoRetrain: !settings.autoRetrain })}
                                    className={`w-12 h-6 rounded-full transition-colors ${settings.autoRetrain ? 'bg-[#10a37f]' : 'bg-[#2a2a2a]'}`}
                                >
                                    <div className={`w-5 h-5 bg-white rounded-full transition-transform ${settings.autoRetrain ? 'translate-x-6' : 'translate-x-0.5'}`} />
                                </button>
                            </div>

                            {settings.autoRetrain && (
                                <div>
                                    <label className="block text-sm text-[#666] mb-2">Retrain Interval (days)</label>
                                    <input
                                        type="number"
                                        min="1"
                                        max="30"
                                        value={settings.retrainInterval}
                                        onChange={(e) => setSettings({ ...settings, retrainInterval: parseInt(e.target.value) })}
                                        className="w-full px-4 py-3 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f]"
                                    />
                                </div>
                            )}

                            <button className="w-full py-3 bg-[#2a2a2a] text-white rounded-lg hover:bg-[#3a3a3a] transition-colors">
                                Trigger Manual Retrain
                            </button>
                        </div>
                    )}

                    {/* API Settings */}
                    {activeSection === 'api' && (
                        <div className="space-y-6">
                            <h2 className="text-xl font-semibold text-white mb-4">API Settings</h2>

                            <div>
                                <label className="block text-sm text-[#666] mb-2">API Key</label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={settings.apiKey}
                                        readOnly
                                        className="flex-1 px-4 py-3 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-[#666] font-mono text-sm"
                                    />
                                    <button className="px-4 py-3 bg-[#2a2a2a] text-white rounded-lg hover:bg-[#3a3a3a] transition-colors">
                                        Regenerate
                                    </button>
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm text-[#666] mb-2">Rate Limit (requests/min)</label>
                                <input
                                    type="number"
                                    value={settings.apiRateLimit}
                                    onChange={(e) => setSettings({ ...settings, apiRateLimit: parseInt(e.target.value) })}
                                    className="w-full px-4 py-3 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg text-white focus:outline-none focus:border-[#10a37f]"
                                />
                            </div>

                            <div className="flex items-center justify-between py-3 border-b border-[#2a2a2a]">
                                <div>
                                    <p className="text-white">Enable CORS</p>
                                    <p className="text-sm text-[#666]">Allow cross-origin requests</p>
                                </div>
                                <button
                                    onClick={() => setSettings({ ...settings, enableCORS: !settings.enableCORS })}
                                    className={`w-12 h-6 rounded-full transition-colors ${settings.enableCORS ? 'bg-[#10a37f]' : 'bg-[#2a2a2a]'}`}
                                >
                                    <div className={`w-5 h-5 bg-white rounded-full transition-transform ${settings.enableCORS ? 'translate-x-6' : 'translate-x-0.5'}`} />
                                </button>
                            </div>

                            <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                <p className="text-sm text-[#666] mb-2">API Endpoints</p>
                                <code className="text-xs text-[#10a37f]">http://localhost:8000/api/v2</code>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
