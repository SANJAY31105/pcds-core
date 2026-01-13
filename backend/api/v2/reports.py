"""
Threat Reporting Engine
Generate PDF and HTML threat reports

Report Types:
- Executive Summary
- Weekly/Monthly Threat Report
- Compliance Report (SOC2, HIPAA, PCI-DSS)
- Incident Report
- Risk Assessment
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import io
import base64


router = APIRouter(tags=["Threat Reports"])


# Report storage
generated_reports: Dict[str, Dict] = {}


class ReportRequest(BaseModel):
    """Report generation request"""
    report_type: str  # executive, weekly, monthly, compliance, incident
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    compliance_framework: Optional[str] = None  # soc2, hipaa, pci
    include_charts: bool = True
    include_metrics: bool = True


class IncidentReportRequest(BaseModel):
    """Incident-specific report request"""
    incident_id: str
    title: str
    severity: str
    summary: str
    timeline: List[Dict]
    affected_systems: List[str]
    mitre_techniques: List[str]
    remediation_steps: List[str]
    analyst: str


# Sample data generators for demo
def get_threat_stats(days: int = 7) -> Dict:
    """Get threat statistics for report"""
    import random
    return {
        "total_alerts": random.randint(500, 2000),
        "critical": random.randint(5, 20),
        "high": random.randint(20, 100),
        "medium": random.randint(100, 500),
        "low": random.randint(200, 1000),
        "blocked": random.randint(400, 1500),
        "investigated": random.randint(50, 200),
        "false_positives": random.randint(10, 50),
        "mean_time_to_detect": f"{random.randint(1, 10)} minutes",
        "mean_time_to_respond": f"{random.randint(5, 30)} minutes"
    }


def get_top_threats(limit: int = 10) -> List[Dict]:
    """Get top threat categories"""
    threats = [
        {"name": "DoS/DDoS", "count": 342, "trend": "up"},
        {"name": "Brute Force", "count": 218, "trend": "down"},
        {"name": "Port Scan", "count": 156, "trend": "stable"},
        {"name": "Web Attack", "count": 89, "trend": "up"},
        {"name": "Malware", "count": 67, "trend": "down"},
        {"name": "Credential Theft", "count": 45, "trend": "up"},
        {"name": "Lateral Movement", "count": 34, "trend": "stable"},
        {"name": "Data Exfiltration", "count": 23, "trend": "down"},
        {"name": "Ransomware", "count": 12, "trend": "stable"},
        {"name": "C2 Communication", "count": 8, "trend": "down"},
    ]
    return threats[:limit]


def get_mitre_coverage() -> Dict:
    """Get MITRE ATT&CK coverage stats"""
    return {
        "total_techniques": 201,
        "detected_techniques": 156,
        "coverage_percent": 77.6,
        "top_tactics": [
            {"name": "Initial Access", "detections": 45},
            {"name": "Execution", "detections": 89},
            {"name": "Persistence", "detections": 67},
            {"name": "Privilege Escalation", "detections": 34},
            {"name": "Defense Evasion", "detections": 78},
        ]
    }


def generate_executive_html(stats: Dict, threats: List, mitre: Dict) -> str:
    """Generate stunning executive summary HTML with premium design"""
    
    # Calculate risk level
    risk_level = "HIGH" if stats['critical'] > 15 else "MODERATE" if stats['critical'] > 5 else "LOW"
    risk_color = "#ef4444" if risk_level == "HIGH" else "#f59e0b" if risk_level == "MODERATE" else "#10b981"
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCDS Executive Threat Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f23 100%);
            color: #e5e5e5;
            min-height: 100vh;
            padding: 40px;
            line-height: 1.6;
        }}
        
        .report-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        /* Header */
        .header {{
            text-align: center;
            padding: 32px 40px;
            background: linear-gradient(135deg, rgba(16, 163, 127, 0.12) 0%, rgba(16, 163, 127, 0.03) 100%);
            border-radius: 16px;
            border: 1px solid rgba(16, 163, 127, 0.2);
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(16, 163, 127, 0.1) 0%, transparent 50%);
            animation: pulse 4s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.5; }}
            50% {{ transform: scale(1.1); opacity: 1; }}
        }}
        
        .logo {{
            font-size: 32px;
            margin-bottom: 8px;
        }}
        
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #10a37f 0%, #34d399 50%, #10a37f 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            position: relative;
            z-index: 1;
        }}
        
        .header .subtitle {{
            color: #888;
            font-size: 16px;
            position: relative;
            z-index: 1;
        }}
        
        .header .date {{
            margin-top: 8px;
            color: #10a37f;
            font-weight: 600;
            position: relative;
            z-index: 1;
        }}
        
        /* Sections */
        .section {{
            background: rgba(20, 20, 20, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 32px;
            margin-bottom: 24px;
        }}
        
        .section-title {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 20px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .section-icon {{
            font-size: 24px;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, rgba(30, 30, 40, 0.9) 0%, rgba(20, 20, 30, 0.9) 100%);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s, border-color 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(16, 163, 127, 0.5);
        }}
        
        .stat-value {{
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #10a37f, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .stat-value.critical {{ background: linear-gradient(135deg, #ef4444, #f87171); -webkit-background-clip: text; }}
        .stat-value.high {{ background: linear-gradient(135deg, #f59e0b, #fbbf24); -webkit-background-clip: text; }}
        .stat-value.blocked {{ background: linear-gradient(135deg, #10b981, #34d399); -webkit-background-clip: text; }}
        
        .stat-label {{
            color: #888;
            font-size: 14px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        /* Risk Gauge */
        .risk-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            align-items: center;
        }}
        
        .risk-gauge {{
            text-align: center;
            padding: 40px;
            background: radial-gradient(circle at center, rgba(245, 158, 11, 0.1) 0%, transparent 70%);
            border-radius: 20px;
        }}
        
        .risk-label {{
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }}
        
        .risk-value {{
            font-size: 64px;
            font-weight: 800;
            color: {risk_color};
            text-shadow: 0 0 60px {risk_color}40;
        }}
        
        .risk-description {{
            margin-top: 12px;
            color: #888;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}
        
        .metric-item {{
            background: rgba(30, 30, 40, 0.6);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .metric-label {{
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}
        
        .metric-value {{
            font-size: 24px;
            font-weight: 700;
            color: #10a37f;
        }}
        
        /* Table */
        .threat-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .threat-table th {{
            text-align: left;
            padding: 16px;
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .threat-table td {{
            padding: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .threat-table tr:hover {{
            background: rgba(16, 163, 127, 0.05);
        }}
        
        .threat-name {{
            font-weight: 600;
            color: #fff;
        }}
        
        .threat-count {{
            font-weight: 700;
            color: #10a37f;
        }}
        
        .trend {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .trend-up {{
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
        }}
        
        .trend-down {{
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
        }}
        
        .trend-stable {{
            background: rgba(107, 114, 128, 0.15);
            color: #9ca3af;
        }}
        
        /* Progress Bar */
        .progress-section {{
            margin-top: 24px;
        }}
        
        .progress-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
        }}
        
        .progress-bar {{
            height: 12px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10a37f, #34d399);
            border-radius: 6px;
            width: {mitre['coverage_percent']}%;
        }}
        
        /* Recommendations */
        .recommendation-list {{
            list-style: none;
        }}
        
        .recommendation-item {{
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 16px;
            background: rgba(30, 30, 40, 0.5);
            border-radius: 12px;
            margin-bottom: 12px;
            border-left: 3px solid #10a37f;
        }}
        
        .recommendation-icon {{
            font-size: 20px;
        }}
        
        .recommendation-text {{
            color: #e5e5e5;
        }}
        
        .recommendation-priority {{
            margin-left: auto;
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
        }}
        
        .priority-critical {{
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }}
        
        .priority-high {{
            background: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 14px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 40px;
        }}
        
        .footer-logo {{
            color: #10a37f;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        
        /* Print Styles */
        @media print {{
            body {{
                background: #0a0a0a;
                padding: 20px;
            }}
            .section {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <!-- Header -->
        <div class="header">
            <div class="logo">🛡️</div>
            <h1>PCDS Executive Threat Report</h1>
            <p class="subtitle">Comprehensive Security Intelligence Analysis</p>
            <p class="date">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')} • Reporting Period: Last 7 Days</p>
        </div>
        
        <!-- Threat Overview -->
        <div class="section">
            <div class="section-title">
                <span class="section-icon">📊</span>
                Threat Overview
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats['total_alerts']:,}</div>
                    <div class="stat-label">Total Alerts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value critical">{stats['critical']}</div>
                    <div class="stat-label">Critical</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value high">{stats['high']}</div>
                    <div class="stat-label">High</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value blocked">{stats['blocked']:,}</div>
                    <div class="stat-label">Blocked</div>
                </div>
            </div>
        </div>
        
        <!-- Risk Assessment -->
        <div class="section">
            <div class="section-title">
                <span class="section-icon">⚡</span>
                Risk Assessment & Response Metrics
            </div>
            <div class="risk-section">
                <div class="risk-gauge">
                    <div class="risk-label">Current Risk Level</div>
                    <div class="risk-value">{risk_level}</div>
                    <p class="risk-description">Overall security posture assessment</p>
                </div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Mean Time to Detect</div>
                        <div class="metric-value">{stats['mean_time_to_detect']}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Mean Time to Respond</div>
                        <div class="metric-value">{stats['mean_time_to_respond']}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Investigated Incidents</div>
                        <div class="metric-value">{stats['investigated']}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">False Positive Rate</div>
                        <div class="metric-value">{round(stats['false_positives']/stats['total_alerts']*100, 1)}%</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Top Threats -->
        <div class="section">
            <div class="section-title">
                <span class="section-icon">🎯</span>
                Top Threat Categories
            </div>
            <table class="threat-table">
                <thead>
                    <tr>
                        <th>Threat Type</th>
                        <th>Detections</th>
                        <th>Trend</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td class="threat-name">{t['name']}</td>
                        <td class="threat-count">{t['count']}</td>
                        <td><span class="trend trend-{t['trend']}">{'↑' if t['trend'] == 'up' else '↓' if t['trend'] == 'down' else '→'} {t['trend'].capitalize()}</span></td>
                    </tr>
                    ''' for t in threats[:6])}
                </tbody>
            </table>
        </div>
        
        <!-- MITRE Coverage -->
        <div class="section">
            <div class="section-title">
                <span class="section-icon">🗺️</span>
                MITRE ATT&CK Framework Coverage
            </div>
            <div class="stats-grid" style="grid-template-columns: repeat(3, 1fr);">
                <div class="stat-card">
                    <div class="stat-value">{mitre['coverage_percent']}%</div>
                    <div class="stat-label">Coverage Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{mitre['detected_techniques']}</div>
                    <div class="stat-label">Techniques Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{mitre['total_techniques']}</div>
                    <div class="stat-label">Total Techniques</div>
                </div>
            </div>
            <div class="progress-section">
                <div class="progress-header">
                    <span style="color: #888;">Framework Coverage Progress</span>
                    <span style="color: #10a37f; font-weight: 600;">{mitre['coverage_percent']}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
            </div>
        </div>
        
        <!-- Recommendations -->
        <div class="section">
            <div class="section-title">
                <span class="section-icon">📋</span>
                Priority Recommendations
            </div>
            <ul class="recommendation-list">
                <li class="recommendation-item">
                    <span class="recommendation-icon">🔴</span>
                    <span class="recommendation-text">Immediately investigate and remediate {stats['critical']} critical severity alerts</span>
                    <span class="recommendation-priority priority-critical">CRITICAL</span>
                </li>
                <li class="recommendation-item">
                    <span class="recommendation-icon">🟠</span>
                    <span class="recommendation-text">Review {stats['high']} high-severity detections for potential threat escalation</span>
                    <span class="recommendation-priority priority-high">HIGH</span>
                </li>
                <li class="recommendation-item">
                    <span class="recommendation-icon">🟡</span>
                    <span class="recommendation-text">Tune detection rules to reduce false positive rate of {round(stats['false_positives']/stats['total_alerts']*100, 1)}%</span>
                    <span class="recommendation-priority priority-high">MEDIUM</span>
                </li>
                <li class="recommendation-item">
                    <span class="recommendation-icon">🔵</span>
                    <span class="recommendation-text">Update security awareness training based on trending DoS/DDoS attack patterns</span>
                    <span class="recommendation-priority" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">ONGOING</span>
                </li>
            </ul>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p class="footer-logo">PCDS Enterprise</p>
            <p>Predictive Cyber Defence System • AI-Powered Security Intelligence</p>
            <p style="margin-top: 12px; color: #444;">This report is automatically generated. For questions, contact your security operations team.</p>
            <p style="margin-top: 8px; font-size: 11px; color: #333;">Report ID: RPT-{datetime.now().strftime('%Y%m%d%H%M%S')} • Classification: Internal Use Only</p>
        </div>
    </div>
</body>
</html>
"""
    return html



def generate_compliance_html(framework: str, stats: Dict) -> str:
    """Generate compliance report HTML"""
    
    frameworks = {
        "soc2": {
            "name": "SOC 2 Type II",
            "controls": [
                {"id": "CC6.1", "name": "Logical Access Security", "status": "pass", "evidence": "Access logs reviewed"},
                {"id": "CC6.2", "name": "Data Protection", "status": "pass", "evidence": "Encryption verified"},
                {"id": "CC6.6", "name": "System Operations", "status": "attention", "evidence": "Minor gaps identified"},
                {"id": "CC7.1", "name": "Incident Detection", "status": "pass", "evidence": "SIEM operational"},
                {"id": "CC7.2", "name": "Incident Response", "status": "pass", "evidence": "Playbooks tested"},
            ]
        },
        "hipaa": {
            "name": "HIPAA Security Rule",
            "controls": [
                {"id": "164.308", "name": "Administrative Safeguards", "status": "pass", "evidence": "Policies documented"},
                {"id": "164.310", "name": "Physical Safeguards", "status": "pass", "evidence": "Access controls verified"},
                {"id": "164.312", "name": "Technical Safeguards", "status": "pass", "evidence": "Audit logs enabled"},
                {"id": "164.314", "name": "Organizational Requirements", "status": "attention", "evidence": "Review pending"},
            ]
        },
        "pci": {
            "name": "PCI-DSS v4.0",
            "controls": [
                {"id": "1.1", "name": "Firewall Configuration", "status": "pass", "evidence": "Rules reviewed"},
                {"id": "2.1", "name": "Default Passwords", "status": "pass", "evidence": "No defaults found"},
                {"id": "10.1", "name": "Audit Trails", "status": "pass", "evidence": "Logging enabled"},
                {"id": "11.4", "name": "IDS/IPS", "status": "pass", "evidence": "PCDS operational"},
            ]
        }
    }
    
    fw = frameworks.get(framework, frameworks["soc2"])
    
    pass_count = len([c for c in fw["controls"] if c["status"] == "pass"])
    total = len(fw["controls"])
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PCDS Compliance Report - {fw['name']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #fff; color: #333; }}
        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 3px solid #0066cc; padding-bottom: 20px; }}
        .header h1 {{ color: #0066cc; margin: 0; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #0066cc; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .score {{ font-size: 48px; font-weight: bold; color: #28a745; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
        th {{ background: #0066cc; color: white; }}
        .status-pass {{ color: #28a745; font-weight: bold; }}
        .status-attention {{ color: #ffc107; font-weight: bold; }}
        .status-fail {{ color: #dc3545; font-weight: bold; }}
        .footer {{ text-align: center; color: #888; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📋 Compliance Report</h1>
        <h2>{fw['name']}</h2>
        <p>Generated: {datetime.now().strftime('%B %d, %Y')}</p>
    </div>

    <div class="summary">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div class="score">{int(pass_count/total*100)}%</div>
                <div>Compliance Score</div>
            </div>
            <div>
                <div style="font-size: 32px; color: #28a745;">{pass_count}</div>
                <div>Controls Passed</div>
            </div>
            <div>
                <div style="font-size: 32px; color: #ffc107;">{total - pass_count}</div>
                <div>Needs Attention</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Control Assessment</h2>
        <table>
            <tr>
                <th>Control ID</th>
                <th>Control Name</th>
                <th>Status</th>
                <th>Evidence</th>
            </tr>
            {''.join(f'''
            <tr>
                <td>{c['id']}</td>
                <td>{c['name']}</td>
                <td class="status-{c['status']}">{'✓ PASS' if c['status'] == 'pass' else '⚠ ATTENTION' if c['status'] == 'attention' else '✗ FAIL'}</td>
                <td>{c['evidence']}</td>
            </tr>
            ''' for c in fw['controls'])}
        </table>
    </div>

    <div class="section">
        <h2>Security Metrics</h2>
        <table>
            <tr><td>Total Alerts Processed</td><td>{stats['total_alerts']}</td></tr>
            <tr><td>Critical Incidents</td><td>{stats['critical']}</td></tr>
            <tr><td>Mean Time to Detect</td><td>{stats['mean_time_to_detect']}</td></tr>
            <tr><td>Mean Time to Respond</td><td>{stats['mean_time_to_respond']}</td></tr>
        </table>
    </div>

    <div class="footer">
        <p>PCDS Enterprise - Compliance Reporting</p>
        <p>This report is for compliance documentation purposes.</p>
    </div>
</body>
</html>
"""
    return html


@router.post("/generate")
async def generate_report(request: ReportRequest) -> Dict:
    """
    Generate a threat report
    
    Returns report ID and HTML content
    """
    stats = get_threat_stats()
    threats = get_top_threats()
    mitre = get_mitre_coverage()
    
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    if request.report_type == "executive":
        html = generate_executive_html(stats, threats, mitre)
        title = request.title or "Executive Threat Report"
    elif request.report_type == "compliance":
        html = generate_compliance_html(request.compliance_framework or "soc2", stats)
        title = request.title or f"Compliance Report - {request.compliance_framework or 'SOC2'}"
    else:
        html = generate_executive_html(stats, threats, mitre)
        title = request.title or "Threat Report"
    
    generated_reports[report_id] = {
        "id": report_id,
        "type": request.report_type,
        "title": title,
        "html": html,
        "created_at": datetime.now().isoformat(),
        "stats": stats
    }
    
    return {
        "report_id": report_id,
        "title": title,
        "type": request.report_type,
        "created_at": datetime.now().isoformat()
    }


@router.get("/view/{report_id}")
async def view_report(report_id: str) -> HTMLResponse:
    """View generated report as HTML"""
    if report_id not in generated_reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return HTMLResponse(content=generated_reports[report_id]["html"])


@router.get("/download/{report_id}")
async def download_report(report_id: str) -> Response:
    """Download report as HTML file"""
    if report_id not in generated_reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report = generated_reports[report_id]
    
    return Response(
        content=report["html"],
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="{report_id}.html"'
        }
    )


@router.get("/download/{report_id}/pdf")
async def download_report_pdf(report_id: str) -> Response:
    """Download report as PDF file"""
    if report_id not in generated_reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report = generated_reports[report_id]
    html_content = report["html"]
    
    # Use xhtml2pdf (pure Python, works on Windows)
    try:
        from xhtml2pdf import pisa
        import io
        import re
        
        # Strip out complex CSS that xhtml2pdf can't handle
        # Remove style blocks with complex selectors
        html_clean = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
        # Remove inline styles with complex properties
        html_clean = re.sub(r'style="[^"]*transform[^"]*"', '', html_clean)
        html_clean = re.sub(r'style="[^"]*animation[^"]*"', '', html_clean)
        html_clean = re.sub(r'style="[^"]*gradient[^"]*"', '', html_clean)
        
        # Create simple PDF-friendly HTML
        full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{ font-family: Arial, sans-serif; font-size: 11pt; padding: 20px; }}
h1 {{ color: #0078d4; font-size: 24pt; margin-bottom: 20px; }}
h2 {{ color: #333; font-size: 16pt; margin-top: 30px; border-bottom: 2px solid #0078d4; padding-bottom: 5px; }}
h3 {{ color: #555; font-size: 14pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th {{ background-color: #0078d4; color: white; padding: 10px; text-align: left; }}
td {{ border: 1px solid #ddd; padding: 8px; }}
tr:nth-child(even) {{ background-color: #f9f9f9; }}
.critical {{ color: #dc3545; font-weight: bold; }}
.high {{ color: #fd7e14; font-weight: bold; }}
.medium {{ color: #ffc107; }}
.low {{ color: #28a745; }}
.stat {{ font-size: 24pt; font-weight: bold; color: #0078d4; }}
.label {{ font-size: 10pt; color: #666; }}
</style>
</head>
<body>
{html_clean}
</body>
</html>"""
        
        result = io.BytesIO()
        pisa_status = pisa.CreatePDF(full_html, dest=result)
        
        if pisa_status.err:
            raise HTTPException(status_code=500, detail="PDF conversion error")
        
        pdf_bytes = result.getvalue()
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{report_id}.pdf"'
            }
        )
        
    except ImportError:
        raise HTTPException(
            status_code=500, 
            detail="PDF generation not available. Install xhtml2pdf: pip install xhtml2pdf"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}"
        )


@router.get("/list")
async def list_reports() -> Dict:
    """List all generated reports"""
    return {
        "reports": [
            {
                "id": r["id"],
                "type": r["type"],
                "title": r["title"],
                "created_at": r["created_at"]
            }
            for r in generated_reports.values()
        ],
        "count": len(generated_reports)
    }


@router.delete("/{report_id}")
async def delete_report(report_id: str) -> Dict:
    """Delete a generated report"""
    if report_id in generated_reports:
        del generated_reports[report_id]
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Report not found")


@router.get("/templates")
async def get_report_templates() -> Dict:
    """Get available report templates"""
    return {
        "templates": [
            {
                "id": "executive",
                "name": "Executive Summary",
                "description": "High-level threat overview for leadership",
                "sections": ["Threat Overview", "Risk Assessment", "Top Threats", "MITRE Coverage", "Recommendations"]
            },
            {
                "id": "weekly",
                "name": "Weekly Threat Report",
                "description": "Detailed weekly threat analysis",
                "sections": ["Weekly Summary", "Alert Breakdown", "Trend Analysis", "Incidents", "Next Steps"]
            },
            {
                "id": "compliance",
                "name": "Compliance Report",
                "description": "Framework compliance assessment",
                "frameworks": ["SOC 2", "HIPAA", "PCI-DSS"]
            },
            {
                "id": "incident",
                "name": "Incident Report",
                "description": "Detailed incident documentation",
                "sections": ["Executive Summary", "Timeline", "Impact", "Root Cause", "Remediation"]
            }
        ]
    }


@router.post("/incident")
async def generate_incident_report(request: IncidentReportRequest) -> Dict:
    """Generate detailed incident report"""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Incident Report - {request.incident_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #fff; color: #333; }}
        .header {{ border-bottom: 3px solid #dc3545; padding-bottom: 20px; margin-bottom: 30px; }}
        .header h1 {{ color: #dc3545; margin: 0; }}
        .severity {{ display: inline-block; padding: 8px 16px; border-radius: 4px; color: white; font-weight: bold; }}
        .severity-critical {{ background: #dc3545; }}
        .severity-high {{ background: #fd7e14; }}
        .severity-medium {{ background: #ffc107; color: #333; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #0066cc; }}
        .timeline {{ border-left: 3px solid #0066cc; padding-left: 20px; }}
        .timeline-item {{ margin-bottom: 15px; }}
        .timeline-time {{ color: #888; font-size: 14px; }}
        ul {{ line-height: 1.8; }}
        .footer {{ text-align: center; color: #888; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚨 Incident Report</h1>
        <p><strong>Incident ID:</strong> {request.incident_id}</p>
        <p><strong>Title:</strong> {request.title}</p>
        <p><span class="severity severity-{request.severity}">{request.severity.upper()}</span></p>
        <p><strong>Analyst:</strong> {request.analyst}</p>
        <p><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>{request.summary}</p>
    </div>

    <div class="section">
        <h2>Timeline of Events</h2>
        <div class="timeline">
            {''.join(f'''
            <div class="timeline-item">
                <div class="timeline-time">{e.get('time', 'N/A')}</div>
                <div>{e.get('event', 'N/A')}</div>
            </div>
            ''' for e in request.timeline)}
        </div>
    </div>

    <div class="section">
        <h2>Affected Systems</h2>
        <ul>
            {''.join(f'<li>{s}</li>' for s in request.affected_systems)}
        </ul>
    </div>

    <div class="section">
        <h2>MITRE ATT&CK Techniques</h2>
        <ul>
            {''.join(f'<li>{t}</li>' for t in request.mitre_techniques)}
        </ul>
    </div>

    <div class="section">
        <h2>Remediation Steps</h2>
        <ol>
            {''.join(f'<li>{s}</li>' for s in request.remediation_steps)}
        </ol>
    </div>

    <div class="footer">
        <p>PCDS Enterprise - Incident Response</p>
        <p>CONFIDENTIAL - For internal use only</p>
    </div>
</body>
</html>
"""
    
    report_id = f"INC-{request.incident_id}"
    generated_reports[report_id] = {
        "id": report_id,
        "type": "incident",
        "title": request.title,
        "html": html,
        "created_at": datetime.now().isoformat(),
        "stats": {}
    }
    
    return {
        "report_id": report_id,
        "title": request.title,
        "type": "incident"
    }


@router.get("/quick-stats")
async def get_quick_stats() -> Dict:
    """Get quick stats for report generation"""
    return {
        "stats": get_threat_stats(),
        "top_threats": get_top_threats(5),
        "mitre_coverage": get_mitre_coverage()
    }
