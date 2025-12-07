"""
PCDS Enterprise - Email Notification Service
Sends email alerts for critical detections and approval requests
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from datetime import datetime
import os


class EmailNotificationService:
    """
    Email notification service for PCDS Enterprise
    
    Configuration via environment variables:
    - PCDS_SMTP_HOST: SMTP server hostname
    - PCDS_SMTP_PORT: SMTP port (default 587)
    - PCDS_SMTP_USER: SMTP username
    - PCDS_SMTP_PASS: SMTP password
    - PCDS_ALERT_EMAIL: Default recipient for alerts
    """
    
    def __init__(self):
        self.smtp_host = os.getenv('PCDS_SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('PCDS_SMTP_PORT', '587'))
        self.smtp_user = os.getenv('PCDS_SMTP_USER', '')
        self.smtp_pass = os.getenv('PCDS_SMTP_PASS', '')
        self.default_recipient = os.getenv('PCDS_ALERT_EMAIL', '')
        self.from_email = os.getenv('PCDS_FROM_EMAIL', 'pcds-alerts@enterprise.local')
        
        self.enabled = bool(self.smtp_user and self.smtp_pass)
        
        if not self.enabled:
            print("⚠️ Email notifications disabled - SMTP credentials not configured")
    
    def send_email(self, to: List[str], subject: str, html_body: str, 
                   text_body: Optional[str] = None) -> bool:
        """Send an email notification"""
        if not self.enabled:
            print(f"📧 [SIMULATED] Email to {to}: {subject}")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to)
            
            # Attach text and HTML versions
            if text_body:
                msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.from_email, to, msg.as_string())
            
            print(f"✅ Email sent to {to}: {subject}")
            return True
            
        except Exception as e:
            print(f"❌ Email failed: {e}")
            return False
    
    def send_critical_alert(self, detection: dict, recipients: Optional[List[str]] = None) -> bool:
        """Send alert for critical detection"""
        to = recipients or [self.default_recipient] if self.default_recipient else []
        if not to:
            return False
        
        subject = f"🚨 CRITICAL ALERT: {detection.get('title', 'Unknown Threat')}"
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #1e293b; color: #f1f5f9; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: #0f172a; border-radius: 10px; padding: 20px; border: 1px solid #dc2626;">
                <h2 style="color: #dc2626; margin-top: 0;">🚨 Critical Security Alert</h2>
                
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; color: #94a3b8;">Detection</td>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; font-weight: bold;">{detection.get('title', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; color: #94a3b8;">Severity</td>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; color: #dc2626; font-weight: bold;">{detection.get('severity', 'N/A').upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; color: #94a3b8;">Entity</td>
                        <td style="padding: 10px; border-bottom: 1px solid #334155;">{detection.get('entity_id', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; color: #94a3b8;">MITRE Technique</td>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; font-family: monospace; color: #22d3ee;">{detection.get('technique_id', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; color: #94a3b8;">Description</td>
                        <td style="padding: 10px;">{detection.get('description', 'No description')}</td>
                    </tr>
                </table>
                
                <div style="margin-top: 20px; padding: 15px; background: #dc2626; border-radius: 8px; text-align: center;">
                    <a href="http://localhost:3000/detections" style="color: white; text-decoration: none; font-weight: bold;">
                        View in PCDS Dashboard →
                    </a>
                </div>
                
                <p style="color: #64748b; font-size: 12px; margin-top: 20px; text-align: center;">
                    PCDS Enterprise v2.0 | Automated Security Alert
                </p>
            </div>
        </body>
        </html>
        """
        
        text_body = f"""
        🚨 CRITICAL SECURITY ALERT
        
        Detection: {detection.get('title', 'N/A')}
        Severity: {detection.get('severity', 'N/A').upper()}
        Entity: {detection.get('entity_id', 'N/A')}
        MITRE: {detection.get('technique_id', 'N/A')}
        Description: {detection.get('description', 'No description')}
        
        View in dashboard: http://localhost:3000/detections
        """
        
        return self.send_email(to, subject, html_body, text_body)
    
    def send_approval_request(self, approval: dict, recipients: Optional[List[str]] = None) -> bool:
        """Send notification for pending approval request"""
        to = recipients or [self.default_recipient] if self.default_recipient else []
        if not to:
            return False
        
        subject = f"⏳ Approval Needed: {approval.get('action', 'Unknown Action')} on {approval.get('entity_id', 'Unknown')}"
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #1e293b; color: #f1f5f9; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: #0f172a; border-radius: 10px; padding: 20px; border: 1px solid #f59e0b;">
                <h2 style="color: #f59e0b; margin-top: 0;">⏳ Action Requires Approval</h2>
                
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; color: #94a3b8;">Proposed Action</td>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; font-weight: bold;">{approval.get('action', 'N/A').upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; color: #94a3b8;">Target</td>
                        <td style="padding: 10px; border-bottom: 1px solid #334155;">{approval.get('entity_id', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #334155; color: #94a3b8;">Confidence</td>
                        <td style="padding: 10px; border-bottom: 1px solid #334155;">{approval.get('confidence', 0) * 100:.0f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; color: #94a3b8;">Reason</td>
                        <td style="padding: 10px;">{approval.get('reason', 'No reason provided')}</td>
                    </tr>
                </table>
                
                <div style="margin-top: 20px; display: flex; gap: 10px; justify-content: center;">
                    <a href="http://localhost:3000/approvals" style="padding: 12px 24px; background: #22c55e; color: white; text-decoration: none; border-radius: 8px; font-weight: bold;">
                        Review & Approve
                    </a>
                </div>
                
                <p style="color: #64748b; font-size: 12px; margin-top: 20px; text-align: center;">
                    PCDS Enterprise v2.0 | Approval Request
                </p>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(to, subject, html_body)


# Global instance
email_service = EmailNotificationService()


# Convenience functions
def send_critical_alert(detection: dict, recipients: List[str] = None) -> bool:
    """Send critical detection alert"""
    return email_service.send_critical_alert(detection, recipients)


def send_approval_request(approval: dict, recipients: List[str] = None) -> bool:
    """Send approval request notification"""
    return email_service.send_approval_request(approval, recipients)
