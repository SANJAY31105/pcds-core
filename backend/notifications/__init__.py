# Email notification service for PCDS Enterprise
from .email_service import (
    email_service, 
    send_critical_alert, 
    send_approval_request,
    EmailNotificationService
)

__all__ = [
    'email_service',
    'send_critical_alert',
    'send_approval_request',
    'EmailNotificationService'
]
