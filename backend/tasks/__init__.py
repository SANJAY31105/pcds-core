"""
Tasks package initialization
"""
from .celery_app import (
    celery_app,
    queue_entity_scoring,
    queue_campaign_correlation,
    queue_report_generation
)

__all__ = [
    'celery_app',
    'queue_entity_scoring',
    'queue_campaign_correlation',
    'queue_report_generation'
]
