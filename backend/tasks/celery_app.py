"""
Celery Task Queue Configuration for PCDS Enterprise
Background task processing for scoring, correlation, and analysis
"""
from celery import Celery
from celery.schedules import crontab
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    'pcds_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
)


# ===== BACKGROUND TASKS =====

@celery_app.task(name='tasks.calculate_entity_risk')
def calculate_entity_risk(entity_id: str):
    """
    Background task: Calculate entity risk score
    Allows API to return instantly while processing happens async
    """
    try:
        from engine.scoring_engine import calculate_risk_score
        from config.database import db_manager
        
        logger.info(f"Calculating risk for entity: {entity_id}")
        
        # Calculate risk score (CPU intensive)
        risk_score = calculate_risk_score(entity_id)
        
        # Update database
        db_manager.execute_update("""
            UPDATE entities 
            SET threat_score = ?, last_updated = ?
            WHERE identifier = ?
        """, (risk_score, datetime.utcnow(), entity_id))
        
        logger.info(f"Risk calculated for {entity_id}: {risk_score}")
        return {'entity_id': entity_id, 'risk_score': risk_score}
    
    except Exception as e:
        logger.error(f"Risk calculation error for {entity_id}: {e}")
        raise


@celery_app.task(name='tasks.correlate_campaign')
def correlate_campaign(detection_id: int):
    """
    Background task: Correlate detection to campaigns
    """
    try:
        from engine.campaign_correlator import correlate_to_campaigns
        
        logger.info(f"Correlating detection {detection_id} to campaigns")
        
        # Perform correlation (computationally expensive)
        campaigns = correlate_to_campaigns(detection_id)
        
        logger.info(f"Detection {detection_id} correlated to {len(campaigns)} campaigns")
        return {'detection_id': detection_id, 'campaigns': campaigns}
    
    except Exception as e:
        logger.error(f"Campaign correlation error for {detection_id}: {e}")
        raise


@celery_app.task(name='tasks.analyze_threat_intelligence')
def analyze_threat_intelligence():
    """
    Periodic task: Analyze threat patterns and update intelligence
    """
    try:
        from engine import threat_analyzer
        
        logger.info("Running threat intelligence analysis")
        
        # Analyze recent detections for patterns
        insights = threat_analyzer.analyze_patterns()
        
        logger.info(f"Threat analysis complete: {len(insights)} insights")
        return insights
    
    except Exception as e:
        logger.error(f"Threat intelligence analysis error: {e}")
        raise


@celery_app.task(name='tasks.generate_report')
def generate_report(report_type: str, time_range: str):
    """
    Background task: Generate comprehensive reports
    """
    try:
        logger.info(f"Generating {report_type} report for {time_range}")
        
        # Report generation logic
        # This can take 30+ seconds, so run in background
        
        logger.info(f"Report generated: {report_type}")
        return {'report_type': report_type, 'status': 'complete'}
    
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise


@celery_app.task(name='tasks.cleanup_old_data')
def cleanup_old_data():
    """
    Periodic task: Clean up old data based on retention policy
    """
    try:
        from config.database import db_manager
        from datetime import timedelta
        
        logger.info("Running data cleanup")
        
        # Delete detections older than 90 days
        cutoff = datetime.utcnow() - timedelta(days=90)
        
        result = db_manager.execute_update("""
            DELETE FROM detections 
            WHERE detected_at < ?
        """, (cutoff,))
        
        logger.info(f"Cleanup complete: {result} records deleted")
        return {'deleted': result}
    
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise


# ===== PERIODIC TASKS (Cron-like scheduling) =====

celery_app.conf.beat_schedule = {
    'threat-intelligence-hourly': {
        'task': 'tasks.analyze_threat_intelligence',
        'schedule': crontab(minute=0),  # Every hour
    },
    'cleanup-daily': {
        'task': 'tasks.cleanup_old_data',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
}


# ===== HELPER FUNCTIONS =====

def queue_entity_scoring(entity_id: str):
    """Queue entity risk scoring in background"""
    return calculate_entity_risk.delay(entity_id)


def queue_campaign_correlation(detection_id: int):
    """Queue campaign correlation in background"""
    return correlate_campaign.delay(detection_id)


def queue_report_generation(report_type: str, time_range: str):
    """Queue report generation in background"""
    return generate_report.delay(report_type, time_range)
