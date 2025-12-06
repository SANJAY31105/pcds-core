"""
Campaign Correlator
Automatically correlates detections into multi-stage attack campaigns

Features:
- Groups related detections by entity, time, and techniques
- Identifies attack progression through kill chain stages
- Calculates campaign severity based on constituent detections
- Tracks affected entities
- Generates campaign metadata and summaries
"""

from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
from config.settings import settings


class CampaignCorrelator:
    """
    Correlates detections into attack campaigns
    
    Correlation criteria:
    1. Same entity (or related entities)
    2. Temporal proximity (within configured time window)
    3. Kill chain progression
    4. Technique relationships
    """
    
    def __init__(self, time_window_hours: int = None):
        """
        Initialize campaign correlator
        
        Args:
            time_window_hours: Max time between detections (default from settings)
        """
        self.time_window_hours = time_window_hours or settings.CAMPAIGN_TIME_WINDOW_HOURS
        self.active_campaigns = {}  # campaign_id -> campaign data
    
    def correlate_detections(
        self,
        detections: List[Dict],
        existing_campaigns: List[Dict] = None
    ) -> List[Dict]:
        """
        Correlate detections into campaigns
        
        Args:
            detections: List of detection dictionaries
            existing_campaigns: Previously created campaigns
        
        Returns:
            List of campaign dictionaries
        """
        if existing_campaigns:
            for campaign in existing_campaigns:
                self.active_campaigns[campaign['id']] = campaign
        
        # Group detections by entity
        by_entity = defaultdict(list)
        for detection in detections:
            entity_id = detection.get('entity_id')
            if entity_id:
                by_entity[entity_id].append(detection)
        
        campaigns = []
        
        # Process each entity's detections
        for entity_id, entity_detections in by_entity.items():
            # Sort by time
            entity_detections.sort(key=lambda d: d.get('detected_at', ''))
            
            # Find or create campaigns
            entity_campaigns = self._correlate_entity_detections(entity_id, entity_detections)
            campaigns.extend(entity_campaigns)
        
        return campaigns
    
    def _correlate_entity_detections(
        self,
        entity_id: str,
        detections: List[Dict]
    ) -> List[Dict]:
        """Correlate detections for a single entity"""
        campaigns = []
        current_campaign = None
        current_detections = []
        
        for detection in detections:
            detected_at = self._parse_timestamp(detection.get('detected_at'))
            
            # Check if this detection belongs to current campaign
            if current_campaign and current_detections:
                last_detection_time = self._parse_timestamp(current_detections[-1].get('detected_at'))
                time_diff = (detected_at - last_detection_time).total_seconds() / 3600
                
                # Same campaign if within time window
                if time_diff <= self.time_window_hours:
                    current_detections.append(detection)
                else:
                    # Time window exceeded - finalize current campaign
                    if len(current_detections) >= 2:  # Only create campaign if 2+ detections
                        campaigns.append(self._create_campaign(entity_id, current_detections))
                    
                    # Start new campaign
                    current_campaign = None
                    current_detections = [detection]
            else:
                # Start first campaign
                current_detections.append(detection)
                current_campaign = True
        
        # Finalize last campaign
        if current_detections and len(current_detections) >= 2:
            campaigns.append(self._create_campaign(entity_id, current_detections))
        
        return campaigns
    
    def _create_campaign(self, entity_id: str, detections: List[Dict]) -> Dict:
        """Create a campaign from correlated detections"""
        campaign_id = f"campaign_{uuid.uuid4().hex[:12]}"
        
        # Extract unique tactics and techniques
        tactics = set()
        techniques = set()
        kill_chain_stages = set()
        
        for detection in detections:
            if detection.get('tactic_id'):
                tactics.add(detection['tactic_id'])
            if detection.get('technique_id'):
                techniques.add(detection['technique_id'])
            if detection.get('kill_chain_stage'):
                kill_chain_stages.add(detection['kill_chain_stage'])
        
        # Calculate campaign severity (highest detection severity)
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        campaign_severity = 'low'
        for detection in detections:
            det_severity = detection.get('severity', 'low')
            if severity_order.get(det_severity, 0) > severity_order.get(campaign_severity, 0):
                campaign_severity = det_severity
        
        # Time range
        start_time = min(self._parse_timestamp(d.get('detected_at')) for d in detections)
        end_time = max(self._parse_timestamp(d.get('detected_at')) for d in detections)
        
        # Generate campaign name
        campaign_name = self._generate_campaign_name(tactics, techniques, entity_id)
        
        # Generate description
        description = self._generate_campaign_description(detections, tactics, techniques)
        
        campaign = {
            'id': campaign_id,
            'name': campaign_name,
            'description': description,
            'severity': campaign_severity,
            'total_detections': len(detections),
            'affected_entities': 1,  # For now, single entity
            'started_at': start_time.isoformat(),
            'last_activity': end_time.isoformat(),
            'status': 'active',
            'tactics_used': list(tactics),
            'techniques_used': list(techniques),
            'kill_chain_progress': max(kill_chain_stages) if kill_chain_stages else 0,
            'metadata': {
                'entity_id': entity_id,
                'detection_ids': [d.get('id') for d in detections if d.get('id')],
                'unique_tactics': len(tactics),
                'unique_techniques': len(techniques),
                'kill_chain_stages': sorted(list(kill_chain_stages)),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            }
        }
        
        return campaign
    
    def _generate_campaign_name(
        self,
        tactics: Set[str],
        techniques: Set[str],
        entity_id: str
    ) -> str:
        """Generate human-readable campaign name"""
        # Use primary tactic or generic name
        tactic_names = {
            'TA0001': 'Initial Access',
            'TA0002': 'Execution',
            'TA0003': 'Persistence',
            'TA0004': 'Privilege Escalation',
            'TA0005': 'Defense Evasion',
            'TA0006': 'Credential Access',
            'TA0007': 'Discovery',
            'TA0008': 'Lateral Movement',
            'TA0009': 'Collection',
            'TA0010': 'Command and Control',
            'TA0011': 'Exfiltration',
            'TA0040': 'Impact'
        }
        
        if len(tactics) == 1:
            tactic_id = list(tactics)[0]
            tactic_name = tactic_names.get(tactic_id, 'Attack')
            return f"{tactic_name} Campaign on {entity_id}"
        elif len(tactics) >= 4:
            return f"Multi-Stage Attack Campaign on {entity_id}"
        else:
            primary_tactic = max(tactics) if tactics else 'TA0001'
            tactic_name = tactic_names.get(primary_tactic, 'Attack')
            return f"{tactic_name} Campaign on {entity_id}"
    
    def _generate_campaign_description(
        self,
        detections: List[Dict],
        tactics: Set[str],
        techniques: Set[str]
    ) -> str:
        """Generate campaign description"""
        lines = [
            f"Multi-stage attack campaign with {len(detections)} detections",
            f"spanning {len(tactics)} tactics and {len(techniques)} techniques."
        ]
        
        # Add key detections
        high_severity = [d for d in detections if d.get('severity') in ['critical', 'high']]
        if high_severity:
            lines.append(f"Includes {len(high_severity)} high-severity detections:")
            for detection in high_severity[:3]:  # Top 3
                lines.append(f"  - {detection.get('title', 'Unknown')}")
        
        return " ".join(lines)
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse ISO timestamp to datetime"""
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            return datetime.utcnow()
    
    def update_campaign(
        self,
        campaign_id: str,
        new_detections: List[Dict]
    ) -> Dict:
        """Add new detections to existing campaign"""
        if campaign_id not in self.active_campaigns:
            return None
        
        campaign = self.active_campaigns[campaign_id]
        
        # Update detection count
        campaign['total_detections'] += len(new_detections)
        
        # Update last activity time
        latest_detection = max(new_detections, key=lambda d: d.get('detected_at', ''))
        campaign['last_activity'] = latest_detection.get('detected_at')
        
        # Update techniques and tactics
        for detection in new_detections:
            if detection.get('technique_id'):
                if detection['technique_id'] not in campaign['techniques_used']:
                    campaign['techniques_used'].append(detection['technique_id'])
            if detection.get('tactic_id'):
                if detection['tactic_id'] not in campaign['tactics_used']:
                    campaign['tactics_used'].append(detection['tactic_id'])
        
        # Update kill chain progress
        kill_chain_stages = [d.get('kill_chain_stage', 0) for d in new_detections if d.get('kill_chain_stage')]
        if kill_chain_stages:
            campaign['kill_chain_progress'] = max(
                campaign['kill_chain_progress'],
                max(kill_chain_stages)
            )
        
        # Add detection IDs
        detection_ids = campaign['metadata'].get('detection_ids', [])
        detection_ids.extend([d.get('id') for d in new_detections if d.get('id')])
        campaign['metadata']['detection_ids'] = detection_ids
        
        return campaign
    
    def close_campaign(self, campaign_id: str, resolution: str = 'resolved') -> Dict:
        """Close an active campaign"""
        if campaign_id in self.active_campaigns:
            campaign = self.active_campaigns[campaign_id]
            campaign['status'] = resolution
            campaign['ended_at'] = datetime.utcnow().isoformat()
            return campaign
        return None


# Global campaign correlator instance
campaign_correlator = CampaignCorrelator()
