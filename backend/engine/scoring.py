"""
Attack Signal Intelligence - Entity Urgency Scoring Algorithm
Matches Vectra AI methodology for threat prioritization

Formula: Urgency = (Severity × Recency × Confidence × Asset_Value) + Progression_Bonus
Range: 0-100

Urgency Levels:
- Critical: 75-100 (Immediate action required)
- High: 50-74 (Priority investigation)
- Medium: 25-49 (Monitor closely)
- Low: 0-24 (Routine monitoring)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import math
import json


class EntityScoringEngine:
    """
    Enterprise-grade entity urgency scoring matching Vectra AI
    
    Analyzes multiple factors to calculate threat urgency:
    1. Severity - Impact level of detections
    2. Detection Count - Volume of malicious activity
    3. Recency - How recent the activity is
    4. Confidence - Detection accuracy
    5. Attack Progression - Kill chain advancement
    6. Asset Value - Business criticality
    """
    
    # Severity weights (0.0-1.0)
    SEVERITY_WEIGHTS = {
        'critical': 1.0,
        'high': 0.75,
        'medium': 0.5,
        'low': 0.25
    }
    
    # Detection type risk multipliers (increases effective severity)
    DETECTION_RISK_MULTIPLIERS = {
        # Credential Access - Highest risk
        'credential_dumping': 1.5,
        'kerberoasting': 1.4,
        'password_spraying': 1.3,
        'brute_force': 1.2,
        'keylogging': 1.3,
        
        # Lateral Movement - High risk
        'pass_the_hash': 1.5,
        'pass_the_ticket': 1.5,
        'rdp_lateral': 1.3,
        'smb_lateral': 1.2,
        'ssh_lateral': 1.2,
        'psexec': 1.3,
        'wmi_lateral': 1.3,
        
        # Command & Control - Critical
        'c2_beaconing': 1.5,
        'dns_tunneling': 1.4,
        'proxy_usage': 1.2,
        
        # Exfiltration - Critical
        'data_exfiltration': 1.5,
        'large_upload': 1.3,
        'dns_exfiltration': 1.4,
        'cloud_upload': 1.2,
        
        # Privilege Escalation
        'token_manipulation': 1.4,
        'process_injection': 1.3,
        'uac_bypass': 1.3,
        'privilege_escalation': 1.4,
        
        # Impact - Highest risk
        'ransomware': 2.0,
        'data_destruction': 1.8,
        'backup_deletion': 1.6,
        
        # Reconnaissance - Lower but important
        'network_scan': 0.8,
        'port_scan': 0.7,
        'account_enumeration': 0.7,
        'host_enumeration': 0.7,
        
        # Defense Evasion
        'disable_security': 1.4,
        'log_deletion': 1.3,
        
        # Execution
        'powershell_execution': 1.1,
        'cmd_execution': 1.0,
        'script_execution': 1.1,
        
        # Persistence
        'scheduled_task': 1.2,
        'account_creation': 1.3,
    }
    
    def calculate_urgency_score(
        self,
        entity_id: str,
        detections: List[Dict],
        asset_value: int = 50,  # 0-100 scale
        current_urgency: int = 0  # Previous score for trend analysis
    ) -> Dict:
        """
        Calculate comprehensive urgency score for an entity
        
        Args:
            entity_id: Unique entity identifier
            detections: List of detection dictionaries
            asset_value: Business criticality (0-100)
            current_urgency: Previous urgency score
        
        Returns:
            {
                'urgency_score': int (0-100),
                'urgency_level': str (critical|high|medium|low),
                'urgency_change': int (score delta),
                'factors': dict (breakdown of contributing factors),
                'recommendations': list (actionable items),
                'metadata': dict (additional context)
            }
        """
        if not detections:
            return self._empty_score(entity_id)
        
        # Calculate individual factor scores
        severity_score = self._calculate_severity_score(detections)
        count_score = self._calculate_count_score(detections)
        recency_score = self._calculate_recency_score(detections)
        confidence_score = self._calculate_confidence_score(detections)
        progression_score = self._calculate_progression_score(detections)
        
        # Asset value multiplier (0.5x to 1.5x)
        asset_multiplier = 0.5 + (asset_value / 100.0)
        
        # Calculate base score (sum of factors)
        base_score = (
            severity_score +      # 0-40 points
            count_score +         # 0-20 points
            recency_score +       # 0-20 points
            confidence_score +    # 0-10 points
            progression_score     # 0-15 points
        )
        
        # Apply asset multiplier
        final_score = min(int(base_score * asset_multiplier), 100)
        
        # Determine urgency level
        urgency_level = self._determine_urgency_level(final_score)
        
        # Calculate score change
        urgency_change = final_score - current_urgency
        
        # Generate actionable recommendations
        recommendations = self._generate_recommendations(
            detections, urgency_level, progression_score, urgency_change
        )
        
        # Compile metadata
        metadata = self._compile_metadata(detections)
        
        return {
            'urgency_score': final_score,
            'urgency_level': urgency_level,
            'urgency_change': urgency_change,
            'trend': 'increasing' if urgency_change > 5 else 'decreasing' if urgency_change < -5 else 'stable',
            'factors': {
                'severity_score': round(severity_score, 1),
                'count_score': round(count_score, 1),
                'recency_score': round(recency_score, 1),
                'confidence_score': round(confidence_score, 1),
                'progression_score': round(progression_score, 1),
                'asset_multiplier': round(asset_multiplier, 2),
                'base_score': round(base_score, 1)
            },
            'recommendations': recommendations,
            'metadata': metadata
        }
    
    def _calculate_severity_score(self, detections: List[Dict]) -> float:
        """
        Severity scoring with weighted average + peak severity bonus
        Max: 40 points
        
        Combines:
        - Maximum severity (70% weight)
        - Average weighted severity (30% weight)
        - Detection type risk multipliers
        """
        if not detections:
            return 0
        
        # Get highest severity
        max_severity_weight = max(
            self.SEVERITY_WEIGHTS.get(d.get('severity', 'low'), 0.25)
            for d in detections
        )
        
        # Apply detection type multipliers
        weighted_scores = []
        for detection in detections:
            severity = detection.get('severity', 'low')
            detection_type = detection.get('detection_type', '')
            
            base_weight = self.SEVERITY_WEIGHTS.get(severity, 0.25)
            multiplier = self.DETECTION_RISK_MULTIPLIERS.get(detection_type, 1.0)
            
            weighted_scores.append(base_weight * multiplier)
        
        # Average weighted score
        avg_weighted = sum(weighted_scores) / len(weighted_scores)
        
        # Combine: 70% max severity + 30% average
        combined = (max_severity_weight * 0.7) + (avg_weighted * 0.3)
        
        return combined * 40
    
    def _calculate_count_score(self, detections: List[Dict]) -> float:
        """
        Detection count scoring with logarithmic scaling
        Max: 20 points
        
        Prevents score inflation from detection spam while
        still rewarding multiple distinct attacks
        """
        count = len(detections)
        
        if count == 0:
            return 0
        elif count == 1:
            return 5
        elif count <= 3:
            return 10
        elif count <= 5:
            return 14
        elif count <= 10:
            return 17
        else:
            # Logarithmic scaling for high counts
            return min(20, 17 + math.log10(count - 9) * 3)
    
    def _calculate_recency_score(self, detections: List[Dict]) -> float:
        """
        Recency scoring - exponential decay
        Max: 20 points
        
        Most recent detection determines urgency.
        Exponential decay favors recent activity.
        """
        if not detections:
            return 0
        
        now = datetime.utcnow()
        
        # Find most recent detection
        most_recent = max(
            detections,
            key=lambda d: self._parse_timestamp(d.get('detected_at', '1970-01-01T00:00:00'))
        )
        
        detected_at = self._parse_timestamp(most_recent.get('detected_at', '1970-01-01T00:00:00'))
        hours_ago = (now - detected_at).total_seconds() / 3600
        
        # Exponential decay scoring
        if hours_ago < 1:
            return 20  # Last hour: maximum urgency
        elif hours_ago < 4:
            return 18
        elif hours_ago < 12:
            return 15
        elif hours_ago < 24:
            return 12
        elif hours_ago < 72:  # 3 days
            return 8
        elif hours_ago < 168:  # 1 week
            return 5
        else:
            # Decay with 2-week half-life
            return max(1, 5 * math.exp(-hours_ago / 336))
    
    def _calculate_confidence_score(self, detections: List[Dict]) -> float:
        """
        Confidence scoring - average confidence weighted by severity
        Max: 10 points
        
        Higher confidence in high-severity detections matters more
        """
        if not detections:
            return 0
        
        weighted_confidences = []
        for detection in detections:
            confidence = detection.get('confidence_score', 0.5)
            severity = detection.get('severity', 'low')
            severity_weight = self.SEVERITY_WEIGHTS.get(severity, 0.25)
            
            weighted_confidences.append(confidence * severity_weight)
        
        avg_confidence = sum(weighted_confidences) / len(weighted_confidences)
        
        return avg_confidence * 10
    
    def _calculate_progression_score(self, detections: List[Dict]) -> float:
        """
        Attack progression (kill chain advancement) scoring
        Max: 15 points
        
        Bonus points for:
        - Multiple kill chain stages
        - Sequential progression
        - Short time between stages (rapid attack)
        """
        if not detections:
            return 0
        
        # Get unique kill chain stages
        stages = set()
        for detection in detections:
            stage = detection.get('kill_chain_stage')
            if stage:
                stages.add(stage)
        
        if not stages:
            return 0
        
        # Points for stage diversity
        stage_count = len(stages)
        if stage_count == 1:
            base_points = 0
        elif stage_count == 2:
            base_points = 5
        elif stage_count == 3:
            base_points = 9
        elif stage_count >= 4:
            base_points = 12
        
        # Bonus for progression beyond initial access
        max_stage = max(stages)
        if max_stage >= 8:  # Lateral movement or beyond
            base_points += 3
        
        # Bonus for rapid progression
        if len(detections) >= 2:
            time_span_hours = self._get_time_span_hours(detections)
            if time_span_hours < 4 and stage_count >= 2:
                base_points += 2  # Rapid multi-stage attack
        
        return min(base_points, 15)
    
    def _determine_urgency_level(self, score: int) -> str:
        """Determine urgency level from score"""
        if score >= 75:
            return 'critical'
        elif score >= 50:
            return 'high'
        elif score >= 25:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(
        self,
        detections: List[Dict],
        urgency_level: str,
        progression_score: float,
        urgency_change: int
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Critical/High urgency actions
        if urgency_level == 'critical':
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Isolate entity from network")
            recommendations.append("🔒 Initiate incident response procedure")
            recommendations.append("💾 Preserve forensic evidence immediately")
        
        if urgency_level in ['critical', 'high']:
            recommendations.append("📋 Create investigation case")
            recommendations.append("📊 Review entity activity timeline")
            recommendations.append("🔍 Check for lateral movement to other entities")
        
        # Attack progression warnings
        if progression_score >= 9:
            recommendations.append("⚠️ Multi-stage attack detected - review full kill chain")
            recommendations.append("🎯 Identify initial access vector")
        
        # Detection-specific recommendations
        detection_types = set(d.get('detection_type', '') for d in detections)
        
        if any(t in detection_types for t in ['credential_dumping', 'kerberoasting']):
            recommendations.append("🔐 Reset credentials for affected accounts")
            recommendations.append("📝 Review Active Directory event logs")
        
        if any('lateral' in t for t in detection_types):
            recommendations.append("🗺️ Map lateral movement path")
            recommendations.append("🔑 Identify compromised credentials")
        
        if any(t in detection_types for t in ['c2_beaconing', 'dns_tunneling']):
            recommendations.append("🚫 Block C2 infrastructure at firewall")
            recommendations.append("🔍 Analyze network traffic for other infected hosts")
        
        if any(t in detection_types for t in ['data_exfiltration', 'large_upload']):
            recommendations.append("📁 Identify what data was accessed")
            recommendations.append("📋 Review DLP policies")
        
        if 'ransomware' in detection_types:
            recommendations.append("💾 Initiate backup recovery procedures")
            recommendations.append("🚨 Activate ransomware response playbook")
        
        # Trend-based recommendations
        if urgency_change > 10:
            recommendations.append("📈 Urgency increasing rapidly - prioritize investigation")
        
        return recommendations
    
    def _compile_metadata(self, detections: List[Dict]) -> Dict:
        """Compile additional metadata about detections"""
        unique_techniques = set(d.get('technique_id') for d in detections if d.get('technique_id'))
        unique_tactics = set(d.get('tactic_id') for d in detections if d.get('tactic_id'))
        
        severity_breakdown = {
            'critical': sum(1 for d in detections if d.get('severity') == 'critical'),
            'high': sum(1 for d in detections if d.get('severity') == 'high'),
            'medium': sum(1 for d in detections if d.get('severity') == 'medium'),
            'low': sum(1 for d in detections if d.get('severity') == 'low')
        }
        
        return {
            'total_detections': len(detections),
            'unique_techniques': len(unique_techniques),
            'unique_tactics': len(unique_tactics),
            'severity_breakdown': severity_breakdown,
            'time_span_hours': round(self._get_time_span_hours(detections), 2),
            'technique_ids': list(unique_techniques),
            'tactic_ids': list(unique_tactics)
        }
    
    def _get_time_span_hours(self, detections: List[Dict]) -> float:
        """Get time span between first and last detection in hours"""
        if len(detections) < 2:
            return 0
        
        times = [
            self._parse_timestamp(d.get('detected_at', '1970-01-01T00:00:00'))
            for d in detections
        ]
        
        return (max(times) - min(times)).total_seconds() / 3600
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse ISO timestamp string to datetime"""
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            return datetime(1970, 1, 1)
    
    def _empty_score(self, entity_id: str) -> Dict:
        """Return empty score structure when no detections"""
        return {
            'urgency_score': 0,
            'urgency_level': 'low',
            'urgency_change': 0,
            'trend': 'stable',
            'factors': {
                'severity_score': 0,
                'count_score': 0,
                'recency_score': 0,
                'confidence_score': 0,
                'progression_score': 0,
                'asset_multiplier': 1.0,
                'base_score': 0
            },
            'recommendations': [],
            'metadata': {
                'total_detections': 0,
                'unique_techniques': 0,
                'unique_tactics': 0,
                'severity_breakdown': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                'time_span_hours': 0,
                'technique_ids': [],
                'tactic_ids': []
            }
        }


# Global scoring engine instance
scoring_engine = EntityScoringEngine()
