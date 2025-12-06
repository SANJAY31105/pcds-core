"""
PCDS Enterprise - Explainable AI Module
Provides human-readable explanations for ML decisions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class FeatureImportanceExplainer:
    """Compute feature importance using gradient-like approximations"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def compute_importance(self, features: np.ndarray, baseline: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute feature importance relative to baseline"""
        if baseline is None:
            baseline = np.zeros_like(features)
        
        # Absolute difference from baseline
        diff = np.abs(features - baseline)
        
        # Normalize to sum to 1
        total = np.sum(diff) + 1e-6
        importance = diff / total
        
        return {name: float(importance[i]) for i, name in enumerate(self.feature_names)}
    
    def get_top_features(self, importance: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k most important features"""
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_k]


class DecisionExplainer:
    """Generate human-readable explanations for decisions"""
    
    # Feature descriptions
    FEATURE_DESCRIPTIONS = {
        'packet_size_norm': 'Packet size',
        'port_norm': 'Network port',
        'protocol_enc': 'Protocol type',
        'ip_entropy': 'IP randomness',
        'bytes_in_norm': 'Incoming data volume',
        'bytes_out_norm': 'Outgoing data volume',
        'packet_count': 'Number of packets',
        'connection_duration': 'Connection length',
        'hour_of_day': 'Time of day',
        'day_of_week': 'Day of week',
        'is_weekend': 'Weekend activity',
        'is_business_hours': 'During business hours',
        'time_since_last_event': 'Time gap from last event',
        'events_per_minute': 'Event frequency',
        'burst_indicator': 'Sudden activity burst',
        'session_duration': 'Session length',
        'login_frequency': 'Login rate',
        'failed_attempts': 'Failed login attempts',
        'access_anomaly': 'Unusual access pattern',
        'privilege_level': 'User privilege level',
        'resource_access_count': 'Resources accessed',
        'data_volume_ratio': 'Data transfer ratio',
        'new_destination': 'New destination contacted',
        'geo_anomaly': 'Geographic anomaly',
        'known_bad_ip': 'Known malicious IP',
        'port_scan_indicator': 'Port scanning behavior',
        'lateral_movement': 'Lateral movement pattern',
        'exfil_indicator': 'Data exfiltration signal',
        'c2_beacon_pattern': 'Command & control beacon',
        'encryption_anomaly': 'Unusual encryption',
        'dns_anomaly': 'Suspicious DNS activity',
        'mitre_technique_match': 'Matches known attack technique'
    }
    
    # Risk phrases
    RISK_PHRASES = {
        'critical': ['CRITICAL: Immediate action required', 'Severe threat detected'],
        'high': ['HIGH RISK: Investigation recommended', 'Significant anomaly found'],
        'medium': ['MEDIUM: Monitor closely', 'Potential threat indicator'],
        'low': ['LOW: Normal variation', 'Minor anomaly detected']
    }
    
    def explain_prediction(self, 
                          features: np.ndarray, 
                          feature_names: List[str],
                          score: float,
                          is_anomaly: bool,
                          model_contributions: Dict[str, float] = None) -> Dict:
        """Generate comprehensive explanation"""
        
        # Determine risk level
        if score > 0.8:
            risk_level = 'critical'
        elif score > 0.6:
            risk_level = 'high'
        elif score > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Get feature importance
        importance_explainer = FeatureImportanceExplainer(feature_names)
        importance = importance_explainer.compute_importance(features)
        top_features = importance_explainer.get_top_features(importance, top_k=5)
        
        # Generate reason strings
        reasons = []
        for feature_name, imp in top_features:
            if imp > 0.1:  # Only significant features
                desc = self.FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
                value = features[feature_names.index(feature_name)]
                
                if value > 0.7:
                    reasons.append(f"High {desc.lower()} detected ({value:.2f})")
                elif value > 0.4:
                    reasons.append(f"Elevated {desc.lower()} observed ({value:.2f})")
        
        # Build explanation
        explanation = {
            'verdict': 'ANOMALY DETECTED' if is_anomaly else 'NORMAL BEHAVIOR',
            'risk_level': risk_level,
            'risk_score': float(score),
            'summary': self.RISK_PHRASES[risk_level][0],
            'top_contributing_factors': [
                {
                    'feature': name,
                    'description': self.FEATURE_DESCRIPTIONS.get(name, name),
                    'importance': float(imp),
                    'value': float(features[feature_names.index(name)])
                }
                for name, imp in top_features
            ],
            'reasons': reasons[:3],  # Top 3 reasons
            'model_contributions': model_contributions or {},
            'recommended_actions': self._get_recommended_actions(risk_level, top_features)
        }
        
        return explanation
    
    def _get_recommended_actions(self, risk_level: str, top_features: List[Tuple[str, float]]) -> List[str]:
        """Get recommended actions based on detection"""
        actions = []
        
        if risk_level == 'critical':
            actions.append("🚨 Isolate affected systems immediately")
            actions.append("🔍 Initiate incident response investigation")
            actions.append("📞 Notify security team")
        elif risk_level == 'high':
            actions.append("🔍 Investigate within 1 hour")
            actions.append("📊 Review related entity activity")
        elif risk_level == 'medium':
            actions.append("👀 Monitor for escalation")
            actions.append("📝 Create investigation ticket")
        else:
            actions.append("✅ Continue monitoring")
        
        # Feature-specific actions
        feature_names = [f[0] for f in top_features]
        
        if 'lateral_movement' in feature_names:
            actions.append("🔗 Check network segmentation")
        if 'exfil_indicator' in feature_names:
            actions.append("💾 Review data transfer logs")
        if 'failed_attempts' in feature_names:
            actions.append("🔐 Review authentication logs")
        if 'c2_beacon_pattern' in feature_names:
            actions.append("🌐 Block suspicious outbound connections")
        
        return actions[:4]  # Limit to 4 actions


class AttackNarrativeGenerator:
    """Generate natural language attack narratives"""
    
    MITRE_NARRATIVES = {
        'T1059': 'Attacker executed commands via {technique}',
        'T1078': 'Valid credentials were used to access {target}',
        'T1021': 'Remote services were used to access {target}',
        'T1041': 'Data was exfiltrated via {method}',
        'T1071': 'Command and control communication detected via {protocol}',
        'T1110': 'Brute force attack attempted against {target}',
        'T1003': 'Credential dumping detected on {target}'
    }
    
    def generate_narrative(self, detection: Dict, entities: List[Dict] = None) -> str:
        """Generate attack narrative from detection"""
        narrative_parts = []
        
        # Opening based on severity
        severity = detection.get('severity', 'medium')
        if severity == 'critical':
            narrative_parts.append("🚨 CRITICAL SECURITY EVENT:")
        elif severity == 'high':
            narrative_parts.append("⚠️ HIGH SEVERITY ALERT:")
        else:
            narrative_parts.append("📊 SECURITY OBSERVATION:")
        
        # Describe the detection
        detection_type = detection.get('detection_type', 'Unknown')
        narrative_parts.append(f"A {detection_type.replace('_', ' ')} event was detected.")
        
        # Add entity context
        if entities:
            entity_names = [e.get('name', e.get('id', 'unknown')) for e in entities[:3]]
            narrative_parts.append(f"Affected entities: {', '.join(entity_names)}")
        
        # MITRE ATT&CK context
        technique_id = detection.get('technique_id', '')
        if technique_id and technique_id in self.MITRE_NARRATIVES:
            template = self.MITRE_NARRATIVES[technique_id]
            narrative_parts.append(
                template.format(
                    technique=detection.get('technique_name', 'unknown technique'),
                    target=entities[0].get('name', 'target system') if entities else 'target',
                    method='network channel',
                    protocol='HTTPS'
                )
            )
        
        # Timeline
        timestamp = detection.get('timestamp', detection.get('detected_at', 'unknown time'))
        narrative_parts.append(f"Detected at: {timestamp}")
        
        return " ".join(narrative_parts)
    
    def generate_attack_chain_narrative(self, chain: List[Dict]) -> str:
        """Generate narrative for attack chain"""
        if not chain:
            return "No attack chain detected."
        
        narrative = ["🔗 ATTACK CHAIN DETECTED:\n"]
        
        for i, step in enumerate(chain):
            narrative.append(f"  Step {i+1}: {step.get('action', 'Unknown action')} on {step.get('target', 'unknown target')}")
        
        narrative.append(f"\nTotal chain length: {len(chain)} steps")
        narrative.append("Recommendation: Investigate the initial infection vector and contain spread.")
        
        return "\n".join(narrative)


class ExplainableAI:
    """Main explainability interface"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.decision_explainer = DecisionExplainer()
        self.narrative_generator = AttackNarrativeGenerator()
    
    def explain(self, 
                features: np.ndarray,
                score: float,
                is_anomaly: bool,
                detection: Dict = None,
                entities: List[Dict] = None,
                model_contributions: Dict[str, float] = None) -> Dict:
        """Generate complete explanation"""
        
        # Base explanation
        explanation = self.decision_explainer.explain_prediction(
            features, self.feature_names, score, is_anomaly, model_contributions
        )
        
        # Add narrative if detection provided
        if detection:
            explanation['narrative'] = self.narrative_generator.generate_narrative(
                detection, entities
            )
        
        # Add MITRE context if available
        if detection and detection.get('technique_id'):
            explanation['mitre_context'] = {
                'technique_id': detection.get('technique_id'),
                'technique_name': detection.get('technique_name', 'Unknown'),
                'tactic': detection.get('tactic', 'Unknown'),
                'reference': f"https://attack.mitre.org/techniques/{detection.get('technique_id')}"
            }
        
        return explanation


# Feature names for global instance
DEFAULT_FEATURE_NAMES = [
    'packet_size_norm', 'port_norm', 'protocol_enc', 'ip_entropy',
    'bytes_in_norm', 'bytes_out_norm', 'packet_count', 'connection_duration',
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours',
    'time_since_last_event', 'events_per_minute', 'burst_indicator', 'session_duration',
    'login_frequency', 'failed_attempts', 'access_anomaly', 'privilege_level',
    'resource_access_count', 'data_volume_ratio', 'new_destination', 'geo_anomaly',
    'known_bad_ip', 'port_scan_indicator', 'lateral_movement', 'exfil_indicator',
    'c2_beacon_pattern', 'encryption_anomaly', 'dns_anomaly', 'mitre_technique_match'
]

explainable_ai = ExplainableAI(DEFAULT_FEATURE_NAMES)
