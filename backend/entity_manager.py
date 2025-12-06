"""Entity Management and Scoring System"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
import random

class UrgencyLevel(str, Enum):
    """Entity urgency levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class EntityType(str, Enum):
    """Types of entities"""
    HOST = "host"
    USER = "user"
    SERVICE = "service"
    NETWORK = "network"

class EntityManager:
    """Manages entities and calculates urgency scores"""
    
    def __init__(self):
        self.entities: Dict[str, Dict] = {}
        self.entity_counter = 0
    
    def create_or_update_entity(
        self,
        entity_type: EntityType,
        identifier: str,
        metadata: Dict = None
    ) -> Dict:
        """Create or update an entity"""
        entity_id = f"{entity_type.value}_{identifier}"
        
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            entity['last_seen'] = datetime.now()
            if metadata:
                entity['metadata'].update(metadata)
        else:
            self.entity_counter += 1
            entity = {
                'id': entity_id,
                'entity_number': self.entity_counter,
                'type': entity_type.value,
                'identifier': identifier,
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'detections': [],
                'risk_score': 0,
                'urgency_level': UrgencyLevel.LOW.value,
                'metadata': metadata or {},
                'baseline': {},
                'is_whitelisted': False
            }
            self.entities[entity_id] = entity
        
        return entity
    
    def add_detection(self, entity_id: str, detection: Dict):
        """Add a detection to an entity and recalculate score"""
        if entity_id not in self.entities:
            return
        
        entity = self.entities[entity_id]
        entity['detections'].append({
            **detection,
            'timestamp': datetime.now()
        })
        
        # Recalculate urgency score
        self.calculate_urgency_score(entity_id)
    
    def calculate_urgency_score(self, entity_id: str) -> int:
        """Calculate urgency score based on multiple factors"""
        entity = self.entities.get(entity_id)
        if not entity or entity.get('is_whitelisted'):
            return 0
        
        detections = entity.get('detections', [])
        if not detections:
            entity['risk_score'] = 0
            entity['urgency_level'] = UrgencyLevel.LOW.value
            return 0
        
        # Factor 1: Detection count (0-30 points)
        detection_count = len(detections)
        count_score = min(detection_count * 3, 30)
        
        # Factor 2: Severity (0-40 points)
        severity_weights = {'critical': 40, 'high': 25, 'medium': 10, 'low': 5}
        severity_score = max([
            severity_weights.get(d.get('severity', 'low'), 5) 
            for d in detections
        ])
        
        # Factor 3: Recency (0-20 points)
        now = datetime.now()
        most_recent = max([d.get('timestamp', now) for d in detections])
        hours_ago = (now - most_recent).total_seconds() / 3600
        if hours_ago < 1:
            recency_score = 20
        elif hours_ago < 24:
            recency_score = 15
        elif hours_ago < 168:  # 1 week
            recency_score = 10
        else:
            recency_score = 5
        
        # Factor 4: Attack progression (0-10 points)
        # Check if detections span multiple kill chain stages
        unique_tactics = set([
            d.get('mitre', {}).get('tactic_id', '') 
            for d in detections
        ])
        progression_score = min(len(unique_tactics) * 2, 10)
        
        # Calculate total score (0-100)
        total_score = count_score + severity_score + recency_score + progression_score
        entity['risk_score'] = min(total_score, 100)
        
        # Determine urgency level
        if total_score >= 75:
            entity['urgency_level'] = UrgencyLevel.CRITICAL.value
        elif total_score >= 50:
            entity['urgency_level'] = UrgencyLevel.HIGH.value
        elif total_score >= 25:
            entity['urgency_level'] = UrgencyLevel.MEDIUM.value
        else:
            entity['urgency_level'] = UrgencyLevel.LOW.value
        
        return entity['risk_score']
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get entity by ID"""
        return self.entities.get(entity_id)
    
    def get_all_entities(
        self,
        urgency_filter: Optional[str] = None,
        entity_type_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get entities with optional filtering"""
        entities = list(self.entities.values())
        
        # Filter by urgency
        if urgency_filter:
            entities = [e for e in entities if e['urgency_level'] == urgency_filter]
        
        # Filter by type
        if entity_type_filter:
            entities = [e for e in entities if e['type'] == entity_type_filter]
        
        # Sort by risk score (highest first)
        entities.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return entities[:limit]
    
    def get_entity_timeline(self, entity_id: str) -> List[Dict]:
        """Get chronological timeline of detections for an entity"""
        entity = self.entities.get(entity_id)
        if not entity:
            return []
        
        detections = entity.get('detections', [])
        # Sort by timestamp and kill chain stage
        sorted_detections = sorted(
            detections,
            key=lambda x: (
                x.get('timestamp', datetime.now()),
                x.get('mitre', {}).get('kill_chain_stage', 0)
            )
        )
        
        return sorted_detections
    
    def get_attack_graph(self, entity_id: str) -> Dict:
        """Build attack graph showing entity relationships and attack flow"""
        entity = self.entities.get(entity_id)
        if not entity:
            return {'nodes': [], 'edges': []}
        
        nodes = [{'id': entity_id, 'label': entity['identifier'], 'type': entity['type'], 'risk': entity['risk_score']}]
        edges = []
        
        # Add related entities (simplified - would be more complex in production)
        for detection in entity.get('detections', []):
            if 'source_ip' in detection.get('metadata', {}):
                source_id = f"host_{detection['metadata']['source_ip']}"
                if source_id != entity_id:
                    nodes.append({'id': source_id, 'label': detection['metadata']['source_ip'], 'type': 'host', 'risk': 30})
                    edges.append({
                        'source': source_id,
                        'target': entity_id,
                        'label': detection.get('mitre', {}).get('technique_name', detection.get('type', '')),
                        'timestamp': detection.get('timestamp', datetime.now()).isoformat()
                    })
        
        return {'nodes': nodes, 'edges': edges}
    
    def get_statistics(self) -> Dict:
        """Get entity statistics"""
        total = len(self.entities)
        critical = sum(1 for e in self.entities.values() if e['urgency_level'] == UrgencyLevel.CRITICAL.value)
        high = sum(1 for e in self.entities.values() if e['urgency_level'] == UrgencyLevel.HIGH.value)
        medium = sum(1 for e in self.entities.values() if e['urgency_level'] == UrgencyLevel.MEDIUM.value)
        low = sum(1 for e in self.entities.values() if e['urgency_level'] == UrgencyLevel.LOW.value)
        
        return {
            'total_entities': total,
            'critical': critical,
            'high': high,
            'medium': medium,
            'low': low,
            'distribution': {
                'critical': round(critical / total * 100, 1) if total > 0 else 0,
                'high': round(high / total * 100, 1) if total > 0 else 0,
                'medium': round(medium / total * 100, 1) if total > 0 else 0,
                'low': round(low / total * 100, 1) if total > 0 else 0
            }
        }

# Global instance
entity_manager = EntityManager()
