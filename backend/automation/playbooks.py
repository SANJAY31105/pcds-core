import logging
from automation.actions import ResponseActions

logger = logging.getLogger("pcds.automation")

class PlaybookEngine:
    """
    Evaluates detections against playbook rules and triggers actions.
    """
    
    def __init__(self):
        self.playbooks = [
            {
                "name": "Ransomware Containment",
                "trigger": lambda d: d.get('severity') == 'critical' and 'ransomware' in (d.get('title', '').lower() + d.get('description', '').lower()),
                "action": "isolate_host"
            },
            {
                "name": "Critical Threat Isolation",
                "trigger": lambda d: d.get('severity') == 'critical' and d.get('confidence_score', 0) > 0.9,
                "action": "isolate_host"
            }
        ]
    
    def evaluate(self, detection: dict):
        """Evaluate a detection against all playbooks"""
        for playbook in self.playbooks:
            try:
                if playbook['trigger'](detection):
                    logger.info(f"⚡ Triggering Playbook: {playbook['name']} for detection {detection['id']}")
                    self.execute_playbook(playbook, detection)
            except Exception as e:
                logger.error(f"Error evaluating playbook {playbook['name']}: {e}")

    def execute_playbook(self, playbook: dict, detection: dict):
        """Execute the action defined in the playbook"""
        action_name = playbook['action']
        entity_id = detection.get('entity_id')
        
        if not entity_id:
            logger.warning("Playbook triggered but no entity_id found.")
            return

        if action_name == "isolate_host":
            ResponseActions.isolate_host(entity_id, reason=f"Triggered by {playbook['name']}")

# Global instance
playbook_engine = PlaybookEngine()
