"""Engine module for PCDS Enterprise"""

from .scoring import EntityScoringEngine, scoring_engine
from .mitre_mapper import MITREMapper, mitre_mapper, enrich_detection_with_mitre
from .campaign_correlator import CampaignCorrelator, campaign_correlator
from .detection_engine import ReconnaissanceDetector, CredentialTheftDetector
from .detection_modules import (
    LateralMovementDetector,
    PrivilegeEscalationDetector,
    CommandAndControlDetector,
    DataExfiltrationDetector
)

__all__ = [
    'EntityScoringEngine',
    'scoring_engine',
    'MITREMapper',
    'mitre_mapper',
    'enrich_detection_with_mitre',
    'CampaignCorrelator',
    'campaign_correlator',
    'ReconnaissanceDetector',
    'CredentialTheftDetector',
    'LateralMovementDetector',
    'PrivilegeEscalationDetector',
    'CommandAndControlDetector',
    'DataExfiltrationDetector'
]

