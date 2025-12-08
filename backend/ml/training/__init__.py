"""
ML Training Pipeline Package

Modules:
- dataset_loaders: Load CIC-IDS 2017, UNSW-NB15, CTU-13 datasets
- feature_extraction: Extract flow, behavioral, attack context features
- attack_classifiers: Per-attack-category classifier models
- training_pipeline: Complete end-to-end training pipeline

Usage:
    from ml.training import TrainingPipeline
    
    pipeline = TrainingPipeline()
    pipeline.run_full_pipeline(['cicids2017'], epochs=50)
"""

from .dataset_loaders import (
    CICIDS2017Loader,
    UNSWNB15Loader,
    CTU13Loader,
    UnifiedDatasetLoader,
    load_dataset
)

from .feature_extraction import (
    FeatureExtractor,
    AttackFeatureExtractor,
    create_feature_extractor
)

from .attack_classifiers import (
    BaseAttackClassifier,
    AttackCategoryManager,
    create_attack_classifier,
    create_category_manager
)

from .training_pipeline import (
    TrainingPipeline,
    train_from_dataset,
    train_all_datasets
)

__all__ = [
    # Dataset loaders
    'CICIDS2017Loader',
    'UNSWNB15Loader',
    'CTU13Loader',
    'UnifiedDatasetLoader',
    'load_dataset',
    
    # Feature extraction
    'FeatureExtractor',
    'AttackFeatureExtractor',
    'create_feature_extractor',
    
    # Classifiers
    'BaseAttackClassifier',
    'AttackCategoryManager',
    'create_attack_classifier',
    'create_category_manager',
    
    # Pipeline
    'TrainingPipeline',
    'train_from_dataset',
    'train_all_datasets'
]
