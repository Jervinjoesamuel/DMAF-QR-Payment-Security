"""
Deep Multimodal Authentication Framework (DMAF)
Implementation for Malicious QR Code Detection in Mobile Payment Systems.
"""

from dmaf.config import ModelConfig, TrainingConfig, DatasetConfig
from dmaf.models.dmaf_network import DMAFNetwork
from dmaf.explainability import ExplainableRiskAssessor
from dmaf.continual_learning import ContinualLearningEngine
from dmaf.preprocessing import ImagePreprocessor, TextTokenizer, TabularStandardizer
from dmaf.dataset import QRMultimodalDataset

__version__ = "1.0.0"

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DatasetConfig",
    "DMAFNetwork",
    "ExplainableRiskAssessor",
    "ContinualLearningEngine",
    "ImagePreprocessor",
    "TextTokenizer",
    "TabularStandardizer",
    "QRMultimodalDataset",
]
