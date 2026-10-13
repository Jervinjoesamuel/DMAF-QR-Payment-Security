import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    embed_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    num_classes: int = 2  # Binary (Benign vs Malicious) or 6 for detailed threat types
    
    # Vision encoder (Lightweight ViT)
    img_size: int = 128
    patch_size: int = 16
    in_channels: int = 3
    vit_depth: int = 4
    
    # Text & URL encoders
    vocab_size: int = 5000
    max_text_len: int = 128
    max_url_len: int = 128
    text_hidden_dim: int = 128
    
    # Tabular encoders (Metadata & Behavioral)
    meta_feature_dim: int = 16
    behavior_seq_len: int = 10
    behavior_feature_dim: int = 8
    
    # Classification head
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 15
    num_workers: int = 4
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    tau_threshold: float = 0.85
    continual_lr: float = 5e-5


@dataclass
class DatasetConfig:
    data_dir: str = "data"
    image_size: int = 128
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
