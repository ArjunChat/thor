"""THOR Models: tokenizer, Transformer, and training pipeline."""

from .config import PRESETS, ModelConfig, get_preset
from .model import ThorModel
from .tokenizer import Tokenizer

__all__ = ["PRESETS", "ModelConfig", "ThorModel", "Tokenizer", "get_preset"]
__version__ = "0.1.0"
