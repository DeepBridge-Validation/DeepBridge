"""
Package containing pre-trained distilled models and their metadata.
This directory is used for model persistence and loading.
"""

from pathlib import Path

MODELS_DIR = Path(__file__).parent

__all__ = ["MODELS_DIR"]