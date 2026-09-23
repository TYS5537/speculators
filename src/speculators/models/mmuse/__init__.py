"""MMUSE: experimental backbone, Selector, and Correction architecture."""

from speculators.models.mmuse.config import MMuseSpeculatorConfig
from speculators.models.mmuse.core import MMuseDraftModel

__all__ = ["MMuseDraftModel", "MMuseSpeculatorConfig"]
