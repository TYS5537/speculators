"""MUSE: experimental backbone, Selector, and Correction architecture."""

from speculators.models.muse.config import MuseSpeculatorConfig
from speculators.models.muse.core import MuseDraftModel

__all__ = ["MuseDraftModel", "MuseSpeculatorConfig"]
