from speculators.models.attention import ALL_ATTENTION_FUNCTIONS  # noqa: F401

from .dflash import DFlashDraftModel, DFlashSpeculatorConfig
from .dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from .dspark import DSparkDraftModel, DSparkSpeculatorConfig
from .eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from .leagle import LEagleDraftModel, LEagleSpeculatorConfig
from .mmuse import MMuseDraftModel, MMuseSpeculatorConfig
from .mtp import MTPDraftModel, MTPSpeculatorConfig
from .peagle import PEagleDraftModel, PEagleSpeculatorConfig

__all__ = [
    "DFlash2DraftModel",
    "DFlash2SpeculatorConfig",
    "DFlashDraftModel",
    "DFlashSpeculatorConfig",
    "DSparkDraftModel",
    "DSparkSpeculatorConfig",
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
    "LEagleDraftModel",
    "LEagleSpeculatorConfig",
    "MMuseDraftModel",
    "MMuseSpeculatorConfig",
    "MTPDraftModel",
    "MTPSpeculatorConfig",
    "PEagleDraftModel",
    "PEagleSpeculatorConfig",
]
