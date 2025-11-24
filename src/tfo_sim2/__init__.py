"""
TFO Sim2 - A comprehensive wrapper around PMCX for photon transport simulations.

This package provides high-level abstractions for:
- Tissue model definitions
- Simulation parameter management
- Detector positioning
- Running simulations
- Result storage and analysis
- Visualization with predefined and custom plots
- Batch experiment execution
"""

from .tissue_model import *  # noqa: F401, F403
from .simulation_params import *  # noqa: F401, F403
from .detectors import *  # noqa: F401, F403
from .simulator import *  # noqa: F401, F403
from .result_storage import *  # noqa: F401, F403
from .plotter import *  # noqa: F401, F403
from .experiment_handler import *  # noqa: F401, F403
from .tissue_model_extended import *  # noqa: F401, F403

__version__ = "0.1.0"
