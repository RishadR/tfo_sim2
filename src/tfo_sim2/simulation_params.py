"""
Simulation Parameters for PMCX simulations.

This module provides a class to manage all simulation settings that get passed
to the PMCX cfg dictionary.
"""

from typing import Dict, List, Optional, Any, Tuple, Literal
from dataclasses import dataclass, field, asdict


@dataclass
class SimulationParameters:
    """
    Container for PMCX simulation parameters.

    This class holds all the settings needed to configure a PMCX simulation,
    and provides methods to generate or update PMCX cfg dictionaries.
    """

    # Simulation Settings
    session: str = "pmcx_simulation"
    """Prefix of output file (Session)."""

    isreflect: int = 0
    """Boundary condition: 0 for no reflection back (photons die when crossing boundary), 1 for reflection."""

    seed: int = field(default_factory=lambda: 123456789)
    """Random seed for reproducibility."""

    # Photon settings
    nphoton: int = 1000000
    """Number of photons to simulate."""

    # Timing
    tstart: float = 0.0
    """Start time for time-resolved simulation (seconds)."""

    tend: float = 5e-9
    """End time for time-resolved simulation (seconds)."""

    tstep: float = 5e-9
    """Time step for time-resolved data collection (seconds)."""

    # Source properties
    srcpos: List[float] = field(default_factory=lambda: [30, 30, 0])
    """Source position [x, y, z]."""

    srcdir: List[float] = field(default_factory=lambda: [0, 0, 1])
    """Source direction [dx, dy, dz]."""

    # Detector settings
    issavedet: int = 1
    """Whether to save detected photon data."""
    
    maxdetphoton: int = 100000000
    """Maximum number of detected photons to save."""

    savedetflag: str = "dpx"
    """Flags indicating what detected photon data to save:
    d: detected photon ID
    p: partial path
    x: exit position
    s: scattering count
    v: direction vector
    m: momentum transfer
    w: initial weight
    """

    # Coordinate system
    issrcfrom0: int = 1
    """Whether source/detector coordinates start from 0 (not 1)."""

    # Advanced options
    issaveseed: int = 0
    """Whether to save random seeds for photon replay."""

    issaveref: int = 0
    """Whether to save diffuse reflectance."""

    debug: str = ""
    """Debug output flags."""
    
    unitinmm: float = 1.0
    """Unit conversion factor to mm."""

    # GPU and execution
    autopilot: int = 1
    """Whether to use autopilot mode."""

    gpuid: int = 1
    """GPU device ID."""

    # Output type
    outputtype: Literal["fluence", "flux"] = "fluence"
    """Output type: 'fluence', 'jacobian', etc."""

    # Additional optional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    """Additional PMCX parameters not covered by the main fields."""

    def to_cfg(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert parameters to PMCX cfg dictionary.

        Args:
            cfg: Optional existing cfg dictionary to update. If None, creates new.

        Returns:
            The updated/created cfg dictionary.
        """
        if cfg is None:
            cfg = {}

        # Add all dataclass fields as cfg entries
        for key, value in asdict(self).items():
            if key != "extra_params" and value is not None:
                cfg[key] = value

        # Add any extra parameters
        cfg.update(self.extra_params)

        return cfg

    def update_from_cfg(self, cfg: Dict[str, Any]) -> None:
        """
        Update parameters from a cfg dictionary.

        Args:
            cfg: PMCX configuration dictionary.
        """
        for key, value in cfg.items():
            if hasattr(self, key) and key != "extra_params":
                setattr(self, key, value)
            else:
                self.extra_params[key] = value

    def __repr__(self) -> str:
        params = asdict(self)
        # Only show non-default values for brevity
        key_params = {
            "nphoton": self.nphoton,
            "tend": self.tend,
            "srcpos": self.srcpos,
            "srcdir": self.srcdir,
            "outputtype": self.outputtype,
        }
        return f"SimulationParameters({', '.join(f'{k}={v}' for k, v in key_params.items())})"


__all__ = [
    "SimulationParameters",
]
