"""
Main simulator that orchestrates PMCX simulations.

This module provides the Simulator class which combines tissue models,
simulation parameters, and detectors to run PMCX simulations.
"""

from typing import Dict, Optional, Any
import warnings
import pmcx


from .tissue_model import TissueModel
from .simulation_params import SimulationParameters
from .detectors import DetectorArray


class Simulator:
    """
    Orchestrates PMCX photon transport simulations.

    This class combines tissue models, simulation parameters, and detector
    arrays to run complete simulations.
    
    Simulator Run Sequence:
    -----------------------
    Build New CFG -> Simulate -> Store Result -> Error Check -> Store Src/Det Pos
    """

    def __init__(
        self,
        tissue_model: TissueModel,
        simulation_params: SimulationParameters,
        detectors: Optional[DetectorArray] = None,
    ):
        """
        Initialize the simulator.

        Args:
            tissue_model: TissueModel instance defining the medium.
            simulation_params: SimulationParameters instance with simulation settings.
            detectors: Optional DetectorArray instance with detector positions.
        """
        self.tissue_model = tissue_model
        self.simulation_params = simulation_params
        self.detectors = detectors or DetectorArray()
        self._cfg = self.build_cfg()
        self._last_result = None

    def build_cfg(self) -> Dict[str, Any]:
        """
        Build the PMCX configuration dictionary.

        Combines tissue model, simulation parameters, and detector settings
        into a single cfg dict suitable for pmcx.mcxlab().
        
        Returns:
            The complete PMCX configuration dictionary.
        """
        # Start with simulation parameters
        self._cfg = self.simulation_params.to_cfg()

        # Add tissue model
        self.tissue_model.to_cfg(self._cfg)

        # Add detectors
        self.detectors.to_cfg(self._cfg)

        return self._cfg

    def run(self) -> Dict:
        """
        Run the simulation.

        Returns:
            The simulation results as a dictionary with keys like 'flux',
            'detp', 'stat', etc.

        Raises:
            ImportError: If PMCX is not installed.
        """
        # Rebuild cfg to ensure it's up to date
        cfg = self.build_cfg()

        # Run the simulation
        self._last_result = pmcx.mcxlab(cfg)

        if not isinstance(self._last_result, dict):
            raise RuntimeError("PMCX did not return a valid result dictionary!")

        # Add detector and source positions to result if available
        self._last_result['detpos'] = cfg.get('detpos', None)
        self._last_result['srcpos'] = cfg.get('srcpos', None)  # Add source positions to result
        return self._last_result

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the last simulation result without re-running.

        Returns:
            The last result dictionary, or None if no simulation has been run.
        """
        return self._last_result

    def get_cfg(self) -> Dict[str, Any]:
        """
        Get the current PMCX configuration.

        Returns:
            The built configuration dictionary.
        """
        if self._cfg is None:
            self.build_cfg()
        assert self._cfg is not None
        return self._cfg

    def set_tissue_model(self, tissue_model: TissueModel) -> None:
        """Change the tissue model."""
        self.tissue_model = tissue_model
        self._cfg = self.build_cfg()  # Rebuild cfg

    def set_simulation_params(self, simulation_params: SimulationParameters) -> None:
        """Change the simulation parameters."""
        self.simulation_params = simulation_params
        self._cfg = self.build_cfg()  # Rebuild cfg

    def set_detectors(self, detectors: DetectorArray) -> None:
        """Change the detector array."""
        self.detectors = detectors
        self._cfg = self.build_cfg()  # Rebuild cfg

    def __repr__(self) -> str:
        return (
            f"Simulator(\n"
            f"  tissue={self.tissue_model},\n"
            f"  params={self.simulation_params},\n"
            f"  detectors={self.detectors}\n"
            f")"
        )


__all__ = [
    "Simulator",
]
