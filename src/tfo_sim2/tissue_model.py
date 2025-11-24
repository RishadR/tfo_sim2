"""
Tissue Model abstractions for PMCX simulations.

This module provides different tissue model implementations that can be converted
into PMCX-compatible cfg parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class TissueModel(ABC):
    """
    Abstract base class for tissue models.

    Each tissue model represents a heterogeneous medium that can be used in
    PMCX simulations. Subclasses must implement the methods to generate the
    volume (voxel) data and optical properties required by PMCX.
    
    Treat TissueModel as a DataClass with two private functions - _generate_volume and _generate_properties.
    Each time the model gets simulated, these two functions will be called immedietely before the simulation. 
    If you want to sweep over variables, make those class variables. If you want some dynamic action, either define
    them with @property or set them in the _generate_volume or _generate_properties
    
    """

    def __init__(self, name: str = "TissueModel"):
        """
        Initialize the tissue model.

        Args:
            name: Descriptive name for the tissue model.
        """
        self.name = name
        self._vol = None
        self._prop = None

    @property
    def vol(self) -> np.ndarray:
        """Get the volume (voxel) data."""
        if self._vol is None:
            self._generate_volume()
        if not isinstance(self._vol, np.ndarray):
            raise ValueError("Volume data must be a numpy ndarray.")
        return self._vol

    @property
    def prop(self) -> List[List]:
        """Get the optical properties."""
        if self._prop is None:
            self._generate_properties()
        if not isinstance(self._prop, list):
            raise ValueError("Optical properties must be a list of lists.")
        return self._prop

    @abstractmethod
    def _generate_volume(self):
        """
        Generate the volume data. Must set self._vol.

        The volume should be a 3D numpy array where each voxel contains a tag
        referring to the tissue type at that location.
        """
        pass

    @abstractmethod
    def _generate_properties(self):
        """
        Generate optical properties. Must set self._prop.

        Properties should be a list of lists where each row contains:
        [mua, mus, g, n] for each tissue type.
        mua: absorption coefficient
        mus: scattering coefficient
        g: anisotropy
        n: refractive index
        """
        pass

    def to_cfg(self, cfg: Dict[str, Any]) -> None:
        """
        Update PMCX cfg dict with this tissue model's parameters.

        Args:
            cfg: The PMCX configuration dictionary to update.
        """
        cfg["vol"] = self.vol.copy()
        cfg["prop"] = self.prop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class UniformTissueModel(TissueModel):
    """
    A simple uniform tissue model with a single tissue type throughout.
    """

    def __init__(
        self,
        size: Tuple[int, int, int],
        mua: float,
        mus: float,
        g: float,
        n: float,
        name: str = "UniformTissue",
    ):
        """
        Initialize a uniform tissue model with an air layer on top (Z = size(2) to size(2)+2).

        Args:
            size: (x, y, z) dimensions of the volume in voxels.
            mua: Absorption coefficient.
            mus: Scattering coefficient.
            g: Anisotropy.
            n: Refractive index.
            name: Descriptive name.
        """
        super().__init__(name)
        self.size = size
        self.optical_props = [mua, mus, g, n]

    def _generate_volume(self):
        """Generate a uniform volume with tissue tag 1."""
        vol = np.ones(self.size, dtype="uint8")
        air_vol = np.zeros((self.size[0], self.size[1], 2), dtype="uint8")
        self._vol = np.concatenate((vol, air_vol), axis=2)

    def _generate_properties(self):
        """
        Generate properties for two media:
        - Medium 0 (background): non-scattering, non-absorbing
        - Medium 1 (tissue): the specified optical properties
        """
        self._prop = [
            [0, 0, 1, 1],  # background
            self.optical_props,  # tissue
        ]


class LayeredTissueModel(TissueModel):
    """
    A tissue model with horizontal layers of different optical properties.
    """

    def __init__(
        self,
        size: Tuple[int, int, int],
        layers: List[Dict[str, Any]],
        name: str = "LayeredTissue",
    ):
        """
        Initialize a layered tissue model with an air layer on top (Z = size(2) to size(2)+2).

        Args:
            size: (x, y, z) dimensions of the volume in voxels.
            layers: List of layer definitions, each with:
                - 'z_start': Starting z-coordinate (in voxels)
                - 'z_end': Ending z-coordinate (in voxels)
                - 'mua': Absorption coefficient
                - 'mus': Scattering coefficient
                - 'g': Anisotropy
                - 'n': Refractive index
                - 'tag': Tissue tag (optional, auto-assigned if not provided)
            name: Descriptive name.
        """
        super().__init__(name)
        self.size = size
        self.layers = layers
        self._validate_layers()

    def _validate_layers(self):
        """Validate layer definitions."""
        for i, layer in enumerate(self.layers):
            required_keys = {"z_start", "z_end", "mua", "mus", "g", "n"}
            if not required_keys.issubset(layer.keys()):
                raise ValueError(
                    f"Layer {i} missing required keys: {required_keys}"
                )
            if layer["z_start"] >= layer["z_end"]:
                raise ValueError(f"Layer {i}: z_start must be < z_end")
            if "tag" not in layer:
                layer["tag"] = i + 1  # Auto-assign tags starting from 1

    def _generate_volume(self):
        """Generate volume with layered structure."""
        self._vol = np.zeros(self.size, dtype="uint8")
        for layer in self.layers:
            z_start = layer["z_start"]
            z_end = layer["z_end"]
            tag = layer["tag"]
            self._vol[:, :, z_start:z_end] = tag

        air_vol = np.zeros((self.size[0], self.size[1], 2), dtype="uint8")
        self._vol = np.concatenate((self._vol, air_vol), axis=2)

    def _generate_properties(self):
        """Generate properties for each layer plus background."""
        self._prop = [[0, 0, 1, 1]]  # background
        for layer in self.layers:
            self._prop.append(
                [
                    layer["mua"],
                    layer["mus"],
                    layer["g"],
                    layer["n"],
                ]
            )

__all__ = [
    "TissueModel",
    "UniformTissueModel",
    "LayeredTissueModel",
]
