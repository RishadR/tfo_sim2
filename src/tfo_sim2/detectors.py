"""
Detector management for PMCX simulations.

This module provides classes to define and manage detector arrays for photon
detection simulations.
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class Detector:
    """
    A single point detector for photon detection.

    Attributes:
        position: (x, y, z) position in voxel coordinates.
        radius: Detection radius in voxels.
        id: Optional unique identifier for the detector.
    """

    position: Tuple[float, float, float]
    radius: float = 1.0
    id: int = 0

    def to_list(self) -> List[float]:
        """Convert to PMCX format [x, y, z, radius]."""
        return [self.position[0], self.position[1], self.position[2], self.radius]

    def __repr__(self) -> str:
        return f"Detector(pos={self.position}, r={self.radius}, id={self.id})"


class DetectorArray:
    """
    Manages a collection of detectors for a simulation.
    """

    def __init__(self, name: str = "DetectorArray"):
        """
        Initialize an empty detector array.

        Args:
            name: Descriptive name for the detector array.
        """
        self.name = name
        self.detectors: List[Detector] = []
        self._next_id = 1

    def add_detector(
        self,
        position: Tuple[float, float, float],
        radius: float = 1.0,
    ) -> int:
        """
        Add a detector to the array.

        Args:
            position: (x, y, z) position in voxel coordinates.
            radius: Detection radius in voxels.

        Returns:
            The ID assigned to this detector.
        """
        detector_id = self._next_id
        self.detectors.append(Detector(position, radius, detector_id))
        self._next_id += 1
        return detector_id

    def add_detectors_at_positions(
        self,
        positions: List[Tuple[float, float, float]],
        radius: float = 1.0,
    ) -> List[int]:
        """
        Add multiple detectors at specified positions.

        Args:
            positions: List of (x, y, z) positions.
            radius: Detection radius for all detectors.

        Returns:
            List of detector IDs.
        """
        ids = []
        for pos in positions:
            ids.append(self.add_detector(pos, radius))
        return ids

    def add_detector_line(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        num_detectors: int,
        radius: float = 1.0,
    ) -> List[int]:
        """
        Add detectors along a line between two points using linspace. The end point
        is included!

        Args:
            start: Starting position.
            end: Ending position. (Non-inclusive)
            num_detectors: Number of detectors to place.
            radius: Detection radius for all detectors.

        Returns:
            List of detector IDs.
        """
        start_arr = np.array(start, dtype=float)
        end_arr = np.array(end, dtype=float)
        positions = np.linspace(start_arr, end_arr, num_detectors, endpoint=True)

        ids = []
        for pos in positions:
            ids.append(self.add_detector(tuple(pos), radius))
        return ids

    def add_detector_grid(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z: float,
        nx: int,
        ny: int,
        radius: float = 1.0,
    ) -> List[int]:
        """
        Add detectors in a rectangular grid pattern.

        Args:
            x_range: (x_min, x_max) range.
            y_range: (y_min, y_max) range.
            z: Fixed z coordinate.
            nx: Number of detectors in x direction.
            ny: Number of detectors in y direction.
            radius: Detection radius for all detectors.

        Returns:
            List of detector IDs.
        """
        x_positions = np.linspace(x_range[0], x_range[1], nx)
        y_positions = np.linspace(y_range[0], y_range[1], ny)

        ids = []
        for x in x_positions:
            for y in y_positions:
                ids.append(self.add_detector((x, y, z), radius))
        return ids

    def add_detector_circle(
        self,
        center: Tuple[float, float, float],
        plane_normal: Tuple[float, float, float],
        radius_circle: float,
        num_detectors: int,
        detector_radius: float = 1.0,
    ) -> List[int]:
        """
        Add detectors in a circular pattern.

        Args:
            center: Center of the circle.
            plane_normal: Normal vector to the plane of the circle.
            radius_circle: Radius of the circle pattern.
            num_detectors: Number of detectors around the circle.
            detector_radius: Detection radius for each detector.

        Returns:
            List of detector IDs.
        """
        center_arr = np.array(center, dtype=float)
        normal = np.array(plane_normal, dtype=float)
        normal = normal / np.linalg.norm(normal)

        # Create orthogonal basis vectors
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        # Generate points on the circle
        angles = np.linspace(0, 2 * np.pi, num_detectors, endpoint=False)

        ids = []
        for angle in angles:
            pos = center_arr + radius_circle * (np.cos(angle) * u + np.sin(angle) * v)
            ids.append(self.add_detector(tuple(pos), detector_radius))
        return ids

    def to_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update PMCX cfg dict with detector positions and return it

        Args:
            cfg: The PMCX configuration dictionary to update.
        """
        if len(self.detectors) == 0:
            cfg["issavedet"] = 0
        else:
            cfg["detpos"] = [det.to_list() for det in self.detectors]
            cfg["issavedet"] = 1
        return cfg

    def get_detector(self, detector_id: int) -> Detector:
        """Get a detector by ID."""
        return self.detectors[detector_id]

    def get_position(self, detector_id: int) -> Tuple[float, float, float]:
        """Get the position of a detector."""
        return self.detectors[detector_id].position

    def __len__(self) -> int:
        """Return the number of detectors."""
        return len(self.detectors)
    
    def clear(self) -> None:
        """Remove all detectors from the array."""
        self.detectors.clear()
        self._next_id = 1

    def create_copy(self) -> "DetectorArray":
        """Create a deep copy of this DetectorArray."""
        new_array = DetectorArray(name=self.name)
        for det in self.detectors:
            new_array.add_detector(det.position, det.radius)
        return new_array

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', num_detectors={len(self.detectors)})"


__all__ = [
    "Detector",
    "DetectorArray",
]
