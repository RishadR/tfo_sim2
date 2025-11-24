"""
Visualization and plotting for simulation results.

This module provides the Plotter class with predefined plot types and support
for custom plotting functions.
"""

from enum import Enum
from typing import Callable, Dict, Any, Optional, List, Tuple
import warnings
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .result_storage import ResultStorage


default_colormap = "viridis"


class Plotter(ABC):
    """
    Base class for creating plots from simulation results.

    Required Methods:
    - plot(axes: Axes, **kwargs) -> None: Create the plot on the given axes. Additional parameters can be passed on
        to the plot command via kwargs.
    - check_valid_data() -> bool: Check if required data for plotting is available.

    Attributes:
    - result_storage: ResultStorage object containing simulation results.
    """

    def __init__(self, result_storage: ResultStorage | None = None):
        """
        Initialize the Plotter with simulation result data.

        Args:
            result_storage: ResultStorage object containing simulation results - can be set later by passing in None.
        """
        self.result_storage = result_storage

    @abstractmethod
    def plot(self, axes: Axes, **kwargs) -> None:
        """
        Abstract method to create a plot on the given axes.

        Args:
            axes: Matplotlib Axes object to plot on.
            **kwargs: Additional keyword arguments for customization.
        """
        pass

    @abstractmethod
    def check_valid_data(self) -> bool:
        """
        Abstract method to check if the required data for plotting is available.

        Returns:
            True if the required data is present, False otherwise.
        """
        pass

    @property
    def troubleshoot_string(self) -> str:
        """
        Provide a troubleshooting string if data is invalid.

        Returns:
            A string with troubleshooting information.
        """
        return "The required data for this plot is not available in the ResultStorage."


class DetectorIntensity1DPlotter(Plotter):
    """
    Plots the detector intensity vs. Source-Detector distance in 1D.


    Attributes:
        plot_log: Whether to plot the intensity on a logarithmic scale. In case of log scale, log10(intensity) is
                    plotted with a small offset of 1e-30 to avoid log(0).
    """

    def __init__(
        self, result_storage: ResultStorage | None = None, plot_log: bool = True
    ):
        super().__init__(result_storage)
        self.plot_log = plot_log

    def check_valid_data(self) -> bool:
        if self.result_storage is None:
            return False
        has_detpos = self.result_storage.detpos is not None
        has_detint = self.result_storage.detint is not None
        has_srcpos = self.result_storage.srcpos is not None
        return has_detpos and has_detint and has_srcpos

    def compute_sd_distances(self) -> np.ndarray:
        if self.result_storage is None:
            raise ValueError("Result storage is not set.")
        detpos = self.result_storage.detpos  # shape (N, 4)
        srcpos = self.result_storage.srcpos  # shape (3,)
        if detpos is None or srcpos is None:
            raise ValueError(
                "Detector positions or source position is not available."
            )
        sd_distances = np.linalg.norm(
            detpos[:, :3] - srcpos.reshape(1, 3), axis=1, ord=2.0
        )
        return sd_distances

    def plot(self, axes: Axes, **kwargs) -> None:
        if self.result_storage is None:
            raise ValueError("Result storage is not set.")
        sd_distances = self.compute_sd_distances()
        detint = self.result_storage.detint
        if self.plot_log:
            detint = np.log10(detint + 1e-30)  # Avoid log(0)   # type: ignore
        axes.plot(sd_distances, detint, **kwargs)  # type: ignore
        axes.set_xlabel("Source-Detector Distance")
        axes.set_ylabel("Detector Intensity")
        axes.set_title("Detector Intensity vs. Source-Detector Distance")

    @property
    def troubleshoot_string(self) -> str:
        return (
            "The ResultStorage must have DetectorIntensity, DetectorPositions, and SourcePosition data to plot "
            "Detector Intensity vs. Source-Detector Distance."
        )


class FluxPlotter(Plotter):
    """
    Plots a the 2D flux as an image slice.
    """

    def __init__(
        self, result_storage: ResultStorage | None = None, plot_log: bool = True
    ):
        super().__init__(result_storage)
        self._colorbar_values = {"vmin": 0.0, "vmax": 1.0, "norm": None}
        self.plot_log = plot_log

    def check_valid_data(self) -> bool:
        if self.result_storage is None:
            return False
        has_flux = self.result_storage.flux is not None
        return has_flux

    def plot(
        self,
        axes: Axes,
        slice_along_axis: int = 2,
        slice_axis_value: int = 30,
        time_index: int = 0,
        **kwargs
    ) -> None:
        if self.result_storage is None:
            raise ValueError("Result storage is not set.")
        flux = (
            self.result_storage.flux
        )  # This a 4D numpy array -> X, Y, Z, Time
        if flux is None:
            raise ValueError("Flux data is not available.")
        flux = flux[:, :, :, time_index]  # Select time index
        # Extract 2D slice
        if slice_along_axis == 0:
            flux = flux[slice_axis_value, :, :]
        elif slice_along_axis == 1:
            flux = flux[:, slice_axis_value, :]
        elif slice_along_axis == 2:
            flux = flux[:, :, slice_axis_value]
        else:
            raise ValueError("slice_along_axis must be 0, 1, or 2.")

        # If user didn't provide normalization/vmin/vmax, compute and set them so imshow has consistent scale
        if "cmap" not in kwargs:
            kwargs["cmap"] = default_colormap
        if self.plot_log:
            # Avoid log(0) by setting non-positive values to a small positive number
            flux = np.log10(flux)
        im = axes.imshow(flux, **kwargs)
        axes.set_title("2D Flux Slice")
        plt.colorbar(im, ax=axes)

    @property
    def troubleshoot_string(self) -> str:
        return "The ResultStorage must have Slice() data to plot a flux slice."


class FluxSlicePlotter(Plotter):
    """
    Plot an already 2D sliced version of flux as an image.
    """

    def __init__(
        self, result_storage: ResultStorage | None = None, plot_log: bool = True
    ):
        super().__init__(result_storage)
        self._colorbar_values = {"vmin": 0.0, "vmax": 1.0, "norm": None}
        self.plot_log = plot_log

    def check_valid_data(self) -> bool:
        if self.result_storage is None:
            return False
        has_flux_slice = self.result_storage.flux_slice is not None
        return has_flux_slice

    def plot(self, axes: Axes, **kwargs) -> None:
        if self.result_storage is None:
            raise ValueError("Result storage is not set.")
        flux_slice = self.result_storage.flux_slice  # This is a 2D numpy array
        if flux_slice is None:
            raise ValueError("Flux slice data is not available.")

        if "cmap" not in kwargs:
            kwargs["cmap"] = default_colormap

        if self.plot_log:
            flux_slice = np.log10(flux_slice)
        im = axes.imshow(flux_slice, **kwargs)
        axes.set_title("2D Flux Slice")
        plt.colorbar(im, ax=axes)

    @property
    def troubleshoot_string(self) -> str:
        return "The ResultStorage must have Slice() data to plot a flux slice."


class PlotType(Enum):
    FLUX_SLICE = "flux_slice"
    STAT_HISTOGRAM = "stat_histogram"
    DETECTOR_INTENSITY = "detector_intensity"


__all__ = [
    "Plotter",
    "FluxPlotter",
    "FluxSlicePlotter",
    "DetectorIntensity1DPlotter",
    "PlotType",
]
