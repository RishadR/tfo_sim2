"""
Experiment handler module for managing and executing batch simulations.

This module provides the ExperimentHandler class to run parameter sweeps over
SimulationParameters and TissueModels, storing results and tracking parameter values.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Literal
from pathlib import Path
import json
import itertools
import copy
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as Figure
from abc import ABC, abstractmethod

from tfo_sim2.simulator import Simulator
from tfo_sim2.simulation_params import SimulationParameters
from tfo_sim2.tissue_model import TissueModel
from tfo_sim2.result_storage import ResultStorage, StorableResults, StorageFormat
from tfo_sim2.plotter import Plotter
from tfo_sim2.detectors import DetectorArray


@dataclass
class ParameterSweep:
    """
    Defines a parameter sweep configuration.

    Attributes:
        param_path: Dot-separated path to the parameter (e.g., "nphoton" or "optical_props.mua")
        values: List of values to sweep over
        object_type: Either "simulation_params" or "tissue_model" to indicate which object to modify
    """

    param_path: str
    values: List[Any]
    object_type: Literal["simulation_params", "tissue_model"]

    def __post_init__(self):
        """Validate the sweep configuration."""
        if self.object_type not in ("simulation_params", "tissue_model"):
            raise ValueError(
                f"object_type must be 'simulation_params' or 'tissue_model', got '{self.object_type}'"
            )
        if not self.values:
            raise ValueError("values list cannot be empty")


class DynamicParameter(ABC):
    """
    Represents a parameter that changes dynamically with other parameters during setting up experiments.

    How to Use:
    1. Subclass DynamicParameter and implement the modify() method.
    2. In modify(), adjust the tissue_model or simulation_params based on current values.

    What this Does:
    - Called during experiment setup after static parameters are set.
    - Allows complex dependencies between parameters.

    Examples:
    Adjust source position based on tissue thickness, or modify optical properties
    SourceDynamicParameter(DynamicParameter):
        def modify(self, tissue_model: Any, simulation_params: Any) -> None:
            thickness = tissue_model.get_thickness()
            simulation_params.srcpos[2] = thickness + 1  # Place source just above

    """

    @abstractmethod
    def modify(
        self,
        tissue_model: TissueModel,
        simulation_params: SimulationParameters,
        detector_array: DetectorArray,
    ) -> None:
        pass


class ExperimentHandler:
    """
    Manages batch simulation experiments with parameter sweeps.

    Orchestrates running multiple simulations with different parameter combinations,
    collecting results, and storing them with metadata about the parameter sweep.
    """

    def __init__(
        self,
        base_simulation_params: SimulationParameters,
        base_tissue_model: TissueModel,
        output_dir: Union[str, Path],
        detector_array: Optional[DetectorArray] = None,
        storage_format: StorageFormat = StorageFormat.NPZ,
        results_to_store: Optional[List[StorableResults]] = None,
        dynamic_parameters: Optional[List[DynamicParameter]] = None,
        plotter: Optional[Plotter] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        fig_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the experiment handler.

        Args:
            base_simulation_params: Base simulation parameters to use as template
            base_tissue_model: Base tissue model instance (implementation of TissueModel ABC)
            output_dir: Directory to save results
            detector_array: Optional DetectorArray to use in simulations
            storage_format: Format for storing results (NPZ, HDF5, or JSON)
            results_to_store: List of StorableResults to include in ResultStorage.
                             If None, defaults to Flux and Statistics.
            dynamic_parameters: List of DynamicParameter instances for complex parameter dependencies
            plotter: Optional Plotter instance for generating & saving plots.
        """
        self.base_simulation_params = base_simulation_params
        self.base_tissue_model = base_tissue_model
        self.output_dir = Path(output_dir)
        self.detector_array = detector_array if detector_array is not None else DetectorArray()
        self.storage_format = storage_format
        self.results_to_store = results_to_store
        self.dynamic_parameters = (
            dynamic_parameters if dynamic_parameters is not None else []
        )
        self.plotter = plotter
        self.plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
        self.fig_kwargs = fig_kwargs if fig_kwargs is not None else {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track experiments
        self.experiments: List[Dict[str, Any]] = []
        self.results: List[Tuple[ResultStorage, Dict[str, Any]]] = []
        self._sweeps: List[ParameterSweep] = []
        self.figure_list: List[Figure] = []

    def add_sweep(self, sweep: ParameterSweep) -> None:
        """
        Add a parameter sweep to the experiment.

        Args:
            sweep: ParameterSweep object defining what to sweep
        """
        self._sweeps.append(sweep)

    def add_sweeps(self, sweeps: List[ParameterSweep]) -> None:
        """
        Add multiple parameter sweeps.

        Args:
            sweeps: List of ParameterSweep objects
        """
        for sweep in sweeps:
            self.add_sweep(sweep)

    def _set_nested_value(self, obj: Any, path: str, value: Any) -> None:
        """
        Set a nested attribute using dot notation.

        Args:
            obj: Object to modify
            path: Dot-separated path (e.g., "optical_props.mua")
            value: Value to set
        """
        parts = path.split(".")
        current = obj

        # Navigate to the parent of the final attribute
        for part in parts[:-1]:
            current = getattr(current, part)

        # Set the final attribute
        setattr(current, parts[-1], value)

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """
        Get a nested attribute using dot notation.

        Args:
            obj: Object to read from
            path: Dot-separated path (e.g., "optical_props.mua")

        Returns:
            The value at the path
        """
        parts = path.split(".")
        current = obj
        for part in parts:
            current = getattr(current, part)
        return current

    def _generate_parameter_grid(self) -> List[Dict[str, Tuple[str, Any]]]:
        """
        Generate all parameter combinations from the sweeps.

        Returns:
            List of dicts, each containing parameter combinations for one simulation.
            Each dict maps param_path -> (object_type, value)
        """
        if not self._sweeps:
            return [{}]

        # Separate sweeps by object type
        sim_sweeps = [s for s in self._sweeps if s.object_type == "simulation_params"]
        tissue_sweeps = [s for s in self._sweeps if s.object_type == "tissue_model"]

        # Generate all combinations
        sim_combinations = list(itertools.product(*[s.values for s in sim_sweeps]))
        tissue_combinations = list(
            itertools.product(*[s.values for s in tissue_sweeps])
        )

        # If no sweeps of a type, add a single empty tuple
        if not sim_combinations:
            sim_combinations = [()]
        if not tissue_combinations:
            tissue_combinations = [()]

        # Create parameter grid
        parameter_grid = []
        for sim_combo in sim_combinations:
            for tissue_combo in tissue_combinations:
                params = {}

                # Map simulation parameters
                for sweep, value in zip(sim_sweeps, sim_combo):
                    params[sweep.param_path] = ("simulation_params", value)

                # Map tissue parameters
                for sweep, value in zip(tissue_sweeps, tissue_combo):
                    params[sweep.param_path] = ("tissue_model", value)

                parameter_grid.append(params)

        return parameter_grid

    def run(self, name_prefix: str = "experiment") -> None:
        """
        Run all experiments in the parameter grid.

        Args:
            name_prefix: Prefix for result filenames
        """
        parameter_grid = self._generate_parameter_grid()

        self.experiments = []
        self.results = []

        for idx, params in enumerate(parameter_grid):
            # Create copies of base objects
            sim_params = copy.deepcopy(self.base_simulation_params)
            tissue_model = copy.deepcopy(self.base_tissue_model)
            detector_array = self.detector_array.create_copy()

            # Apply dynamic parameters before setting static ones
            for dynamic_param in self.dynamic_parameters:
                dynamic_param.modify(tissue_model, sim_params, detector_array)

            # Apply parameter values
            for param_path, (obj_type, value) in params.items():
                if obj_type == "simulation_params":
                    self._set_nested_value(sim_params, param_path, value)
                else:  # tissue_model
                    self._set_nested_value(tissue_model, param_path, value)

            # Run simulation
            simulator = Simulator(
                tissue_model=tissue_model,
                simulation_params=sim_params,
                detectors=detector_array,
            )
            result_dict = simulator.run()

            # Create ResultStorage
            storage = ResultStorage(
                data_dict=result_dict,
                results_to_store=self.results_to_store,
                name=f"{name_prefix}_{idx}",
            )

            # Track results with their parameters
            self.results.append((storage, params))
            self.experiments.append(
                {
                    "index": idx,
                    "name": f"{name_prefix}_{idx}",
                    "sweep_parameters": params,
                }
            )
            if self.plotter is not None:
                self.plotter.result_storage = storage
                if not self.plotter.check_valid_data():
                    raise ValueError(
                        f"Plotting Failed: {self.plotter.troubleshoot_string}"
                    )
                figure = plt.figure(**self.fig_kwargs)
                ax = figure.add_subplot(1, 1, 1)
                self.plotter.plot(ax, **self.plot_kwargs)
                self.figure_list.append(figure)
                plt.close(figure)

    def save_results(self) -> None:
        """
        Save all results to disk and create parameter mapping JSON.

        Creates:
        - Individual result files (experiment_0000.npz, etc.)
        - parameter_mapping.json with metadata about sweeps and base parameters
        """
        if not self.results:
            raise RuntimeError("No results to save. Run experiments first with .run()")

        # Prepare mapping data
        mapping_data = {
            "metadata": {
                "total_experiments": len(self.results),
                "storage_format": self.storage_format.value,
                "sweep_parameters": [
                    {
                        "param_path": s.param_path,
                        "object_type": s.object_type,
                        "values": s.values,
                    }
                    for s in self._sweeps
                ],
                "base_simulation_params": self._serialize_object(
                    self.base_simulation_params
                ),
                "base_tissue_model": self._serialize_object(self.base_tissue_model),
            },
            "experiments": [],
        }

        # Save results and build mapping
        for idx, (storage, params) in enumerate(self.results):
            # Create filename
            filename = f"experiment_{idx:04d}"
            filepath = self.output_dir / filename

            # Save result
            storage.save(str(filepath), format=self.storage_format)

            # Add to mapping
            mapping_data["experiments"].append(
                {
                    "filename": f"{filename}.{self.storage_format.value}",
                    "index": idx,
                    "sweep_parameters": self._params_to_readable(params),
                }
            )

        # Save any generated figures and link them in the mapping
        for idx, fig in enumerate(self.figure_list):
            img_name = f"experiment_{idx:04d}.png"
            img_path = self.output_dir / img_name
            try:
                fig.savefig(str(img_path), bbox_inches="tight")
            except Exception:
                fig.savefig(str(img_path))
            # If the corresponding experiment mapping exists, attach the figure filename
            if idx < len(mapping_data["experiments"]):
                mapping_data["experiments"][idx]["figure"] = img_name

        # Save mapping file
        mapping_path = self.output_dir / "parameter_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping_data, f, indent=2, default=str)

    def _params_to_readable(self, params: Dict[str, Tuple[str, Any]]) -> Dict[str, Any]:
        """
        Convert parameter dict to human-readable format.

        Args:
            params: Parameter dict from grid generation

        Returns:
            Readable dict with param_path: value pairs
        """
        readable = {}
        for param_path, (obj_type, value) in params.items():
            readable[param_path] = {
                "value": value,
                "object_type": obj_type,
            }
        return readable

    def _serialize_object(self, obj: Any) -> Dict[str, Any]:
        """
        Serialize an object to a dict for JSON storage.

        Args:
            obj: Object to serialize (SimulationParameters or TissueModel)

        Returns:
            Dictionary representation
        """
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    result[key] = value
                else:
                    result[key] = str(value)
            return result
        return {"_repr": str(obj)}

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all experiments run.

        Returns:
            Dict with experiment statistics and metadata
        """
        return {
            "total_experiments": len(self.experiments),
            "output_directory": str(self.output_dir),
            "storage_format": self.storage_format.value,
            "number_of_sweeps": len(self._sweeps),
            "sweep_configs": [
                {
                    "param_path": s.param_path,
                    "object_type": s.object_type,
                    "num_values": len(s.values),
                }
                for s in self._sweeps
            ],
            "experiments": self.experiments,
        }


__all__ = [
    "ParameterSweep",
    "DynamicParameter",
    "ExperimentHandler",
]
