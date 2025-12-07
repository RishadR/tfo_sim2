"""
Result storage and export for simulation results.

This module provides classes to store, manage, and export simulation results
in various formats including JSON, HDF5, and numpy binary formats.
"""

from abc import ABC, ABCMeta
from enum import Enum
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings
import numpy as np
from pmcx.utils import detweight
from numpy.typing import NDArray
import numpy as np

try:
    import jdata as jd
except ImportError:
    jd = None

try:
    import h5py
except ImportError:
    h5py = None


class StorageFormat(Enum):
    """Supported result storage formats."""

    JSON = "json"
    NPZ = "npz"
    HDF5 = "hdf5"


class SingletonABCMeta(ABCMeta):
    """Metaclass that combines ABC and Singleton pattern."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Actially let make these classes with their own read and write methods
class StorableResults(ABC, metaclass=SingletonABCMeta):
    """
    Abstract base class for storable simulation results.

    Each class should define a name string, a validity_check method and an extract method.

    Methods:
    ------------------
    name: str
        Name of the storable result type.
    validity_check(data_dict: Dict[str, Any]) -> bool
        Check if the data_dict contains valid data for this result type.
    extract(data_dict: Dict[str, Any]) -> Any
        Extract the relevant data from the data_dict if exists. Otherwise return None.

    """

    @property
    def name(self) -> str:
        """Name of the storable result type."""
        raise NotImplementedError("Subclasses must implement the name property.")

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        """Check if the data_dict contains valid data for this result type."""
        raise NotImplementedError(
            "Subclasses must implement the validity_check method."
        )

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        """Extract the relevant data from the data_dict."""
        raise NotImplementedError("Subclasses must implement the extract method.")

    @property
    def troubleshoot_string(self) -> str:
        """String to help with troubleshooting missing data."""
        raise NotImplementedError(
            "Subclasses must implement the troubleshoot_string property."
        )


class DetectorPositions(StorableResults):
    @property
    def name(self) -> str:
        return "detpos"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        temp = data_dict.get("detpos", None)
        return temp is not None and len(temp) > 0

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return np.array(data_dict["detpos"])  # type: ignore

    @property
    def troubleshoot_string(self) -> str:
        return "The data_dict is missing the 'detpos' key. - must define detectors in the Simulator."


class SourcePosition(StorableResults):
    @property
    def name(self) -> str:
        return "srcpos"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "srcpos" in data_dict

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return np.array(data_dict.get("srcpos", None))  # type: ignore

    @property
    def troubleshoot_string(self) -> str:
        return "The data_dict is missing the 'srcpos' key. - shouldn't happen for this library's simulator was used."


class Flux(StorableResults):
    @property
    def name(self) -> str:
        return "flux"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "flux" in data_dict

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return data_dict.get("flux", None)

    @property
    def troubleshoot_string(self) -> str:
        return "The data_dict is missing the 'flux' key. - the simulation did not complete successfully."


class FluxSlice(StorableResults):
    def __init__(
        self,
        slice_along_axis: int = 2,
        slice_axis_value: int = 20,
        time_index: int = 0,
    ):
        self.slice_along_axis = slice_along_axis
        self.slice_axis_value = slice_axis_value
        self.time_index = time_index

    @property
    def name(self) -> str:
        return "flux_slice"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "flux" in data_dict

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        flux = data_dict.get("flux")
        if flux is not None:
            flux = flux[:, :, :, self.time_index]  # Select time index
            # Extract 2D slice
            if self.slice_along_axis == 0:
                return flux[self.slice_axis_value, :, :]
            elif self.slice_along_axis == 1:
                return flux[:, self.slice_axis_value, :]
            elif self.slice_along_axis == 2:
                return flux[:, :, self.slice_axis_value]
            else:
                raise ValueError("Invalid slice_along_axis value")
        return None

    @property
    def troubleshoot_string(self) -> str:
        return "The data_dict is missing the 'flux' key. - the simulation did not complete successfully."


class Statistics(StorableResults):
    @property
    def name(self) -> str:
        return "stat"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "stat" in data_dict

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return data_dict.get("stat", None)

    @property
    def troubleshoot_string(self) -> str:
        return "The data_dict is missing the 'stat' key. - shouldn't happen for this library's simulator was used."


class DetectorID(StorableResults):
    @property
    def name(self) -> str:
        return "detid"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "detp" in data_dict and "detid" in data_dict["detp"]

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return data_dict["detp"]["detid"]

    @property
    def troubleshoot_string(self) -> str:
        return (
            "Must define the detectors in the Simulator with a non-empty DetectorArray."
        )


class NumScatterings(StorableResults):
    @property
    def name(self) -> str:
        return "nscat"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "detp" in data_dict and "nscat" in data_dict["detp"]

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return data_dict["detp"]["nscat"]

    @property
    def troubleshoot_string(self) -> str:
        return "The SimulatorParameters's savedetflag variable must include 's' to save scattering count information. \
            and must have a detector array defined in the Simulator."


class PartialPath(StorableResults):
    """
    Stores the partial path information from the simulation results along with detector IDs. The detector IDs are in
    the first column of the returned numpy array.
    """

    @property
    def name(self) -> str:
        return "ppath"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return (
            "detp" in data_dict
            and "ppath" in data_dict["detp"]
            and "detid" in data_dict["detp"]
        )

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        ppath = data_dict["detp"]["ppath"]
        detid = data_dict["detp"]["detid"]
        # Combine detid and ppath into a single array
        combined = np.hstack((detid.reshape(-1, 1), ppath))
        return combined

    @property
    def troubleshoot_string(self) -> str:
        return "The SimulatorParameters's savedetflag variable must include 'p' to save partial path information and \
            must have a detector array defined in the Simulator."


class Momentum(StorableResults):
    @property
    def name(self) -> str:
        return "momentum"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "detp" in data_dict and "mom" in data_dict["detp"]

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return data_dict["detp"]["mom"]

    @property
    def troubleshoot_string(self) -> str:
        return "The SimulatorParameters's savedetflag variable must include 'm' to save momentum transfer information \
            and must have a detector array defined in the Simulator."


class ExitPosition(StorableResults):
    @property
    def name(self) -> str:
        return "exit_position"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "detp" in data_dict and "p" in data_dict["detp"]

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return data_dict["detp"]["p"]

    @property
    def troubleshoot_string(self) -> str:
        return "The SimulatorParameters's savedetflag variable must include 'x' to save exit position information and \
            must have a detector array defined in the Simulator."


class ExitVelocity(StorableResults):
    @property
    def name(self) -> str:
        return "exit_velocity"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "detp" in data_dict and "v" in data_dict["detp"]

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return data_dict["detp"]["v"]

    @property
    def troubleshoot_string(self) -> str:
        return "The SimulatorParameters's savedetflag variable must include 'v' to save exit velocity information and \
            must have a detector array defined in the Simulator."


class InitialWeight(StorableResults):
    @property
    def name(self) -> str:
        return "initial_weight"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "detp" in data_dict and "w0" in data_dict["detp"]

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return data_dict["detp"]["w0"]

    @property
    def troubleshoot_string(self) -> str:
        return "The SimulatorParameters's savedetflag variable must include 'w' to save initial weight information and \
            must have a detector array defined in the Simulator."


class OpticalProperties(StorableResults):
    @property
    def name(self) -> str:
        return "optical_properties"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "detp" in data_dict and "prop" in data_dict["detp"]

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return data_dict["detp"]["prop"]

    @property
    def troubleshoot_string(self) -> str:
        return "No detectors defined or the simulation did not complete successfully - could not extract 'prop' from \
            'detp' inside the data_dict."


class UnitInMM(StorableResults):
    @property
    def name(self) -> str:
        return "unit_in_mm"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return "detp" in data_dict and "unitinmm" in data_dict["detp"]

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        return float(data_dict["detp"]["unitinmm"])

    @property
    def troubleshoot_string(self) -> str:
        return "Could not find 'unitinmm' in 'detp' inside the data_dict - which typically should always be present."


class DetectorIntensity(StorableResults):
    @property
    def name(self) -> str:
        return "detint"

    def validity_check(self, data_dict: Dict[str, Any]) -> bool:
        return (
            ("detp" in data_dict)
            and ("detpos" in data_dict)
            and all([x in data_dict["detp"] for x in ["detid", "ppath", "prop"]])
        )

    def extract(self, data_dict: Dict[str, Any]) -> Any:
        all_detectors = np.array(list(range(1, len(data_dict["detpos"]) + 1)))
        individual_intensities = detweight(data_dict["detp"])
        det_int = np.zeros(len(all_detectors))
        for i, det_id in enumerate(all_detectors):
            det_int[i] = np.sum(
                individual_intensities[data_dict["detp"]["detid"] == det_id]
            )
        return det_int

    @property
    def troubleshoot_string(self) -> str:
        return "The SimulatorParameters's savedetflag variable must include 'd' and 'p' to compute detector intensity \
            and must also have a detector array defined in the Simulator."


# TODO: At some point, implement some of the other utils


class ResultStorage:
    """
    Stores and manages simulation results with support for multiple formats.
    """

    result_type_map = {
        "flux": Flux(),
        "flux_slice": FluxSlice(),
        "stat": Statistics(),
        "detid": DetectorID(),
        "nscat": NumScatterings(),
        "ppath": PartialPath(),
        "momentum": Momentum(),
        "exit_position": ExitPosition(),
        "exit_velocity": ExitVelocity(),
        "initial_weight": InitialWeight(),
        "optical_properties": OpticalProperties(),
        "unit_in_mm": UnitInMM(),
        "detint": DetectorIntensity(),
        "srcpos": SourcePosition(),
        "detpos": DetectorPositions(),
    }

    def __init__(
        self,
        data_dict: Dict[str, Any],
        results_to_store: Optional[List[StorableResults]] = None,
        name: str = "Tissue_Sim",
    ):
        """
        Initialize ResultStorage with simulation results.

        Args:
            data_dict: Dictionary containing simulation results from PMCX
            results_to_store: List of StorableResults enum values to store.
                            If None, defaults to FLUX and STATISTICS.
            name: Name for this result set.
        """
        default_stores = [SourcePosition(), OpticalProperties(), UnitInMM(), Statistics()]
        
        if results_to_store is None:
            results_to_store = [Flux()]

        results_to_store = list(set(results_to_store + default_stores))

        self.name = name
        self.data_dict = data_dict

        # Pre-declare all result attributes for type checking
        # These will be populated or set to None below
        self.flux: Optional[NDArray] = None
        self.stat: Optional[Any] = None
        self.flux_slice: Optional[NDArray] = None
        self.detid: Optional[NDArray] = None
        self.nscat: Optional[NDArray] = None
        self.ppath: Optional[NDArray] = None
        self.momentum: Optional[NDArray] = None
        self.exit_position: Optional[NDArray] = None
        self.exit_velocity: Optional[NDArray] = None
        self.initial_weight: Optional[NDArray] = None
        self.optical_properties: Optional[Any] = None
        self.unit_in_mm: Optional[float] = None
        self.detint: Optional[NDArray] = None
        self.detpos: Optional[NDArray] = None
        self.srcpos: Optional[NDArray] = None

        for result_type in results_to_store:
            attr_name = result_type.name
            if result_type.validity_check(data_dict):
                setattr(self, attr_name, result_type.extract(data_dict))
            else:
                warnings.warn(
                    f"Requested result '{attr_name}' could not be computed. {result_type.troubleshoot_string}",
                    UserWarning,
                )
                setattr(self, attr_name, None)

    def __str__(self) -> str:
        return f"ResultStorage(name={self.name}),\n" \
               f"Available Results: {[key for key in self.__dict__.keys() if getattr(self, key) is not None]}"
               
    
    
    def save(self, filepath: str, format: StorageFormat = StorageFormat.NPZ) -> None:
        """
        Save the ResultStorage object to disk.

        Args:
            filepath: Path where the file will be saved.
            format: StorageFormat enum specifying the output format.
                   Options: NPZ, HDF5, JSON

        Raises:
            ValueError: If format is not supported or required libraries are missing.
            IOError: If file cannot be written.
        """
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == StorageFormat.NPZ:
            self._save_npz(filepath_obj)
        elif format == StorageFormat.HDF5:
            self._save_hdf5(filepath_obj)
        elif format == StorageFormat.JSON:
            self._save_json(filepath_obj)
        else:
            raise ValueError(f"Unsupported storage format: {format}")

    def _save_npz(self, filepath: Path) -> None:
        """Save results to NPZ (numpy binary) format."""
        # Collect attributes to save (everything set in __init__ except data_dict)
        data_to_save = {}

        present_result_types = []

        for attr, value in self.__dict__.items():
            if attr == "data_dict":
                continue
            # Skip unset attributes
            if value is None:
                continue
            # Convert lists/tuples to numpy arrays for more compact storage
            data_to_save[attr] = value
        np.savez_compressed(str(filepath), **data_to_save)

    def _save_hdf5(self, filepath: Path) -> None:
        """Save results to HDF5 format."""
        if h5py is None:
            raise ImportError("h5py is required for HDF5 format support.")

        with h5py.File(str(filepath), "w") as f:
            # Store each attribute as a dataset
            for attr, value in self.__dict__.items():
                if attr == "data_dict" or value is None:
                    continue
                f.create_dataset(attr, data=value)

    def _save_json(self, filepath: Path) -> None:
        """Save results to JSON format using jdata."""
        if jd is None:
            raise ImportError("jdata is required for JSON format support.")

        data_to_save = {}

        for attr, value in self.__dict__.items():
            if attr == "data_dict" or value is None:
                continue
            data_to_save[attr] = value
        jd.save(data_to_save, str(filepath))

    @classmethod
    def load(
        cls, filepath: str, format: Optional[StorageFormat] = None
    ) -> "ResultStorage":
        """
        Load a ResultStorage object from disk.

        Args:
            filepath: Path to the saved file.
            format: StorageFormat enum specifying the file format.
                   If None, format is inferred from file extension.

        Returns:
            A new ResultStorage instance populated with the saved data.

        Raises:
            ValueError: If format cannot be determined or is unsupported.
            IOError: If file cannot be read.
        """
        filepath_obj = Path(filepath)

        if not filepath_obj.exists():
            raise IOError(f"File not found: {filepath_obj}")

        # Infer format from extension if not provided
        if format is None:
            suffix = filepath_obj.suffix.lower()
            if suffix == ".npz":
                format = StorageFormat.NPZ
            elif suffix in [".h5", ".hdf5"]:
                format = StorageFormat.HDF5
            elif suffix == ".json":
                format = StorageFormat.JSON
            else:
                raise ValueError(
                    f"Cannot infer format from extension '{suffix}'. "
                    "Please specify format explicitly."
                )

        if format == StorageFormat.NPZ:
            return cls._load_npz(filepath_obj)
        elif format == StorageFormat.HDF5:
            return cls._load_hdf5(filepath_obj)
        elif format == StorageFormat.JSON:
            return cls._load_json(filepath_obj)
        else:
            raise ValueError(f"Unsupported storage format: {format}")

    @classmethod
    def _load_npz(cls, filepath: Path) -> "ResultStorage":
        """Load results from NPZ format."""
        if np is None:
            raise ImportError("NumPy is required for NPZ format support.")

        data = np.load(str(filepath), allow_pickle=True)
        loaded_dict = {key: data[key] for key in data.files}
        instance = cls({}, [], name=loaded_dict.get("name", "Loaded_Result"))
        for attr, value in loaded_dict.items():
            setattr(instance, attr, value)
        return instance

    @classmethod
    def _load_hdf5(cls, filepath: Path) -> "ResultStorage":
        """Load results from HDF5 format."""
        if h5py is None:
            raise ImportError("h5py is required for HDF5 format support.")

        loaded_dict = {}
        with h5py.File(str(filepath), "r") as f:
            for key in f.keys():
                # h5py Dataset/Datatype does not expose __getitem__ to some static type checkers,
                # use a type ignore to silence the warning while still reading the data.
                loaded_dict[key] = f[key][()]  # type: ignore
        instance = cls({}, [], name=loaded_dict.get("name", "Loaded_Result"))
        for attr, value in loaded_dict.items():
            setattr(instance, attr, value)
        return instance

    @classmethod
    def _load_json(cls, filepath: Path) -> "ResultStorage":
        """Load results from JSON format using jdata."""
        if jd is None:
            raise ImportError("jdata is required for JSON format support.")

        loaded_dict = jd.load(str(filepath))
        instance = cls({}, [], name=loaded_dict.get("name", "Loaded_Result"))
        for attr, value in loaded_dict.items():
            setattr(instance, attr, value)
        return instance


__all__ = [
    "StorageFormat",
    "StorableResults",
    "Flux",
    "FluxSlice",
    "Statistics",
    "DetectorID",
    "NumScatterings",
    "PartialPath",
    "Momentum",
    "ExitPosition",
    "ExitVelocity",
    "InitialWeight",
    "OpticalProperties",
    "UnitInMM",
    "DetectorIntensity",
    "SourcePosition",
    "DetectorPositions",
    "ResultStorage",
]
