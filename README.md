# TFO Sim2 - PMCX Wrapper Library

A comprehensive Python wrapper around the PMCX (Python Monte Carlo eXtreme) library, providing high-level abstractions for photon transport simulations in biological tissues.

## Overview

TFO Sim2 simplifies working with PMCX by providing intuitive interfaces for:
- **Tissue Models**: Define heterogeneous media with different optical properties
- **Simulation Parameters**: Configure photon transport simulations
- **Detectors**: Position detectors for photon detection
- **Simulations**: Run PMCX simulations with automatic cfg management
- **Result Storage**: Save and load results in multiple formats (NPZ, HDF5, JSON)
- **Batch Experiments**: Run parameter sweeps with cross-product combinations
- **Visualization**: Plot results with built-in plotters

## Installation

### Prerequisites

- Python 3.10+
- NumPy
- Matplotlib
- PMCX (required for actual simulations)

### Basic Installation

```bash
cd tfo_sim2
pip install -e .
```

### Optional Dependencies

```bash
# For advanced result storage formats
pip install jdata h5py
```

## Architecture

TFO Sim2 is built around a modular architecture with the following core components:

```
┌─────────────────────────────────────────────────────────┐
│                   ExperimentHandler                     │
│  (Orchestrates batch experiments with parameter sweeps) │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────┴────────┬──────────────┬──────────────┐
    │                 │              │              │
┌───▼──────┐  ┌──────▼─────┐  ┌─────▼─────┐  ┌────▼─────┐
│ Tissue   │  │ Simulation │  │ Detector  │  │ Result   │
│ Model    │  │ Parameters │  │ Array     │  │ Storage  │
└───┬──────┘  └──────┬─────┘  └─────┬─────┘  └────┬─────┘
    │                │              │              │
    └────────┬───────┴──────┬───────┴──────────────┘
             │              │
        ┌────▼──────┐  ┌────▼────────┐
        │ Simulator │  │ Plotter     │
        └───────────┘  └─────────────┘
             │
        ┌────▼─────┐
        │  PMCX    │
        └──────────┘
```

---

## Core Components

### 1. Tissue Models

Tissue models define the optical properties and geometry of the medium being simulated.

#### Base Class: `TissueModel` (ABC)

All tissue models inherit from the abstract `TissueModel` class and must implement:
- `_generate_volume()`: Creates a 3D numpy array of tissue tags
- `_generate_properties()`: Defines optical properties [μₐ, μₛ, g, n] for each tissue type
- `vol` property: Returns the volume array (lazy-loaded)
- `prop` property: Returns the properties array (lazy-loaded)

#### Built-in Models

**`UniformTissueModel`**
```python
from tfo_sim2 import UniformTissueModel

tissue = UniformTissueModel(
    size=(60, 60, 60),  # Volume dimensions in mm
    mua=0.005,          # Absorption coefficient (mm⁻¹)
    mus=1.0,            # Scattering coefficient (mm⁻¹)
    g=0.01,             # Anisotropy factor
    n=1.37,             # Refractive index
)
```

**`LayeredTissueModel`**
```python
from tfo_sim2 import LayeredTissueModel

layers = [
    {'z_start': 0, 'z_end': 20, 'mua': 0.005, 'mus': 0.5, 'g': 0.01, 'n': 1.37},
    {'z_start': 20, 'z_end': 40, 'mua': 0.01, 'mus': 1.0, 'g': 0.01, 'n': 1.37},
]

tissue = LayeredTissueModel(size=(60, 60, 60), layers=layers)
```

**`LapitanTissueModel`** (Extended Model)

Realistic skin tissue model with multiple layers:
- Epidermis (~1mm)
- Dermis (~7mm) with blood volume fraction control
- Subcutaneous tissue (~50mm)
- Wavelength-dependent optical properties (660nm, 810nm, 940nm)

```python
from tfo_sim2.tissue_model_extended import LapitanTissueModel

tissue = LapitanTissueModel(
    wavelength=660,      # Red light for PPG
    Vb_arterial=0.05,    # Arterial blood volume fraction
    Vb_venous=0.05,      # Venous blood volume fraction
)
```

**`DanModel4LayerX`** (Extended Model)

Fetal monitoring tissue model with 4 layers:
- Maternal wall (~2mm)
- Maternal uterus (~4mm)
- Amniotic fluid (~1mm, optional)
- Fetal tissue (~50mm)
- Wavelength range: 600-1000nm

```python
from tfo_sim2.tissue_model_extended import DanModel4LayerX

tissue = DanModel4LayerX(
    wavelength=650.0,
    maternal_hb_conc=15.0,
    maternal_saturation=1.0,
    fetal_saturation=0.6,
)
```

---

### 2. Simulation Parameters

The `SimulationParameters` class configures all aspects of the photon transport simulation.

#### Key Parameters

```python
from tfo_sim2 import SimulationParameters

params = SimulationParameters(
    # Photon configuration
    nphoton=1000000,        # Number of photons to simulate
    
    # Timing
    tstart=0.0,             # Start time (seconds)
    tend=5e-9,              # End time (seconds)
    tstep=5e-9,             # Time step for binning
    
    # Source configuration
    srcpos=[30, 30, 0],     # Source position [x, y, z] in mm
    srcdir=[0, 0, 1],       # Source direction vector
    srctype='pencil',       # Source type (pencil, isotropic, cone, etc.)
    
    # Output configuration
    outputtype='flux',      # 'flux' or 'jacobian'
    
    # Detector data saving
    issavedet=True,         # Enable detector data saving
    savedetflag='dpxv',     # d=ID, p=partial_path, x=position, v=velocity, etc.
)
```

#### Output Types
- **flux**: Standard fluence distribution
- **jacobian**: Sensitivity matrix for inverse problems

#### Detector Save Flags
- `d`: Detector ID
- `p`: Partial path length
- `x`: Exit position
- `v`: Exit velocity
- `w`: Initial weight
- `m`: Momentum transfer

---

### 3. Detector Arrays

The `DetectorArray` class manages detector positioning and configuration.

```python
from tfo_sim2 import DetectorArray

detectors = DetectorArray()

# Single detector
detectors.add_detector(position=(30, 30, 50), radius=2.0)

# Linear array
detectors.add_detector_line(
    start=(10, 10, 50),
    end=(50, 10, 50),
    num_detectors=10,
    radius=2.0
)

# Grid array
detectors.add_detector_grid(
    center=(30, 30, 50),
    spacing=5.0,
    nx=3, ny=3,
    radius=2.0
)

# Circle pattern
detectors.add_detector_circle(
    center=(30, 30, 50),
    radius_circle=10.0,
    num_detectors=8,
    detector_radius=2.0
)
```

---

### 4. Simulator

The `Simulator` class orchestrates the PMCX simulation by combining tissue models, parameters, and detectors.

```python
from tfo_sim2 import Simulator

simulator = Simulator(
    tissue_model=tissue,
    simulation_params=params,
    detector_array=detectors  # Optional
)

# Run simulation
result_dict = simulator.run()

# Access results
flux = result_dict['flux']
if detectors:
    detector_data = result_dict['detp']
```

#### Internal Workflow
1. Builds PMCX configuration dictionary via `build_cfg()`
2. Validates tissue model and simulation parameters
3. Calls `pmcx.mcxlab()` with generated configuration
4. Post-processes results
5. Returns dictionary with flux, statistics, and detector data

---

### 5. Result Storage

The `ResultStorage` class provides a unified interface for storing and loading simulation results in multiple formats.

#### Storage Formats
- **NPZ**: NumPy compressed format (default, fast)
- **HDF5**: Hierarchical format for large datasets
- **JSON**: Human-readable format with optional compression

#### Storable Results (Singletons)

All result types are implemented as singleton classes inheriting from `StorableResults`:

- `Flux()`: 3D fluence distribution
- `FluxSlice()`: 2D slice through the volume at z = z_max // 2
- `Statistics()`: Runtime statistics (nphoton, runtime, etc.)
- `DetectorID()`: Detector IDs for detected photons
- `DetectorPositions()`: Detector position array
- `SourcePosition()`: Source position
- `NumScatterings()`: Scattering events per photon
- `PartialPath()`: Path lengths in each tissue type
- `Momentum()`: Momentum transfer
- `ExitPosition()`: Photon exit positions
- `ExitVelocity()`: Photon exit velocities
- `InitialWeight()`: Initial photon weights
- `OpticalProperties()`: Tissue optical properties
- `UnitInMM()`: Unit conversion factor
- `DetectorIntensity()`: Computed intensity at each detector

```python
from tfo_sim2 import ResultStorage, Flux, FluxSlice, Statistics, StorageFormat

# Create storage with specific results
storage = ResultStorage(
    data_dict=result_dict,
    results_to_store=[Flux(), FluxSlice(), Statistics()],
    name="my_simulation"
)

# Save to disk
storage.save("results/sim_001", format=StorageFormat.NPZ)

# Load from disk
loaded = ResultStorage.load("results/sim_001.npz", format=StorageFormat.NPZ)

# Access results
flux = loaded.flux
flux_slice = loaded.flux_slice
stats = loaded.stat
```

#### Default Results
By default, `ResultStorage` includes:
- `SourcePosition()`
- `OpticalProperties()`
- `UnitInMM()`
- `Statistics()`
- Plus any explicitly specified results

---

### 6. Experiment Handler

The `ExperimentHandler` class orchestrates batch experiments with parameter sweeps.

#### Features
- Cross-product parameter grid generation
- Support for sweeping both `SimulationParameters` and `TissueModel` attributes
- Nested parameter access via dot notation
- Automatic result storage with parameter mapping
- Deep copying ensures parameter isolation between experiments

```python
from tfo_sim2 import ExperimentHandler, ParameterSweep, StorageFormat
from pathlib import Path

# Setup base configuration
base_params = SimulationParameters(nphoton=10000, tend=5e-9)
base_tissue = UniformTissueModel(size=(60,60,60), mua=0.005, mus=1.0, g=0.01, n=1.37)

# Create handler
handler = ExperimentHandler(
    base_simulation_params=base_params,
    base_tissue_model=base_tissue,
    output_dir=Path("./batch_results"),
    storage_format=StorageFormat.NPZ,
    results_to_store=[Flux(), Statistics()],
)

# Define parameter sweeps
nphoton_sweep = ParameterSweep(
    param_path="nphoton",
    values=[5000, 10000, 20000],
    object_type="simulation_params"
)

mua_sweep = ParameterSweep(
    param_path="mua",
    values=[0.005, 0.01, 0.02],
    object_type="tissue_model"
)

handler.add_sweeps([nphoton_sweep, mua_sweep])

# Run all experiments (3 × 3 = 9 simulations)
handler.run(name_prefix="experiment")

# Save results with parameter mapping
handler.save_results()

# Get summary
summary = handler.get_results_summary()
```

#### Parameter Mapping Output

`parameter_mapping.json` contains:
```json
{
  "metadata": {
    "total_experiments": 9,
    "storage_format": "npz",
    "sweep_parameters": [...],
    "base_simulation_params": {...},
    "base_tissue_model": {...}
  },
  "experiments": [
    {
      "filename": "experiment_0000.npz",
      "index": 0,
      "sweep_parameters": {
        "nphoton": {"value": 5000, "object_type": "simulation_params"},
        "mua": {"value": 0.005, "object_type": "tissue_model"}
      }
    },
    ...
  ]
}
```

#### Dynamic Parameters

For advanced use cases, define `DynamicParameter` classes to modify configurations dynamically:

```python
from tfo_sim2 import DynamicParameter

class SourceDynamicParameter(DynamicParameter):
    def modify(self, tissue_model, simulation_params, detector_array):
        # Adjust source position based on tissue geometry
        top_z = tissue_model.topmost_pixel()
        simulation_params.srcpos[2] = top_z + 1
        
        # Clear and rebuild detector array
        detector_array.detectors.clear()
        detector_array.add_detector_line(
            (110, 112, top_z + 1), 
            (110, 210, top_z + 1), 
            10, 2.0
        )

handler = ExperimentHandler(
    ...,
    dynamic_parameters=[SourceDynamicParameter()]
)
```

---

### 7. Visualization

Built-in plotters for common visualization tasks.

#### DetectorIntensity1DPlotter

Plots detector intensity as a function of position for 1D detector arrays.

```python
from tfo_sim2 import DetectorIntensity1DPlotter

plotter = DetectorIntensity1DPlotter()

# Automatic plotting during batch experiments
handler = ExperimentHandler(
    ...,
    plotter=plotter
)
```

---

## Complete Examples

### Example 1: Simple Uniform Tissue Simulation

```python
from tfo_sim2 import (
    UniformTissueModel,
    SimulationParameters,
    DetectorArray,
    Simulator,
)

# Create tissue
tissue = UniformTissueModel(
    size=(60, 60, 60),
    mua=0.005,
    mus=1.0,
    g=0.01,
    n=1.37,
)

# Configure simulation
params = SimulationParameters(
    nphoton=1000000,
    tend=5e-9,
    srcpos=[30, 30, 0],
    srcdir=[0, 0, 1],
)

# Add detectors
detectors = DetectorArray()
detectors.add_detector(position=(30, 30, 50), radius=2.0)

# Run simulation
simulator = Simulator(tissue, params, detectors)
result = simulator.run()

print(f"Flux shape: {result['flux'].shape}")
```

### Example 2: Batch Experiment with Parameter Sweeps

```python
from pathlib import Path
from tfo_sim2 import (
    ExperimentHandler,
    ParameterSweep,
    SimulationParameters,
    UniformTissueModel,
    StorageFormat,
    FluxSlice,
)

# Base configuration
base_sim_params = SimulationParameters(
    nphoton=10000,
    tend=5e-9,
    srcpos=[30, 30, 0],
    srcdir=[0, 0, 1],
)

base_tissue = UniformTissueModel(
    size=(60, 60, 60),
    mua=0.005,
    mus=1.0,
    g=0.01,
    n=1.37,
)

# Create handler
handler = ExperimentHandler(
    base_simulation_params=base_sim_params,
    base_tissue_model=base_tissue,
    output_dir=Path("./batch_results"),
    storage_format=StorageFormat.NPZ,
    results_to_store=[FluxSlice()],
)

# Define sweeps
nphoton_sweep = ParameterSweep(
    param_path="nphoton",
    values=[5000, 10000, 20000],
    object_type="simulation_params",
)

tend_sweep = ParameterSweep(
    param_path="tend",
    values=[1e-9, 5e-9, 10e-9],
    object_type="simulation_params",
)

handler.add_sweeps([nphoton_sweep, tend_sweep])

# Run 3 × 3 = 9 simulations
handler.run(name_prefix="flux_sweep")
handler.save_results()

summary = handler.get_results_summary()
print(f"Total experiments: {summary['total_experiments']}")
```

### Example 3: Realistic Tissue Model with Blood Perfusion

```python
from tfo_sim2 import SimulationParameters, ExperimentHandler, ParameterSweep, StorageFormat, FluxSlice
from tfo_sim2.tissue_model_extended import LapitanTissueModel
from pathlib import Path

# Realistic skin tissue
base_tissue = LapitanTissueModel(
    wavelength=660,
    Vb_arterial=0.05,
    Vb_venous=0.05,
)

base_sim_params = SimulationParameters(
    nphoton=10000,
    tend=5e-9,
    srcpos=[51, 51, 58],  # Top surface
    srcdir=[0, 0, -1],    # Downward
)

handler = ExperimentHandler(
    base_simulation_params=base_sim_params,
    base_tissue_model=base_tissue,
    output_dir=Path("./lapitan_results"),
    storage_format=StorageFormat.NPZ,
    results_to_store=[FluxSlice()],
)

# Sweep blood volume to simulate pulsation
vb_arterial_sweep = ParameterSweep(
    param_path="Vb_arterial",
    values=[0.025, 0.05, 0.1],
    object_type="tissue_model",
)

handler.add_sweeps([vb_arterial_sweep])
handler.run(name_prefix="ppg_simulation")
handler.save_results()
```

---

## Advanced Features

### Nested Parameter Access

Use dot notation to access nested attributes:

```python
# For tissue models with nested optical properties
sweep = ParameterSweep(
    param_path="optical_props.mua",  # Nested attribute
    values=[0.005, 0.01, 0.02],
    object_type="tissue_model"
)
```

### Custom Result Types

Extend `StorableResults` to create custom result extractors:

```python
from tfo_sim2.result_storage import StorableResults

class MyCustomResult(StorableResults):
    @property
    def name(self) -> str:
        return "my_custom"
    
    def validity_check(self, data_dict):
        return "my_data" in data_dict
    
    def extract(self, data_dict):
        return data_dict["my_data"] * 2  # Custom processing
    
    @property
    def troubleshoot_string(self) -> str:
        return "Missing 'my_data' key"
```

---

## API Reference

For detailed API documentation, see the docstrings in:
- `tfo_sim2/tissue_model.py` - Base tissue model classes
- `tfo_sim2/tissue_model_extended.py` - Extended tissue models
- `tfo_sim2/simulation_params.py` - Simulation configuration
- `tfo_sim2/detectors.py` - Detector management
- `tfo_sim2/simulator.py` - Simulation execution
- `tfo_sim2/result_storage.py` - Result management
- `tfo_sim2/experiment_handler.py` - Batch experiments
- `tfo_sim2/plotter.py` - Visualization tools

---

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_experiment_handler.py -v
pytest tests/test_simulator.py -v
pytest tests/test_result_storage.py -v
```

---

## Project Structure

```
tfo_sim2/
├── src/tfo_sim2/
│   ├── __init__.py
│   ├── tissue_model.py              # Base tissue model classes
│   ├── tissue_model_extended.py     # Extended models (Lapitan, Dan4Layer)
│   ├── simulation_params.py         # Simulation configuration
│   ├── detectors.py                 # Detector array management
│   ├── simulator.py                 # PMCX simulation wrapper
│   ├── result_storage.py            # Result storage and loading
│   ├── experiment_handler.py        # Batch experiment orchestration
│   ├── plotter.py                   # Visualization tools
│   └── examples.py                  # Example scripts
├── tests/                           # Unit tests
├── examples/                        # Example scripts
│   ├── batch_experiment_example.py
│   └── lapitan_batch_experiment_example.py
├── experiments/                     # Research experiments
├── docs/                            # Documentation
├── pyproject.toml                   # Project configuration
└── README.md                        # This file
```

---

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Docstrings are provided for all public methods
- Type hints are used where appropriate
- Tests are included for new features

---

## License

This wrapper is provided as-is. PMCX itself is licensed under GNU Public License V3.

---

## References

- PMCX Documentation: https://mcx.space/
- Monte Carlo eXtreme: http://mcx.space/wiki/
- NeuroJSON Project: https://neurojson.org/

---

## Citation

If you use TFO Sim2 in your research, please cite the OG PMCX library (Not mine):

```
Fang Q, Yan S (2019). "Graphics processing unit-accelerated mesh-based 
Monte Carlo photon transport simulations." J. Biomed. Opt. 24(11), 115002.
```
