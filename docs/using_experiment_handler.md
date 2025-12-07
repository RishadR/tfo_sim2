# ExperimentHandler: Orchestrating Batch Simulations

`ExperimentHandler` is a simple framework for managing parameter sweeps and batch simulations in TFO-Sim2. It abstracts away the hassles of running multiple simulations with different parameter combinations, storing results systematically, and optionally generating plots for each run.

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Getting Started: Basic Usage](#getting-started-basic-usage)
3. [Adding Parameters to Sweeps](#adding-parameters-to-sweeps)
4. [Understanding Results Storage](#understanding-results-storage)
5. [Visualization with Plotters](#visualization-with-plotters)
6. [Dynamic Parameters: Complex Dependencies](#dynamic-parameters-complex-dependencies)
7. [The Operational Sequence](#the-operational-sequence)
8. [Complete Example](#complete-example)

---

## Overview & Architecture

Pipeline Overview: 
<!-- TODO: Eventually, someday turn this into an actual diagram. But that day is not today -->

```
Parameter Grid Generation → Parameter Application → Simulation Execution → Result Storage → Visualization
```

### Key Components

- **ParameterSweep**: Defines which parameters to vary and how to create the sweep grid
- **DynamicParameter**: Allows complex, inter-dependent parameter modifications post sweep grid generation
- **ResultStorage**: Manages which types of simulation outputs to save (flux, detector data, etc.)
- **Plotter**: Plots selected results from each simulation (Optional)
- **ExperimentHandler**: Orchestrates the entire workflow

### Design Philosophy

Following a **declarative** design pattern. Specify *what* needs to happen and let my handler manages the rest.

---

## Getting Started: Basic Usage

Documenting a few simple examples here. For more check the `./examples` folder

### Minimal Example

```python
from pathlib import Path
from tfo_sim2 import (
    ExperimentHandler,
    SimulationParameters,
    TissueModel,
    DetectorArray,
    ParameterSweep,
)

# 1. Create base objects
sim_params = SimulationParameters(nphoton=1000000, srcpos=[30, 30, 0])
tissue_model = MyTissueModel()  # Your TissueModel subclass
detector_array = DetectorArray()

# 2. Create the handler
handler = ExperimentHandler(
    base_simulation_params=sim_params,
    base_tissue_model=tissue_model,
    output_dir="./batch_results/my_experiment",
    detector_array=detector_array,
)

# 3. Add a parameter sweep
handler.add_sweep(
    ParameterSweep(
        param_path="nphoton",
        values=[100000, 500000, 1000000],
        object_type="simulation_params",
    )
)

# 4. Run the experiments
handler.run(name_prefix="photon_count_sweep")

# 5. Save results
handler.save_results()

# 6. Inspect results
summary = handler.get_results_summary()
print(f"Completed {summary['total_experiments']} experiments")
```

This creates 3 separate simulations with 100k, 500k, and 1M photons respectively, storing results in `batch_results/my_experiment/`.

### What Gets Created

After `save_results()`, your output directory contains:

```
batch_results/my_experiment/
├── experiment_0000.npz        # First simulation results
├── experiment_0001.npz        # Second simulation results
├── experiment_0002.npz        # Third simulation results
└── parameter_mapping.json     # Metadata about all experiments
```

The `parameter_mapping.json` includes:
- Base parameters used
- Sweep configurations
- Parameter values for each experiment
- Filenames and indices

---

## Adding Parameters to Sweeps



### Parameter Object Types

Sweep-able parameters are split into two categories
1.  `simulation_params` that affect the simulation settings, such as nphoton, timesteps etc. For a comprehensive list check the `simulation_params.SimulationParameters` object. 
2.   `tissue_model` that are feed into the `__init__` method of your tissue model object. In code flow, you define a base tissue model with some set of parameters. The handler will *only* overwrite the variables defined as sweep parameters in base model while keeping the rest unchanged.


### Theory: Dot Notation

Parameters are specified using **dot notation** to handle nested attributes. This enables sweeping over both top-level and deeply nested parameters.Dot notation allows addressing nested attributes using string paths:

```python
# Top-level attribute
"nphoton"  # → sim_params.nphoton

# Nested attribute (hypothetical)
"optical_props.mua"  # → tissue_model.optical_props.mua
"srcpos"  # Can be a list, modified element-wise in subclasses
```
Note: I would recommend avoiding do notations if possible

### Example 1: Sweeping Photon Count

```python
handler.add_sweep(
    ParameterSweep(
        param_path="nphoton",
        values=[100000, 500000, 1000000, 5000000],
        object_type="simulation_params",
    )
)
```

This creates 4 simulations with different photon counts.

### Example 2: Sweeping Multiple Parameters

```python
# Define multiple sweeps
sweeps = [
    ParameterSweep(
        param_path="nphoton",
        values=[100000, 1000000],
        object_type="simulation_params",
    ),
    ParameterSweep(
        param_path="srcpos",  # Modify source position
        values=[[20, 20, 0], [30, 30, 0], [40, 40, 0]],
        object_type="simulation_params",
    ),
]

handler.add_sweeps(sweeps)
```

This creates **2 × 3 = 6** simulations (a Cartesian product of all combinations).

### Example 3: Sweeping Tissue Properties

```python
# Sweep over absorption coefficient in the tissue model
handler.add_sweep(
    ParameterSweep(
        param_path="mua_tissue",  # Custom attribute in your TissueModel
        values=[0.01, 0.05, 0.1, 0.2],
        object_type="tissue_model",
    )
)
```

When this parameter is set during the experiment, `tissue_model.mua_tissue = value` is executed before simulation.

### Example 4: Complex Multi-Parameter Sweep

```python
handler.add_sweeps([
    ParameterSweep(
        param_path="nphoton",
        values=[500000, 1000000],
        object_type="simulation_params",
    ),
    ParameterSweep(
        param_path="depth",
        values=[1, 2, 5],
        object_type="tissue_model",
    ),
    ParameterSweep(
        param_path="tend",
        values=[1e-9, 5e-9, 10e-9],
        object_type="simulation_params",
    ),
])

# Results in 2 × 3 × 3 = 18 experiments total
handler.run(name_prefix="multi_sweep")
```

### Guideline: Choosing Parameters to Sweep

- **Simulation parameters**: `nphoton`, `tend`, `srcpos`, `srcdir`, `savedetflag`
- **Tissue parameters**: Custom attributes you define (e.g., layer thickness, optical properties)
- **Avoid sweeping**: Immutable objects or parameters with side effects (though TFO-Sim2's design minimizes this)

---

## Understanding Results Storage

The `results_to_store` parameter controls which simulation outputs are saved to disk. This is crucial for ensuring you capture the data you need.

### Available Result Types

The `StorableResults` enum defines what can be stored and should cover most of what's needed.

```python
from tfo_sim2.result_storage import StorableResults

# Available results:
StorableResults.FLUX                  # Fluence/flux distribution
StorableResults.STATISTICS            # Simulation statistics
StorableResults.DETECTOR_ID           # Detector indices for detected photons
StorableResults.NUM_SCATTERINGS       # Scattering counts
StorableResults.PARTIAL_PATH          # Path length in each tissue layer
StorableResults.MOMENTUM              # Momentum transfer
StorableResults.EXIT_POSITION         # Exit location of photons
StorableResults.EXIT_VELOCITY         # Exit direction of photons
StorableResults.INITIAL_WEIGHT        # Initial photon weight
StorableResults.OPTICAL_PROPERTIES    # Optical properties used
StorableResults.UNIT_IN_MM            # Unit conversion factor
StorableResults.DETECTOR_INTENSITY    # Computed detector intensity
StorableResults.SOURCE_POSITION       # Source location used
StorableResults.DETECTOR_POSITIONS    # Detector array positions
```

### Default Behavior

If `results_to_store=None`, the handler defaults to storing:

```python
[StorableResults.FLUX, StorableResults.STATISTICS, StorableResults.SOURCE_POSITION]
```

### Example 1: Storing Only Minimal Data

```python
handler = ExperimentHandler(
    base_simulation_params=sim_params,
    base_tissue_model=tissue_model,
    output_dir="./batch_results/minimal",
    results_to_store=[
        StorableResults.FLUX,
        StorableResults.STATISTICS,
    ],
)
```

This minimizes disk usage, saving only the fluence distribution and simulation statistics.

### Example 2: Complete Detector Data

For studies requiring detector information:

```python
from tfo_sim2.result_storage import StorableResults

detailed_storage = [
    StorableResults.FLUX,
    StorableResults.STATISTICS,
    StorableResults.DETECTOR_ID,
    StorableResults.DETECTOR_INTENSITY,
    StorableResults.NUM_SCATTERINGS,
    StorableResults.PARTIAL_PATH,
    StorableResults.EXIT_POSITION,
    StorableResults.SOURCE_POSITION,
    StorableResults.DETECTOR_POSITIONS,
]

handler = ExperimentHandler(
    base_simulation_params=sim_params,
    base_tissue_model=tissue_model,
    output_dir="./batch_results/detailed",
    results_to_store=detailed_storage,
)
```

### Example 3: Another One For the Road

```python
# For optical property analysis
optical_property_storage = [
    StorableResults.FLUX,
    StorableResults.OPTICAL_PROPERTIES,
    StorableResults.STATISTICS,
]

handler = ExperimentHandler(
    base_simulation_params=sim_params,
    base_tissue_model=tissue_model,
    output_dir="./batch_results/custom",
    results_to_store=optical_property_storage,
)
```

### Storage Formats

Results can be saved in different formats:

```python
from tfo_sim2.result_storage import StorageFormat

# NPZ format (numpy binary - default, fast, efficient)
handler = ExperimentHandler(
    ...,
    storage_format=StorageFormat.NPZ,
)

# HDF5 format (hierarchical, good for large datasets)
handler = ExperimentHandler(
    ...,
    storage_format=StorageFormat.HDF5,
)

# JSON format (human-readable, slower)
handler = ExperimentHandler(
    ...,
    storage_format=StorageFormat.JSON,
)
```

### Accessing Results After Loading

```python
from tfo_sim2.result_storage import ResultStorage

# Load a saved result
storage = ResultStorage.load("batch_results/my_experiment/experiment_0000.npz")

# Access stored data
flux_data = storage.flux
detector_ids = storage.detid
statistics = storage.stat
source_position = storage.srcpos

# All available results are None if not stored
if storage.detint is None:
    print("Detector intensity was not stored in this result")
```

---

## Visualization with Plotters

The `Plotter` generates a plot at the end of each sub-experiment. Its generally a good idea to have a plotter for whatever you want to plot later on. That way you can verify that you are storing the proper results. Ideally, you shouldn't need to write a separate plotting code post-simulations.

### Plotter Architecture

A `Plotter` is an abstract base class that you subclass. Each plotter:

1. **Checks data validity**: Verifies required data exists
2. **Plots**: Generates a visualization on a matplotlib Axes
3. **Reports issues**: Provides troubleshooting information

### Available Plotters

Three built-in plotters are available:

#### 1. DetectorIntensity1DPlotter

Plots detector intensity vs. source-detector distance as a line plot. Useful for understanding light detection as a function of distance.

```python
from tfo_sim2.plotter import DetectorIntensity1DPlotter

# Create plotter (plot_log=True applies log10 scale)
plotter = DetectorIntensity1DPlotter(plot_log=True)
```

**Requirements**: `detpos`, `detint`, and `srcpos` must be in `results_to_store`

#### 2. FluxPlotter

Plots a 2D slice through the 3D flux distribution as a heatmap. Configurable to show different slices along any axis.

```python
from tfo_sim2.plotter import FluxPlotter

# Create plotter
plotter = FluxPlotter(plot_log=True)
```

**Requirements**: `flux` must be in `results_to_store`

#### 3. FluxSlicePlotter

Plots a pre-computed 2D flux slice (useful if you're storing sliced data via `FluxSlice` result type).

```python
from tfo_sim2.plotter import FluxSlicePlotter

# Create plotter
plotter = FluxSlicePlotter(plot_log=True)
```

**Requirements**: `flux_slice` must be in `results_to_store`

### Example 1: Basic Plotting with DetectorIntensity1DPlotter

```python
from tfo_sim2.plotter import DetectorIntensity1DPlotter

# Create a plotter
plotter = DetectorIntensity1DPlotter(plot_log=True)

# Pass to handler
handler = ExperimentHandler(
    base_simulation_params=sim_params,
    base_tissue_model=tissue_model,
    output_dir="./results",
    plotter=plotter,
    results_to_store=[
        StorableResults.FLUX,
        StorableResults.DETECTOR_INTENSITY,
        StorableResults.DETECTOR_POSITIONS,
        StorableResults.SOURCE_POSITION,
    ],
)

handler.run(name_prefix="with_plots")
handler.save_results()
```

This automatically generates `experiment_0000.png`, `experiment_0001.png`, etc.

### Example 2: Using FluxPlotter to Visualize Spatial Distribution

```python
from tfo_sim2.plotter import FluxPlotter

# Create a plotter that shows flux slices
plotter = FluxPlotter(plot_log=True)

handler = ExperimentHandler(
    base_simulation_params=sim_params,
    base_tissue_model=tissue_model,
    output_dir="./results",
    plotter=plotter,
    results_to_store=[
        StorableResults.FLUX,
        StorableResults.STATISTICS,
    ],
    plot_kwargs={
        "slice_along_axis": 2,      # Slice along Z axis
        "slice_axis_value": 30,      # At the middle of the volume
        "time_index": 0,             # First time point
        "cmap": "hot",               # Use hot colormap
    },
)

handler.run(name_prefix="flux_distribution")
handler.save_results()
```

This generates heatmap images showing the 2D spatial distribution of photon fluence at different parameter settings.

### Example 3: Advanced Customization - Plot Appearance

You can customize how the plots look by passing style parameters:

```python
from tfo_sim2.plotter import DetectorIntensity1DPlotter
from tfo_sim2.result_storage import StorableResults

plotter = DetectorIntensity1DPlotter(plot_log=True)

handler = ExperimentHandler(
    base_simulation_params=sim_params,
    base_tissue_model=tissue_model,
    output_dir="./results/styled",
    plotter=plotter,
    results_to_store=[
        StorableResults.DETECTOR_INTENSITY,
        StorableResults.DETECTOR_POSITIONS,
        StorableResults.SOURCE_POSITION,
    ],
    # Customize line appearance
    plot_kwargs={
        "linewidth": 2.5,
        "marker": "o",
        "markersize": 6,
        "color": "navy",
        "alpha": 0.8,
    },
    # Customize figure appearance
    fig_kwargs={
        "figsize": (12, 7),
        "dpi": 300,
    },
)

handler.run(name_prefix="styled")
handler.save_results()
```

This generates publication-ready plots with custom styling applied to all experiments.

### Example 4: Creating Custom Plotters

You can extend the `Plotter` base class to create your own visualization logic:

```python
from tfo_sim2.plotter import Plotter
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt

class CustomFluxHistogramPlotter(Plotter):
    """
    Create a histogram of fluence values to analyze the distribution.
    """
    
    def check_valid_data(self) -> bool:
        """Require flux data."""
        if self.result_storage is None:
            return False
        return self.result_storage.flux is not None
    
    def plot(self, axes: Axes, **kwargs) -> None:
        """Create histogram of flux values."""
        flux = self.result_storage.flux
        
        # Flatten the flux and take first time point
        flux_flat = flux[:, :, :, 0].flatten()
        
        # Create histogram
        axes.hist(
            flux_flat[flux_flat > 0],  # Only positive values
            bins=50,
            edgecolor='black',
            alpha=0.7,
            **kwargs
        )
        axes.set_xlabel("Fluence (W/m²)")
        axes.set_ylabel("Frequency")
        axes.set_title("Fluence Distribution Histogram")
        axes.set_yscale('log')
    
    @property
    def troubleshoot_string(self) -> str:
        return "Flux data must be available to plot histogram."

# Use it
plotter = CustomFluxHistogramPlotter()

handler = ExperimentHandler(
    base_simulation_params=sim_params,
    base_tissue_model=tissue_model,
    output_dir="./results",
    plotter=plotter,
    results_to_store=[StorableResults.FLUX],
)

handler.run(name_prefix="flux_histograms")
handler.save_results()
```

### Plotter Best Practices

- Always implement both `check_valid_data()` and `plot()` methods
- Use `troubleshoot_string` to provide helpful error messages
- Respect `**kwargs` for customization
- Remember that `result_storage` might be `None` initially (set via handler)
- Access data safely with `is not None` checks
- Use matplotlib's colorbar and labels for clarity

---

## Dynamic Parameters: Complex Dependencies

Sometimes parameters interact. For example, detector positions should adapt to tissue depth, or source position should account for tissue thickness. The `DynamicParameter` system handles these complex scenarios.

### Understanding DynamicParameter

A `DynamicParameter` is an abstract class with a single method:

```python
class DynamicParameter(ABC):
    @abstractmethod
    def modify(
        self,
        tissue_model: TissueModel,
        simulation_params: SimulationParameters,
        detector_array: DetectorArray,
    ) -> None:
        """Modify tissue_model, simulation_params, or detector_array based on current state."""
        pass
```

It's called **after** static parameters are applied but **before** simulation runs.

### Example 1: Source Position Based on Tissue Depth

```python
from tfo_sim2.experiment_handler import DynamicParameter

class AdaptiveSourcePositionParameter(DynamicParameter):
    """
    Place the source at a fixed distance above the tissue surface,
    adapting to the tissue depth.
    """
    
    def modify(self, tissue_model, simulation_params, detector_array):
        # Get tissue depth from the model
        tissue_depth = tissue_model.get_thickness()
        
        # Place source 1 mm above the tissue
        current_z = simulation_params.srcpos[2]
        simulation_params.srcpos[2] = tissue_depth + 1.0
        
        print(f"Adapted source Z from {current_z} to {simulation_params.srcpos[2]}")

# Use it
dynamic_params = [AdaptiveSourcePositionParameter()]

handler = ExperimentHandler(
    base_simulation_params=sim_params,
    base_tissue_model=tissue_model,
    output_dir="./results",
    dynamic_parameters=dynamic_params,
)

handler.run()
handler.save_results()
```

### Example 2: Detector Array Repositioning

```python
from tfo_sim2.experiment_handler import DynamicParameter

class ConcurrentDetectorSpacingParameter(DynamicParameter):
    """
    Adjust detector spacing based on simulation wavelength or tissue properties.
    """
    
    def modify(self, tissue_model, simulation_params, detector_array):
        # Example: use tissue optical properties to set spacing
        mua = tissue_model.get_absorption_coefficient()
        
        # Decrease spacing for more absorbing tissue
        spacing = 10.0 / (1.0 + mua * 10)
        
        # Recreate detector array with new spacing
        num_dets = 4
        positions = []
        for i in range(num_dets):
            x = 30 + (i + 1) * spacing
            positions.append([x, 30, 0])
        
        detector_array.set_positions(positions)
        print(f"Repositioned {num_dets} detectors with {spacing:.2f} mm spacing")

# Use it
dynamic_params = [ConcurrentDetectorSpacingParameter()]

handler = ExperimentHandler(
    ...,
    dynamic_parameters=dynamic_params,
)
```

### Example 3: Multiple Dynamic Parameters

```python
class SetSourceBasedOnTissueParameter(DynamicParameter):
    """Set source position based on tissue thickness."""
    def modify(self, tissue_model, simulation_params, detector_array):
        thickness = tissue_model.get_thickness()
        simulation_params.srcpos[2] = thickness + 0.5

class SetDetectorBasedOnOpticalPropertiesParameter(DynamicParameter):
    """Set detector positions based on mean free path."""
    def modify(self, tissue_model, simulation_params, detector_array):
        # Calculate mean free path
        mua = tissue_model.get_absorption_coefficient()
        mus = tissue_model.get_scattering_coefficient()
        mfp = 1.0 / (mua + mus)
        
        # Set detectors at multiples of mean free path
        positions = [
            [30 + mfp * i, 30, 0] for i in range(1, 4)
        ]
        detector_array.set_positions(positions)

class ValidateSetupParameter(DynamicParameter):
    """Final validation that everything is reasonable."""
    def modify(self, tissue_model, simulation_params, detector_array):
        if simulation_params.srcpos[2] < 0:
            raise ValueError("Source position is underground!")
        if len(detector_array.positions) == 0:
            raise ValueError("No detectors configured!")

# Chain them together
dynamic_params = [
    SetSourceBasedOnTissueParameter(),
    SetDetectorBasedOnOpticalPropertiesParameter(),
    ValidateSetupParameter(),
]

handler = ExperimentHandler(
    ...,
    dynamic_parameters=dynamic_params,
)
```

### Dynamic Parameter Execution Order

Dynamic parameters are applied in the order you provide them:

```
1. Static parameter sweeps are applied
2. Dynamic parameter 1 modifies the setup
3. Dynamic parameter 2 modifies the setup
4. Dynamic parameter 3 validates everything
5. Simulation runs with final configuration
```

This order is deterministic and reproducible.

---

## The Operational Sequence

Understanding the exact sequence of operations helps predict behavior and debug issues.

### Sequence Diagram

```
ExperimentHandler.run() called
    │
    ├─> Generate parameter grid
    │   (Cartesian product of all sweep values)
    │
    ├─> For each parameter combination:
    │   │
    │   ├─> Deep copy base_simulation_params
    │   ├─> Deep copy base_tissue_model
    │   ├─> Create copy of detector_array
    │   │
    │   ├─> Apply static parameters from sweep
    │   │   (using dot notation setters)
    │   │
    │   ├─> Apply dynamic parameters in order
    │   │   ├─> dynamic_param[0].modify(...)
    │   │   ├─> dynamic_param[1].modify(...)
    │   │   └─> ...
    │   │
    │   ├─> Create Simulator with final configuration
    │   │
    │   ├─> Run simulation
    │   │   ├─> Simulator generates result_dict
    │   │   └─> result_dict contains raw PMCX output
    │   │
    │   ├─> Create ResultStorage
    │   │   ├─> Extract requested results_to_store
    │   │   └─> Populate ResultStorage attributes
    │   │
    │   ├─> [If plotter provided] Generate visualization
    │   │   ├─> Check data validity
    │   │   ├─> Create matplotlib Figure
    │   │   ├─> Call plotter.plot(axes, **plot_kwargs)
    │   │   ├─> Append to figure_list
    │   │   └─> Close figure
    │   │
    │   └─> Track experiment metadata
    │
    └─> [After all simulations] All results collected

ExperimentHandler.save_results() called
    │
    ├─> For each (ResultStorage, params) pair:
    │   ├─> Save to file (experiment_XXXX.npz/.h5/.json)
    │   └─> Record in parameter_mapping.json
    │
    ├─> For each generated figure:
    │   ├─> Save to PNG file
    │   └─> Link in parameter_mapping.json
    │
    └─> Write parameter_mapping.json with metadata
```

### Critical Details

1. **Deep Copying**: Each experiment starts with fresh copies. Mutations don't leak between runs.

2. **Parameter Application**:
   - Dot notation paths are split: `"optical_props.mua"` → `["optical_props", "mua"]`
   - The handler navigates to the parent object and uses `setattr()`
   - This works with any Python object structure

3. **Dynamic Parameter Timing**:
   - Applied *after* static parameter sweeps
   - Can see and modify state set by previous dynamic parameters
   - Errors in dynamic parameters fail the entire experiment

4. **Result Extraction**:
   - `results_to_store` controls which fields are extracted from PMCX output
   - Fields not in `results_to_store` are discarded to save disk space
   - Fields not available in PMCX output trigger warnings

5. **Visualization**:
   - Each simulation generates one figure (if plotter provided)
   - Figures are saved after *all* simulations complete (not during)
   - Provides clean progress output without file I/O overhead

---

## Complete Example

Here's a realistic, end-to-end example combining all concepts:

```python
"""
Complete example: Studying how photon count affects detector sensitivity
across different tissue depths.
"""

from tfo_sim2 import (
    ExperimentHandler,
    SimulationParameters,
    DetectorArray,
    ParameterSweep,
)
from tfo_sim2.experiment_handler import DynamicParameter
from tfo_sim2.plotter import DetectorIntensity1DPlotter
from tfo_sim2.result_storage import StorableResults, StorageFormat
from tfo_sim2.tissue_model import TissueModel
from pathlib import Path
import numpy as np


# Step 1: Define a custom tissue model
class DepthVariableTissueModel(TissueModel):
    """Tissue model where thickness can be swept."""
    
    def __init__(self, thickness_mm=10):
        super().__init__(name="DepthVariable")
        self.thickness_mm = thickness_mm
        self.dim = 60  # 60x60x60 voxel volume
    
    def _generate_volume(self):
        """Create a layered tissue structure."""
        vol = np.zeros((self.dim, self.dim, self.dim), dtype=np.uint32)
        
        # Layer 1: Epidermis (0-2 mm)
        z_end = min(int(2 / 0.1), self.dim)  # 0.1 mm per voxel
        vol[:, :, :z_end] = 1
        
        # Layer 2: Dermis (2-10 mm, up to thickness_mm)
        z_end2 = min(int(self.thickness_mm / 0.1), self.dim)
        vol[:, :, z_end:z_end2] = 2
        
        self._vol = vol
    
    def _generate_properties(self):
        """Set optical properties for each layer."""
        self._prop = [
            # Background (water)
            [0.0, 0.0, 1.0, 1.33],
            # Epidermis
            [0.01, 10.0, 0.9, 1.4],
            # Dermis
            [0.04, 20.0, 0.85, 1.4],
        ]


# Step 2: Define a dynamic parameter to adapt detectors
class AdaptiveDetectorPlacementParameter(DynamicParameter):
    """Place detectors at positions relative to tissue depth."""
    
    def modify(self, tissue_model, simulation_params, detector_array):
        depth = tissue_model.thickness_mm
        
        # Place detectors at increasing distances, scaled to depth
        positions = []
        for i in range(4):
            x = 30 + (i + 1) * (depth / 3)
            positions.append([x, 30, 0])
        
        # This would require adding set_positions to DetectorArray
        # For now, just show the concept
        print(f"Would place detectors at: {positions}")


# Step 3: Set up base parameters
base_sim_params = SimulationParameters(
    nphoton=500000,
    srcpos=[30, 30, 0],
    tend=5e-9,
    tstep=5e-9,
)

base_tissue = DepthVariableTissueModel(thickness_mm=10)

detector_array = DetectorArray()
# Configure detectors (implementation depends on DetectorArray API)


# Step 4: Create the handler
handler = ExperimentHandler(
    base_simulation_params=base_sim_params,
    base_tissue_model=base_tissue,
    output_dir=Path("./results/sensitivity_study"),
    detector_array=detector_array,
    storage_format=StorageFormat.NPZ,
    results_to_store=[
        StorableResults.FLUX,
        StorableResults.DETECTOR_INTENSITY,
        StorableResults.DETECTOR_ID,
        StorableResults.STATISTICS,
        StorableResults.SOURCE_POSITION,
        StorableResults.DETECTOR_POSITIONS,
    ],
    plotter=DetectorIntensity1DPlotter(plot_log=True),
    plot_kwargs={
        "linewidth": 2,
        "marker": "o",
    },
    fig_kwargs={
        "figsize": (10, 6),
    },
    # dynamic_parameters=[AdaptiveDetectorPlacementParameter()],
)


# Step 5: Add parameter sweeps
handler.add_sweeps([
    ParameterSweep(
        param_path="nphoton",
        values=[100000, 500000, 1000000],
        object_type="simulation_params",
    ),
    ParameterSweep(
        param_path="thickness_mm",
        values=[5, 10, 20],
        object_type="tissue_model",
    ),
])

# Results in 3 × 3 = 9 experiments


# Step 6: Execute
print("Running parameter sweep...")
handler.run(name_prefix="sensitivity")

# Step 7: Save results
print("Saving results and metadata...")
handler.save_results()

# Step 8: Inspect results
summary = handler.get_results_summary()
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"Total experiments: {summary['total_experiments']}")
print(f"Output directory: {summary['output_directory']}")
print(f"Storage format: {summary['storage_format']}")
print(f"Number of sweeps: {summary['number_of_sweeps']}")
print("\nSweep configurations:")
for sweep in summary['sweep_configs']:
    print(f"  - {sweep['param_path']}: {sweep['num_values']} values " +
          f"(from {sweep['object_type']})")
print("="*60)

# Step 9: Verify results
import json
mapping_file = Path("./results/sensitivity_study/parameter_mapping.json")
with open(mapping_file) as f:
    mapping = json.load(f)

print(f"\nFirst experiment parameters:")
first_exp = mapping['experiments'][0]
print(json.dumps(first_exp['sweep_parameters'], indent=2))
```

### Expected Output

```
Running parameter sweep...
Completed experiment 0/9
Completed experiment 1/9
...
Completed experiment 8/9
Saving results and metadata...

============================================================
EXPERIMENT SUMMARY
============================================================
Total experiments: 9
Output directory: ./results/sensitivity_study
Storage format: npz
Number of sweeps: 2

Sweep configurations:
  - nphoton: 3 values (from simulation_params)
  - thickness_mm: 3 values (from tissue_model)
============================================================

First experiment parameters:
{
  "nphoton": {
    "value": 100000,
    "object_type": "simulation_params"
  },
  "thickness_mm": {
    "value": 5,
    "object_type": "tissue_model"
  }
}
```

### What Was Created

```
results/sensitivity_study/
├── experiment_0000.npz
├── experiment_0001.npz
├── ...
├── experiment_0008.npz
├── experiment_0000.png  (from plotter)
├── experiment_0001.png
├── ...
├── experiment_0008.png
└── parameter_mapping.json  (metadata)
```

---

## Summary

The `ExperimentHandler` provides a powerful, declarative interface for batch simulations:

- **Sweeps** define what varies
- **Dynamic Parameters** handle complex dependencies
- **Results Storage** controls what gets saved
- **Plotters** create visualizations
- **Metadata** ensures reproducibility (Hopefully)

By understanding the operational sequence and using the right abstractions, you can orchestrate sophisticated parameter studies efficiently and reproducibly.

