"""
Example demonstrating how to use the TFO Sim2 wrapper library.

This example shows how to:
1. Define tissue models (uniform, layered, shape-based)
2. Configure simulation parameters
3. Set up detectors
4. Run single and batch simulations
5. Plot and save results
"""

import numpy as np
import os

print(os.environ.get("PYTHONPATH"))


from tfo_sim2 import (
    UniformTissueModel,
    LayeredTissueModel,
    SimulationParameters,
    DetectorArray,
    Simulator,
    Plotter,
    ExperimentHandler,
    PlotType,
    StorageFormat,
    BatchOperation,
    BatchParameterType,
)


def example_1_simple_uniform_tissue():
    """Example 1: Simple uniform tissue simulation."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Uniform Tissue Simulation")
    print("=" * 60)

    # Create a uniform tissue model
    tissue = UniformTissueModel(
        size=(60, 60, 60),
        mua=0.005,  # absorption coefficient
        mus=1.0,  # scattering coefficient
        g=0.01,  # anisotropy
        n=1.37,  # refractive index
        name="UniformTissue",
    )

    # Create simulation parameters
    sim_params = SimulationParameters(
        nphoton=1000000,
        tend=5e-9,
        srcpos=[30, 30, 0],
        srcdir=[0, 0, 1],
    )

    # Create detector array
    detectors = DetectorArray(name="SingleDetector")
    detectors.add_detector(position=(30, 30, 50), radius=1.0)

    # Create simulator and run
    simulator = Simulator(tissue, sim_params, detectors)
    result = simulator.run()

    print(f"Simulation completed!")
    print(f"Flux shape: {result['flux'].shape}")
    print(
        f"Detected photons: {len(result['detp']['detid']) if 'detp' in result else 'N/A'}"
    )


def example_2_layered_tissue():
    """Example 2: Layered tissue with different optical properties."""
    print("\n" + "=" * 60)
    print("Example 2: Layered Tissue Simulation")
    print("=" * 60)

    # Define layers
    layers = [
        {
            "z_start": 0,
            "z_end": 20,
            "mua": 0.005,
            "mus": 0.5,
            "g": 0.01,
            "n": 1.37,
        },
        {
            "z_start": 20,
            "z_end": 40,
            "mua": 0.01,
            "mus": 1.0,
            "g": 0.01,
            "n": 1.37,
        },
        {
            "z_start": 40,
            "z_end": 60,
            "mua": 0.002,
            "mus": 0.2,
            "g": 0.01,
            "n": 1.37,
        },
    ]

    tissue = LayeredTissueModel(size=(60, 60, 60), layers=layers, name="LayeredTissue")

    # Create simulation parameters
    sim_params = SimulationParameters(
        nphoton=500000,
        tend=5e-9,
    )

    # Create detector array with multiple detectors
    detectors = DetectorArray(name="LinearDetectorArray")
    detectors.add_detector_line(
        start=(30, 30, 50),
        end=(30, 30, 55),
        num_detectors=5,
    )

    # Run simulation
    simulator = Simulator(tissue, sim_params, detectors)
    result = simulator.run()

    print(f"Simulation completed!")
    print(f"Number of detectors: {len(detectors)}")
    print(f"Result keys: {list(result.keys())}")


def example_4_experiment_handler_batch():
    """Example 4: Using ExperimentHandler for batch operations."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Experiments with ExperimentHandler")
    print("=" * 60)

    # Create experiment handler
    handler = ExperimentHandler(
        name="BatchExperiment",
        output_dir="./output/batch_example",
    )

    # Set tissue model
    tissue = UniformTissueModel(
        size=(60, 60, 60),
        mua=0.005,
        mus=1.0,
        g=0.01,
        n=1.37,
    )
    handler.set_tissue_model(tissue)

    # Set simulation parameters
    sim_params = SimulationParameters(
        nphoton=500000,
        tend=5e-9,
    )
    handler.set_simulation_params(sim_params)

    # Set detectors
    detectors = DetectorArray()
    detectors.add_detector(position=(30, 30, 50), radius=1.0)
    handler.set_detectors(detectors)

    # Define batch operation: vary number of photons
    batch_op = BatchOperation(
        param_type=BatchParameterType.SIMULATION_PARAM,
        param_name="nphoton",
        values=[100000, 500000, 1000000],
        description="photon_count_sweep",
    )

    # Run batch
    print("Running batch simulations...")
    results = handler.run_batch([batch_op], save=True, save_format=StorageFormat.JSON)

    print(f"Batch completed!")
    print(f"Number of simulations: {len(results)}")
    print(f"Results stored: {handler.list_results()}")
    print(f"Output directory: {handler.output_dir}")


def example_5_results_and_plotting():
    """Example 5: Storing, loading, and plotting results."""
    print("\n" + "=" * 60)
    print("Example 5: Result Storage and Visualization")
    print("=" * 60)

    from tfo_sim2 import ResultStorage

    # Create a simple result (simulated)
    result_data = {
        "flux": np.random.rand(60, 60, 60),
        "detp": {
            "detid": np.array([1, 1, 2, 2, 1]),
            "p": np.random.rand(5, 3) * 60,
            "ppath": np.random.rand(5, 2) * 100,
        },
        "stat": {"nphoton": 1000000},
    }

    # Store result
    storage = ResultStorage(name="ExampleResult")
    storage.store_result(result_data)
    storage.set_metadata("tissue_name", "Example Tissue")

    # Save to different formats
    output_dir = Path("./output/result_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON format
    storage.save(
        str(output_dir / "result_json.json"), StorageFormat.JSON, compression="zlib"
    )
    print(f"Saved to JSON with compression")

    # NPZ format
    storage.save(str(output_dir / "result_npz.npz"), StorageFormat.NPZ)
    print(f"Saved to NPZ format")

    # Create plotter
    plotter = Plotter()

    # Plot fluence slice
    print("Creating fluence slice plot...")
    plotter.plot(result_data, PlotType.FLUENCE_SLICE, slice_index=30, axis=2)

    # Plot detected positions (if data available)
    if "detp" in result_data:
        print("Creating detected positions plot...")
        plotter.plot(result_data, PlotType.DETECTED_POSITIONS)


def example_6_custom_plots():
    """Example 6: Creating custom plots."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Plotting Functions")
    print("=" * 60)

    # Define a custom plot function
    def custom_energy_distribution(result, **kwargs):
        """Plot total energy distribution."""
        import matplotlib.pyplot as plt

        flux = result.get("flux")
        if flux is not None:
            fig, ax = plt.subplots()
            energy_per_z = np.sum(flux, axis=(0, 1))
            ax.plot(energy_per_z)
            ax.set_xlabel("Z coordinate")
            ax.set_ylabel("Total Energy")
            ax.set_title("Energy Distribution along Z")
            plt.show()

    plotter = Plotter()
    plotter.register_custom_plot("energy_dist", custom_energy_distribution)

    # Create sample result
    result_data = {
        "flux": np.random.rand(60, 60, 60),
    }

    # Plot with custom function
    print("Creating custom energy distribution plot...")
    plotter.plot(result_data, PlotType.CUSTOM, name="energy_dist")


def example_7_detector_patterns():
    """Example 7: Various detector array patterns."""
    print("\n" + "=" * 60)
    print("Example 7: Detector Array Patterns")
    print("=" * 60)

    # Example 1: Linear detector array
    detectors1 = DetectorArray(name="LinearDetectors")
    detectors1.add_detector_line(
        start=(10, 30, 50),
        end=(50, 30, 50),
        num_detectors=5,
    )
    print(f"Linear array: {len(detectors1)} detectors")

    # Example 2: Grid detector array
    detectors2 = DetectorArray(name="GridDetectors")
    detectors2.add_detector_grid(
        x_range=(20, 40),
        y_range=(20, 40),
        z=50,
        nx=3,
        ny=3,
    )
    print(f"Grid array: {len(detectors2)} detectors")

    # Example 3: Circular detector array
    detectors3 = DetectorArray(name="CircularDetectors")
    detectors3.add_detector_circle(
        center=(30, 30, 30),
        plane_normal=(0, 0, 1),
        radius_circle=15,
        num_detectors=8,
    )
    print(f"Circular array: {len(detectors3)} detectors")


if __name__ == "__main__":
    from pathlib import Path

    print("\n" + "=" * 60)
    print("TFO Sim2 - PMCX Wrapper Library Examples")
    print("=" * 60)

    # Note: These examples are designed to be informative.
    # Some require PMCX to be installed to actually run.

    try:
        # Run examples that don't require actual PMCX simulation
        example_7_detector_patterns()

        # These would require PMCX to be installed:
        # example_1_simple_uniform_tissue()
        # example_2_layered_tissue()
        # example_3_shape_based_tissue()
        # example_4_experiment_handler_batch()
        # example_5_results_and_plotting()
        # example_6_custom_plots()

        print("\n" + "=" * 60)
        print("Examples completed! (Note: PMCX-dependent examples commented)")
        print("=" * 60)
        print("\nTo run all examples, install PMCX:")
        print("  pip install pmcx")
        print("\nFor result visualization, install:")
        print("  pip install matplotlib numpy")
        print("\nFor advanced result storage:")
        print("  pip install jdata h5py")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
