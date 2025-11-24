"""
Example script demonstrating batch experiments with the Lapitan tissue model

This example shows how to:
1. Use the Lapitan tissue model representing realistic skin layers
2. Create multiple parameter sweeps (nphoton and Vb_arterial blood volume)
3. Run a batch of experiments with cross-product parameter combinations
4. Save results with only specific StorableResults (FluxSlice)
5. View the experiment summary and parameter mapping
"""

from pathlib import Path
from tfo_sim2 import (
    ExperimentHandler,
    ParameterSweep,
    SimulationParameters,
    StorageFormat,
    FluxSlice,
    FluxSlicePlotter,
    Statistics,
)
from tfo_sim2.tissue_model_extended import LapitanTissueModel


def main():
    """Run a batch experiment with nphoton and Vb_arterial sweeps on Lapitan tissue model."""

    # ============================================================================
    # Step 1: Define base simulation parameters
    # ============================================================================
    base_sim_params = SimulationParameters(
        nphoton=1000000,  # Will be swept over
        tend=5e-9,
        tstep=5e-9,
        srcpos=[51, 51, 58],  # Source at top surface (skin level)
        srcdir=[0, 0, -1],  # Directed downward into tissue
        outputtype="flux"
    )

    # ============================================================================
    # Step 2: Define base tissue model (Lapitan - realistic skin layers)
    # ============================================================================
    base_tissue = LapitanTissueModel(
        wavelength=660,  # Red light for PPG
        Vb_arterial=0.05,  # Will be swept over
        Vb_venous=0.05,
    )

    # ============================================================================
    # Step 3: Create experiment handler
    # ============================================================================
    output_dir = Path("./batch_results/lapitan_batch_results")

    handler = ExperimentHandler(
        base_simulation_params=base_sim_params,
        base_tissue_model=base_tissue,
        output_dir=output_dir,
        storage_format=StorageFormat.NPZ,
        # Only store FluxSlice results (2D slice through tissue)
        results_to_store=[FluxSlice(2, 57, 0), Statistics()],
        plotter=FluxSlicePlotter(),
    )

    # ============================================================================
    # Step 4: Define parameter sweeps
    # ============================================================================
    # Sweep 1: nphoton values (simulation photon count)
    nphoton_sweep = ParameterSweep(
        param_path="nphoton",
        values=[50000, 100000, 200000],
        object_type="simulation_params",
    )

    # Sweep 2: Vb_arterial (arterial blood volume fraction in dermis)
    # This simulates changes in blood perfusion, relevant for PPG signals
    vb_arterial_sweep = ParameterSweep(
        param_path="Vb_arterial",
        values=[0.045, 0.05, 0.055],  # Low, medium, high blood volume
        object_type="tissue_model",
    )

    # Add both sweeps to the handler
    handler.add_sweeps([nphoton_sweep, vb_arterial_sweep])

    # ============================================================================
    # Step 5: Run experiments
    # ============================================================================
    # This will create 3 × 3 = 9 simulations (cross-product of all combinations)
    print("Running batch experiments with Lapitan tissue model...")
    handler.run(name_prefix="lapitan_ppg")

    # ============================================================================
    # Step 6: Save results
    # ============================================================================
    print("Saving results...")
    handler.save_results()

    # ============================================================================
    # Step 7: Display results summary
    # ============================================================================
    summary = handler.get_results_summary()

    print("\n" + "=" * 70)
    print("LAPITAN TISSUE MODEL - BATCH EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total experiments run: {summary['total_experiments']}")
    print(f"Output directory: {summary['output_directory']}")
    print(f"Storage format: {summary['storage_format']}")
    print(f"Number of sweeps: {summary['number_of_sweeps']}")

    print("\nTissue Model Configuration:")
    print(f"  Model: Lapitan (realistic layered skin tissue)")
    print(f"  Wavelength: 660 nm (red light)")
    print(f"  Layers: Subcutaneous (50mm) → Dermis (7mm) → Epidermis (1mm)")
    print(f"  Source Position: [51, 51, 58] (top surface)")

    print("\nSweep configurations:")
    for i, config in enumerate(summary["sweep_configs"], 1):
        print(f"  {i}. {config['param_path']}")
        print(f"     Object: {config['object_type']}")
        print(f"     Values: {config['num_values']}")

    print(f"\nResults saved to: {output_dir}")
    print(f"Parameter mapping: {output_dir / 'parameter_mapping.json'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
