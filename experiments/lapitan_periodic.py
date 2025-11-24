"""
Using a periodic arterial volume to emulate pulsation
"""

from pathlib import Path
import numpy as np
from tfo_sim2 import (
    ExperimentHandler,
    ParameterSweep,
    SimulationParameters,
    StorageFormat,
    Statistics,
    DetectorIntensity,
    SourcePosition,
    DetectorPositions,
    DetectorIntensity1DPlotter
)
from tfo_sim2.detectors import DetectorArray
from tfo_sim2.tissue_model_extended import LapitanTissueModel


def main():
    """Run a batch experiment with nphoton and Vb_arterial sweeps on Lapitan tissue model."""

    # ============================================================================
    # Step 0: Define periodic Vb_arterial function
    # ============================================================================
    time_axis = np.linspace(0, 1.0, 40)
    fundamental = 0.069 * np.sin(2 * np.pi * time_axis)
    harmonic = 0.069 / 1.5 * np.sin(4 * np.pi * time_axis)
    periodic_vb_arterial = 0.05 + fundamental + harmonic

    # ============================================================================
    # Step 1: Define base simulation parameters
    # ============================================================================
    base_sim_params = SimulationParameters(
        nphoton= int(1e8),  
        tend=5e-3,
        tstep=5e-3,
        srcpos=[51, 51, 58],  # Source at top surface (skin level)
        srcdir=[0, 0, -1],  # Directed downward into tissue
        outputtype="flux",
        savedetflag = 'dp'  # Save detector IDs and partial path
    )

    # ============================================================================
    # Step 2: Define base tissue model (Lapitan - realistic skin layers) & Detectors
    # ============================================================================
    base_tissue = LapitanTissueModel(
        wavelength=660,  # Red light for PPG
        Vb_arterial=0.05,  # Will be swept over
        Vb_venous=0.05,
    )
    
    detector_array = DetectorArray()
    detector_array.add_detector_line(
        start = (51, 53, 58),
        end = (51, 62, 58),
        num_detectors = 5,
        radius = 2.0
    )

    # ============================================================================
    # Step 3: Create experiment handler
    # ============================================================================
    output_dir = Path("./batch_results/lapitan_periodic_ppg")

    handler = ExperimentHandler(
        base_simulation_params=base_sim_params,
        base_tissue_model=base_tissue,
        output_dir=output_dir,
        storage_format=StorageFormat.NPZ,
        results_to_store=[DetectorIntensity(), SourcePosition(), DetectorPositions()],
        plotter=DetectorIntensity1DPlotter(),
        detector_array=detector_array,
        plot_kwargs={"marker": "o", "linestyle": "-"}
    )

    # ============================================================================
    # Step 4: Define parameter sweeps
    # ============================================================================
    # Sweep : Vb_arterial (arterial blood volume fraction in dermis)
    # This simulates changes in blood perfusion, relevant for PPG signals
    vb_arterial_sweep = ParameterSweep(
        param_path="Vb_arterial",
        values=periodic_vb_arterial.tolist(),
        object_type="tissue_model",
    )

    # Add both sweeps to the handler
    handler.add_sweep(vb_arterial_sweep)

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
