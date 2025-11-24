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
    DetectorIntensity1DPlotter,
    DanModel4LayerX,
    DetectorArray
)


def main():
    """Run a batch experiment with nphoton and Vb_arterial sweeps on Lapitan tissue model."""
    time_axis = np.linspace(0, 4.0, 40)
    fundamental = 0.375 * np.sin(2 * np.pi * time_axis)
    harmonic = 0.375 / 1.5 * np.sin(4 * np.pi * time_axis)
    periodic_vb_arterial = 15.0 + fundamental + harmonic

    base_tissue = DanModel4LayerX(
        wavelength=650.0,  # Red light for PPG
    )
    topmost = base_tissue.topmost_pixel()
    
    
    base_sim_params = SimulationParameters(
        nphoton= int(1e8),  
        tend=5e-3,
        tstep=5e-3,
        srcpos=[110, 110, topmost + 1],  # Source at top surface (skin level)
        srcdir=[0, 0, -1],  # Directed downward into tissue
        outputtype="flux",
        savedetflag = 'dp'  # Save detector IDs and partial path
    )

    
    detector_array = DetectorArray()
    detector_array.add_detector_line(
        start = (110, 130, topmost + 1),
        end = (110, 210, topmost + 1),
        num_detectors = 5,
        radius = 2.0
    )

    output_dir = Path("./batch_results/dan4_single_periodic")

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

    vb_arterial_sweep = ParameterSweep(
        param_path="fetal_hb_conc",
        values=periodic_vb_arterial.tolist(),
        object_type="tissue_model",
    )

    # Add both sweeps to the handler
    handler.add_sweep(vb_arterial_sweep)

    # ============================================================================
    # Run experiments
    # ============================================================================
    # This will create 3 × 3 = 9 simulations (cross-product of all combinations)
    print("Running batch experiments with Dan4 tissue model...")
    handler.run(name_prefix="dan4_ppg")

    # ============================================================================
    # Step 6: Save results
    # ============================================================================
    print("Saving results...")
    handler.save_results()

    # ============================================================================
    # Step 7: Display results summary
    # ============================================================================
    summary = handler.get_results_summary()



if __name__ == "__main__":
    main()
