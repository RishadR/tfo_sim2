"""
Run experiments sweeping nphoton and fetal depth in Dan's 4-layer tissue model.
"""

from pathlib import Path
from tfo_sim2 import (
    ExperimentHandler,
    ParameterSweep,
    SimulationParameters,
    StorageFormat,
    DetectorPositions,
    SourcePosition,
    DetectorIntensity,
    DanModel4LayerX,
    DetectorIntensity1DPlotter,
    DynamicParameter,
    DetectorArray
)

class SourceDynamicParameter(DynamicParameter):
    """Dynamic parameter to vary source position based on dermal thickness."""

    def modify(self, tissue_model, simulation_params, detector_array) -> None:
        assert isinstance(tissue_model, DanModel4LayerX), "Tissue model must be DanModel4LayerX"
        # Change the source position based on the tissue model
        top_z = tissue_model.topmost_pixel()
        simulation_params.srcpos[2] = top_z + 1
        # Overwrite the detector array
        detector_array.clear()
        detector_array.add_detector_line((110, 112, top_z + 1), (110, 210, top_z + 1), 10, 2.0)
        

def main():
    """Run a batch experiment with nphoton and derm_thickness sweeps on Dan's 4-layer tissue model and store
    DetectorIntensity results."""
    
    
    # ============================================================================
    # Step 1: Define base tissue model (Dan's 4-layer model)
    # ============================================================================
    base_tissue = DanModel4LayerX(
        wavelength=650.0,  # Red light for PPG
    )
    
    
    # ============================================================================
    # Step 2: Define base simulation parameters
    # ============================================================================
    base_sim_params = SimulationParameters(
        nphoton=1000000,  # Will be swept over
        tend=5e-9,
        tstep=5e-9,
        srcpos=[110, 110, 50],  # Source at top surface (skin level)
        srcdir=[0, 0, -1],  # Directed downward into tissue
        outputtype="flux",
    )
    
    dynamic_src_param = SourceDynamicParameter()




    # ============================================================================
    # Step 3: Create experiment handler
    # ============================================================================
    output_dir = Path("./batch_results/dan4_nphoton")

    handler = ExperimentHandler(
        base_simulation_params=base_sim_params,
        base_tissue_model=base_tissue,
        output_dir=output_dir,
        detector_array=DetectorArray(),
        dynamic_parameters=[dynamic_src_param],
        storage_format=StorageFormat.NPZ,
        results_to_store=[SourcePosition(), DetectorPositions(), DetectorIntensity()],
        plotter=DetectorIntensity1DPlotter(),
    )

    # ============================================================================
    # Step 4: Define parameter sweeps
    # ============================================================================
    # Sweep 1: nphoton values (simulation photon count)
    nphoton_sweep = ParameterSweep(
        param_path="nphoton",
        values=[int(1e3), int(1e5)],
        object_type="simulation_params",
    )

    # Sweep 2: Vb_arterial (arterial blood volume fraction in dermis)
    # This simulates changes in blood perfusion, relevant for PPG signals
    depth_sweep = ParameterSweep(
        param_path="derm_thickness",
        values=[4, 5],  # Low, medium, high blood volume
        object_type="tissue_model",
    )

    # Add both sweeps to the handler
    handler.add_sweeps([nphoton_sweep, depth_sweep])

    # ============================================================================
    # Step 5: Run experiments
    # ============================================================================
    # This will create 3 × 3 = 9 simulations (cross-product of all combinations)
    print("Running batch experiments with Dan's 4-layer tissue model...")
    handler.run(name_prefix="dan4_nphoton")

    # ============================================================================
    # Step 6: Save results
    # ============================================================================
    print("Saving results...")
    handler.save_results()

    # ============================================================================
    # Step 7: Display results summary
    # ============================================================================
    summary = handler.get_results_summary()
    print("\n" + "="*70)
    print("Batch Experiment Summary:")
    print(summary)
    print("="*70 + "\n")



if __name__ == "__main__":
    main()
