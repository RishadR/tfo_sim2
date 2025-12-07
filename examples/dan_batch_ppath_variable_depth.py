"""
Get the Partial Pathlengths (PPATH) results for Dan's 4-layer tissue model at different wavelengths
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
    DynamicParameter,
    DetectorArray,
    PartialPath
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
        wavelength=735.0,  # Red light for PPG
    )
    
    
    # ============================================================================
    # Step 2: Define base simulation parameters
    # ============================================================================
    base_sim_params = SimulationParameters(
        nphoton=int(1e4),  # Will be swept over
        tend=5e-6,
        tstep=5e-6,
        srcpos=[110, 110, 50],  # Source at top surface (skin level)
        srcdir=[0, 0, -1],  # Directed downward into tissue
        outputtype="flux",
    )
    
    dynamic_src_param = SourceDynamicParameter()

    # ============================================================================
    # Step 3: Create experiment handler
    # ============================================================================
    output_dir = Path("./batch_results/dan4_ppath/")

    handler = ExperimentHandler(
        base_simulation_params=base_sim_params,
        base_tissue_model=base_tissue,
        output_dir=output_dir,
        detector_array=DetectorArray(),
        dynamic_parameters=[dynamic_src_param],
        storage_format=StorageFormat.NPZ,
        results_to_store=[SourcePosition(), DetectorPositions(), PartialPath()],
    )

    # ============================================================================
    # Step 4: Define parameter sweeps
    # ============================================================================
    derm_thicknesses = [3, 8, 13, 18, 23]  # in mm
    handler.add_sweep(ParameterSweep("derm_thickness", derm_thicknesses, 'tissue_model'))

    # ============================================================================
    # Step 5: Run experiments
    # ============================================================================
    print("Running batch experiments with Dan's 4-layer tissue model...")
    handler.run(name_prefix="dan4_ppath")

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
