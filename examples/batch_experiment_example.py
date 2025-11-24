"""
Example script demonstrating how to set up and run a batch experiment

This example shows how to:
1. Create base simulation parameters and tissue model
2. Define multiple parameter sweeps (nphoton and tend)
3. Run a batch of experiments with cross-product parameter combinations
4. Save results with only specific StorableResults (FluxSlice)
5. View the experiment summary and parameter mapping
"""

from pathlib import Path
from tfo_sim2 import (
    ExperimentHandler,
    ParameterSweep,
    SimulationParameters,
    UniformTissueModel,
    StorageFormat,
    FluxSlice,
    FluxSlicePlotter,
)


def main():
    """Run a batch experiment with nphoton and tend sweeps."""
    
    # ============================================================================
    # Step 1: Define base simulation parameters
    # ============================================================================
    base_sim_params = SimulationParameters(
        nphoton=10000,  # Will be swept over
        tend=5e-9,      # Will be swept over
        tstep=1e-9,
        srcpos=[30, 30, 30],
        srcdir=[0, 0, -1],
    )
    
    # ============================================================================
    # Step 2: Define base tissue model
    # ============================================================================
    base_tissue = UniformTissueModel(
        size=(60, 60, 30),
        mua=0.05,
        mus=1.0,
        g=0.98,
        n=1.37,
    )
    
    # ============================================================================
    # Step 3: Create experiment handler
    # ============================================================================
    output_dir = Path("./batch_results/example1/")
    
    handler = ExperimentHandler(
        base_simulation_params=base_sim_params,
        base_tissue_model=base_tissue,
        output_dir=output_dir,
        storage_format=StorageFormat.NPZ,
        # Only store FluxSlice results (not full flux, statistics, etc.)
        results_to_store=[FluxSlice(2, 20, 0)],
        plotter=FluxSlicePlotter(plot_log=True),
    )
    
    # ============================================================================
    # Step 4: Define parameter sweeps
    # ============================================================================
    # Sweep 1: nphoton values
    nphoton_sweep = ParameterSweep(
        param_path="nphoton",
        values=[5e7, 1e8],
        object_type="simulation_params",
    )
    
    # Sweep 2: tend values (simulation end time)
    tend_sweep = ParameterSweep(
        param_path="tend",
        values=[1e-9, 2e-9],
        object_type="simulation_params",
    )
    
    # Add both sweeps to the handler
    handler.add_sweeps([nphoton_sweep, tend_sweep])
    
    # ============================================================================
    # Step 5: Run experiments
    # ============================================================================
    # This will create 3 × 3 = 9 simulations (cross-product of all combinations)
    print("Running batch experiments...")
    handler.run(name_prefix="flux_sweep")
    
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
    print("BATCH EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total experiments run: {summary['total_experiments']}")
    print(f"Output directory: {summary['output_directory']}")
    print(f"Storage format: {summary['storage_format']}")
    print(f"Number of sweeps: {summary['number_of_sweeps']}")
    
    print("\nSweep configurations:")
    for i, config in enumerate(summary['sweep_configs'], 1):
        print(f"  {i}. {config['param_path']}")
        print(f"     Object: {config['object_type']}")
        print(f"     Values: {config['num_values']}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Parameter mapping: {output_dir / 'parameter_mapping.json'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
