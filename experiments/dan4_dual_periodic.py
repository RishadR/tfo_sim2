"""
Using a periodic arterial volume to emulate pulsation
"""

from typing import List
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
    DetectorArray,
    DynamicParameter,
    TissueModel,
)


class DanModel4LayerXMerged(DanModel4LayerX):
    """
    An altered version of DanModel4LayerX that takes a single time step as the input. Computes maternal and fetal
    Hb Concentration based on the given time. This simulates a dual pulse with different frequencies.
    """

    def __init__(self, wavelength: float, time: float, **kwargs):
        super().__init__(wavelength, **kwargs)
        self.time = time

    @property
    def prop(self) -> List[List]:
        """Get the optical properties."""
        # Regenerate the properties based on the current time step
        maternal_fundamental = 0.375 * np.sin(2 * np.pi * 1 * self.time)
        maternal_harmonic = 0.375 / 1.5 * np.sin(2 * np.pi * 2 * self.time)
        fetal_fundamental = 0.375 * np.sin(2 * np.pi * 2.2 * self.time)
        fetal_harmonic = 0.375 / 1.5 * np.sin(2 * np.pi * 4.4 * self.time)

        self.maternal_hb_conc = 15.0 + maternal_fundamental + maternal_harmonic
        self.fetal_hb_conc = 15.0 + fetal_fundamental + fetal_harmonic
        self._generate_properties()
        if self._prop is not None:
            return self._prop
        else:
            raise ValueError("Could not generate optical properties!")


def main():
    """Run a batch experiment with nphoton and Vb_arterial sweeps on Lapitan tissue model."""
    time_axis = np.linspace(0, 4.0, 80)


    base_tissue = DanModel4LayerXMerged(
        wavelength=650.0,  # Red light for PPG
        time = 0.0  # Will Iterate this
    )
    topmost = base_tissue.topmost_pixel()

    base_sim_params = SimulationParameters(
        nphoton=int(1e8),
        tend=5e-3,
        tstep=5e-3,
        srcpos=[110, 110, topmost + 1],  # Source at top surface (skin level)
        srcdir=[0, 0, -1],  # Directed downward into tissue
        outputtype="flux",
        savedetflag="dp",  # Save detector IDs and partial path
    )

    detector_array = DetectorArray()
    detector_array.add_detector_line(
        start=(110, 130, topmost + 1),
        end=(110, 210, topmost + 1),
        num_detectors=5,
        radius=2.0,
    )

    output_dir = Path("./batch_results/dan4_dual_periodic")

    handler = ExperimentHandler(
        base_simulation_params=base_sim_params,
        base_tissue_model=base_tissue,
        output_dir=output_dir,
        storage_format=StorageFormat.NPZ,
        results_to_store=[DetectorIntensity(), SourcePosition(), DetectorPositions()],
        plotter=None,
        detector_array=detector_array,
        plot_kwargs={"marker": "o", "linestyle": "-"},
    )

    time_sweep = ParameterSweep(
        param_path="time",
        values=time_axis.tolist(),
        object_type="tissue_model",
    )

    # Add both sweeps to the handler
    handler.add_sweep(time_sweep)

    # ============================================================================
    # Run experiments
    # ============================================================================
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
