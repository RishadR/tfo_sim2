"""
Unit tests for the simulator functionality.

These tests verify that the Simulator class can successfully run PMCX simulations
and produce expected output.
"""

import pytest
import numpy as np

from tfo_sim2 import (
    Simulator,
    UniformTissueModel,
    SimulationParameters,
    DetectorArray,
)


class TestSimulatorCreation:
    """Test simulator initialization."""

    def test_simulator_creation(self):
        """Test creating a simulator with basic components."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters()
        detectors = DetectorArray()

        simulator = Simulator(tissue, params, detectors)

        assert simulator.tissue_model == tissue
        assert simulator.simulation_params == params
        assert len(simulator.detectors) == len(detectors)

    def test_simulator_creation_without_detectors(self):
        """Test creating a simulator without explicit detectors."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters()

        simulator = Simulator(tissue, params)

        assert simulator.tissue_model == tissue
        assert simulator.simulation_params == params
        assert simulator.detectors is not None


class TestSimulatorCfgGeneration:
    """Test PMCX configuration generation."""

    def test_build_cfg(self):
        """Test building a PMCX configuration dictionary."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters(nphoton=100000)
        detectors = DetectorArray()
        detectors.add_detector((30, 30, 50))

        simulator = Simulator(tissue, params, detectors)
        cfg = simulator.get_cfg()

        assert "vol" in cfg
        assert "prop" in cfg
        assert "nphoton" in cfg
        assert "detpos" in cfg
        assert cfg["nphoton"] == 100000

    def test_cfg_with_no_detectors(self):
        """Test cfg generation when no detectors are added."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters()
        detectors = DetectorArray()  # Empty detector array

        simulator = Simulator(tissue, params, detectors)
        cfg = simulator.get_cfg()

        assert "vol" in cfg
        assert "prop" in cfg
        assert cfg["issavedet"] == 0  # No detectors

    def test_cfg_with_detectors(self):
        """Test cfg generation when detectors are present."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters()
        detectors = DetectorArray()
        detectors.add_detector((30, 30, 50))
        detectors.add_detector((40, 30, 50))

        simulator = Simulator(tissue, params, detectors)
        cfg = simulator.get_cfg()

        assert cfg["issavedet"] == 1
        assert "detpos" in cfg
        assert len(cfg["detpos"]) == 2


class TestSimulatorRun:
    """Test running simulations."""

    def test_run_with_detectors(self):
        """Test running a simulation with detectors produces flux."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters(nphoton=100000, tend=5e-9)
        detectors = DetectorArray()
        detectors.add_detector((30, 30, 50), radius=2.0)

        simulator = Simulator(tissue, params, detectors)
        result = simulator.run()

        # Verify result is a dictionary
        assert isinstance(result, dict), "Result should be a dictionary"

        # Verify flux is present (detp may not be present if detector is off interface)
        assert "flux" in result, "Result should contain 'flux' key"

        # Verify flux data
        flux = result["flux"]
        assert isinstance(flux, np.ndarray), "Flux should be a numpy array"
        assert flux.ndim >= 3, "Flux should be 3D or higher"
        assert np.all(flux >= 0), "Flux values should be non-negative"
        assert np.any(flux > 0), "Some flux values should be non-zero"

    def test_run_without_detectors(self):
        """Test running a simulation without detectors produces flux."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters(nphoton=100000, tend=5e-9)
        detectors = DetectorArray()  # Empty detector array

        simulator = Simulator(tissue, params, detectors)
        result = simulator.run()

        # Verify result is a dictionary
        assert isinstance(result, dict), "Result should be a dictionary"

        # Verify flux is present
        assert "flux" in result, "Result should contain 'flux' key"

        # Verify detp is NOT present when no detectors
        assert (
            "detp" not in result
        ), "Result should NOT contain 'detp' key when no detectors are present"

        # Verify flux data
        flux = result["flux"]
        assert isinstance(flux, np.ndarray), "Flux should be a numpy array"
        assert flux.ndim >= 3, "Flux should be 3D or higher"
        assert np.all(flux >= 0), "Flux values should be non-negative"
        assert np.any(flux > 0), "Some flux values should be non-zero"


class TestSimulatorGetLastResult:
    """Test retrieving the last simulation result."""

    def test_get_last_result_after_run(self):
        """Test getting the last result after running a simulation."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters(nphoton=100000)
        detectors = DetectorArray()
        detectors.add_detector((30, 30, 50))

        simulator = Simulator(tissue, params, detectors)

        # Before running, last result should be None
        assert simulator.get_last_result() is None

        # Run simulation
        result = simulator.run()

        # After running, should be able to get the result
        last_result = simulator.get_last_result()
        assert last_result is not None
        assert last_result == result

    def test_get_last_result_without_running(self):
        """Test that get_last_result returns None before running."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters()

        simulator = Simulator(tissue, params)

        assert simulator.get_last_result() is None


class TestSimulatorComponents:
    """Test changing simulator components."""

    def test_set_tissue_model(self):
        """Test changing the tissue model."""
        tissue1 = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        tissue2 = UniformTissueModel(
            size=(60, 60, 60), mua=0.01, mus=2.0, g=0.01, n=1.37
        )
        params = SimulationParameters()

        simulator = Simulator(tissue1, params)
        assert simulator.tissue_model == tissue1

        simulator.set_tissue_model(tissue2)
        assert simulator.tissue_model == tissue2

    def test_set_simulation_params(self):
        """Test changing simulation parameters."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params1 = SimulationParameters(nphoton=100000)
        params2 = SimulationParameters(nphoton=500000)

        simulator = Simulator(tissue, params1)
        assert simulator.simulation_params.nphoton == 100000

        simulator.set_simulation_params(params2)
        assert simulator.simulation_params.nphoton == 500000

    def test_set_detectors(self):
        """Test changing the detector array."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )
        params = SimulationParameters()
        det1 = DetectorArray()
        det1.add_detector((30, 30, 50))
        det2 = DetectorArray()
        det2.add_detector((20, 20, 50))
        det2.add_detector((40, 40, 50))

        simulator = Simulator(tissue, params, det1)
        assert len(simulator.detectors) == 1

        simulator.set_detectors(det2)
        assert len(simulator.detectors) == 2


class TestSimulatorComparison:
    """Compare simulation results with and without detectors."""

    def test_flux_always_present(self):
        """Test that flux is always present in results regardless of detectors."""
        tissue = UniformTissueModel(
            size=(60, 60, 60), mua=0.005, mus=1.0, g=0.01, n=1.37
        )

        # With detectors
        params_with = SimulationParameters(nphoton=50000, tend=5e-9)
        detectors = DetectorArray()
        detectors.add_detector((30, 30, 50))
        simulator_with = Simulator(tissue, params_with, detectors)
        result_with = simulator_with.run()

        # Without detectors
        params_without = SimulationParameters(nphoton=50000, tend=5e-9)
        simulator_without = Simulator(tissue, params_without, DetectorArray())
        result_without = simulator_without.run()

        # Both should have flux
        assert "flux" in result_with, "Result with detectors should have flux"
        assert (
            "flux" in result_without
        ), "Result without detectors should have flux"

        # Only the one without detectors should NOT have detp key
        assert (
            "detp" not in result_without
        ), "Result without detectors should not have detp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
