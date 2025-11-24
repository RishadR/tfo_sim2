"""
Unit tests for experiment handler functionality.

These tests verify batch experiment execution, parameter sweeps, and result storage.
"""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np

from tfo_sim2 import (
    ExperimentHandler,
    ParameterSweep,
    SimulationParameters,
    UniformTissueModel,
    StorageFormat,
)


class TestParameterSweep:
    """Test ParameterSweep configuration."""

    def test_sweep_creation_simulation_params(self):
        """Test creating a parameter sweep for simulation parameters."""
        sweep = ParameterSweep(
            param_path="nphoton",
            values=[1000, 5000, 10000],
            object_type="simulation_params"
        )
        
        assert sweep.param_path == "nphoton"
        assert sweep.values == [1000, 5000, 10000]
        assert sweep.object_type == "simulation_params"

    def test_sweep_creation_tissue_model(self):
        """Test creating a parameter sweep for tissue model."""
        sweep = ParameterSweep(
            param_path="mua",
            values=[0.005, 0.01, 0.02],
            object_type="tissue_model"
        )
        
        assert sweep.param_path == "mua"
        assert sweep.values == [0.005, 0.01, 0.02]
        assert sweep.object_type == "tissue_model"

    def test_sweep_invalid_object_type(self):
        """Test that invalid object type raises error."""
        with pytest.raises(ValueError, match="object_type must be"):
            ParameterSweep(
                param_path="nphoton",
                values=[1000],
                object_type="invalid"
            )

    def test_sweep_empty_values(self):
        """Test that empty values list raises error."""
        with pytest.raises(ValueError, match="values list cannot be empty"):
            ParameterSweep(
                param_path="nphoton",
                values=[],
                object_type="simulation_params"
            )


class TestExperimentHandlerInitialization:
    """Test ExperimentHandler initialization."""

    def test_handler_creation(self):
        """Test creating an experiment handler."""
        sim_params = SimulationParameters(nphoton=10000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
                storage_format=StorageFormat.NPZ,
            )
            
            assert handler.base_simulation_params == sim_params
            assert handler.base_tissue_model == tissue
            assert Path(tmpdir).exists()

    def test_handler_output_dir_creation(self):
        """Test that output directory is created automatically."""
        sim_params = SimulationParameters(nphoton=10000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output"
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=output_dir,
            )
            
            assert output_dir.exists()


class TestParameterGrid:
    """Test parameter grid generation."""

    def test_single_simulation_sweep(self):
        """Test generating grid with single simulation parameter sweep."""
        sim_params = SimulationParameters(nphoton=10000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            sweep = ParameterSweep(
                param_path="nphoton",
                values=[5000, 10000, 15000],
                object_type="simulation_params"
            )
            handler.add_sweep(sweep)
            
            grid = handler._generate_parameter_grid()
            assert len(grid) == 3

    def test_cross_product_sweeps(self):
        """Test that multiple sweeps create cross-product combinations."""
        sim_params = SimulationParameters(nphoton=10000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            # Two simulation sweeps: 2 values × 2 values = 4 combinations
            sweep1 = ParameterSweep(
                param_path="nphoton",
                values=[5000, 10000],
                object_type="simulation_params"
            )
            sweep2 = ParameterSweep(
                param_path="tend",
                values=[1e-9, 5e-9],
                object_type="simulation_params"
            )
            handler.add_sweeps([sweep1, sweep2])
            
            grid = handler._generate_parameter_grid()
            assert len(grid) == 4

    def test_mixed_tissue_and_simulation_sweeps(self):
        """Test parameter grid with both tissue and simulation sweeps."""
        sim_params = SimulationParameters(nphoton=10000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            # 1 tissue sweep × 2 simulation sweeps = 2 combinations
            tissue_sweep = ParameterSweep(
                param_path="mua",
                values=[0.005, 0.01],
                object_type="tissue_model"
            )
            sim_sweep = ParameterSweep(
                param_path="nphoton",
                values=[5000, 10000],
                object_type="simulation_params"
            )
            handler.add_sweeps([tissue_sweep, sim_sweep])
            
            grid = handler._generate_parameter_grid()
            # 2 mua values × 2 nphoton values = 4 combinations
            assert len(grid) == 4


class TestExperimentExecution:
    """Test running experiments."""

    def test_run_single_experiment(self):
        """Test running a single experiment."""
        sim_params = SimulationParameters(nphoton=5000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            sweep = ParameterSweep(
                param_path="nphoton",
                values=[5000],
                object_type="simulation_params"
            )
            handler.add_sweep(sweep)
            handler.run()
            
            assert len(handler.results) == 1
            assert len(handler.experiments) == 1

    def test_run_multiple_experiments(self):
        """Test running multiple experiments."""
        sim_params = SimulationParameters(nphoton=5000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            sweep = ParameterSweep(
                param_path="nphoton",
                values=[5000, 10000],
                object_type="simulation_params"
            )
            handler.add_sweep(sweep)
            handler.run()
            
            assert len(handler.results) == 2
            assert len(handler.experiments) == 2

    def test_run_without_sweeps(self):
        """Test running without parameter sweeps (baseline experiment)."""
        sim_params = SimulationParameters(nphoton=5000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            # Run without adding any sweeps
            handler.run()
            
            # Should still run one baseline experiment
            assert len(handler.results) == 1
            assert len(handler.experiments) == 1


class TestResultsSaving:
    """Test saving results."""

    def test_save_results_creates_files(self):
        """Test that save_results creates result files."""
        sim_params = SimulationParameters(nphoton=5000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
                storage_format=StorageFormat.NPZ,
            )
            
            sweep = ParameterSweep(
                param_path="nphoton",
                values=[5000],
                object_type="simulation_params"
            )
            handler.add_sweep(sweep)
            handler.run()
            handler.save_results()
            
            # Check that result files were created
            output_path = Path(tmpdir)
            npz_files = list(output_path.glob("experiment_*.npz"))
            assert len(npz_files) >= 1

    def test_save_results_creates_mapping(self):
        """Test that save_results creates parameter mapping JSON."""
        sim_params = SimulationParameters(nphoton=5000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            sweep = ParameterSweep(
                param_path="nphoton",
                values=[5000, 10000],
                object_type="simulation_params"
            )
            handler.add_sweep(sweep)
            handler.run()
            handler.save_results()
            
            # Check JSON mapping exists
            json_path = Path(tmpdir) / "parameter_mapping.json"
            assert json_path.exists()
            
            with open(json_path) as f:
                mapping = json.load(f)
                assert "metadata" in mapping
                assert "experiments" in mapping
                assert len(mapping["experiments"]) == 2

    def test_mapping_contains_base_parameters(self):
        """Test that mapping contains base parameters."""
        sim_params = SimulationParameters(nphoton=5000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            sweep = ParameterSweep(
                param_path="nphoton",
                values=[5000],
                object_type="simulation_params"
            )
            handler.add_sweep(sweep)
            handler.run()
            handler.save_results()
            
            # Check that mapping contains sweep config and base params
            json_path = Path(tmpdir) / "parameter_mapping.json"
            with open(json_path) as f:
                mapping = json.load(f)
                
                # Check metadata
                assert "base_simulation_params" in mapping["metadata"]
                assert "base_tissue_model" in mapping["metadata"]
                assert "sweep_parameters" in mapping["metadata"]

    def test_save_results_raises_if_no_results(self):
        """Test that save_results raises error if experiments haven't been run."""
        sim_params = SimulationParameters(nphoton=5000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            # Try to save without running
            with pytest.raises(RuntimeError, match="No results to save"):
                handler.save_results()


class TestExperimentSummary:
    """Test experiment summary generation."""

    def test_get_results_summary(self):
        """Test getting experiment summary."""
        sim_params = SimulationParameters(nphoton=5000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
                storage_format=StorageFormat.NPZ,
            )
            
            sweep = ParameterSweep(
                param_path="nphoton",
                values=[5000, 10000],
                object_type="simulation_params"
            )
            handler.add_sweep(sweep)
            handler.run()
            
            summary = handler.get_results_summary()
            
            assert summary["total_experiments"] == 2
            assert summary["storage_format"] == "npz"
            assert summary["number_of_sweeps"] == 1
            assert len(summary["sweep_configs"]) == 1
            assert len(summary["experiments"]) == 2

    def test_summary_includes_sweep_configs(self):
        """Test that summary includes sweep configuration details."""
        sim_params = SimulationParameters(nphoton=5000)
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ExperimentHandler(
                base_simulation_params=sim_params,
                base_tissue_model=tissue,
                output_dir=tmpdir,
            )
            
            sweep = ParameterSweep(
                param_path="nphoton",
                values=[5000, 10000, 15000],
                object_type="simulation_params"
            )
            handler.add_sweep(sweep)
            handler.run()
            
            summary = handler.get_results_summary()
            
            assert summary["sweep_configs"][0]["param_path"] == "nphoton"
            assert summary["sweep_configs"][0]["object_type"] == "simulation_params"
            assert summary["sweep_configs"][0]["num_values"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
