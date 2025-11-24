"""
Unit tests for simulation parameters.

These tests verify the SimulationParameters class functionality.
"""

import pytest

from tfo_sim2 import SimulationParameters


class TestSimulationParametersCreation:
    """Test SimulationParameters creation."""

    def test_simulation_parameters_creation(self):
        """Test parameter initialization."""
        params = SimulationParameters(
            nphoton=1000000,
            tend=5e-9,
            srcpos=[30, 30, 0],
        )

        assert params.nphoton == 1000000
        assert params.tend == 5e-9
        assert params.srcpos == [30, 30, 0]


class TestSimulationParametersConfig:
    """Test SimulationParameters configuration."""

    def test_simulation_parameters_to_cfg(self):
        """Test cfg conversion."""
        params = SimulationParameters()
        cfg = params.to_cfg()

        assert "nphoton" in cfg
        assert "srcpos" in cfg
        assert "srcdir" in cfg
        assert cfg["issavedet"] == 1

    def test_simulation_parameters_update_from_cfg(self):
        """Test cfg update."""
        params = SimulationParameters()
        cfg = {
            "nphoton": 500000,
            "tend": 1e-8,
            "srcpos": [20, 20, 0],
            "new_param": 42,
        }

        params.update_from_cfg(cfg)

        assert params.nphoton == 500000
        assert params.tend == 1e-8
        assert params.srcpos == [20, 20, 0]
        assert params.extra_params["new_param"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
