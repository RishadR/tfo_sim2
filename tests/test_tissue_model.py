"""
Unit tests for tissue model implementations.

These tests verify the structure and basic functionality of tissue models
without requiring PMCX to be installed.
"""

import pytest
import numpy as np

from tfo_sim2 import (
    UniformTissueModel,
    LayeredTissueModel,
)


class TestUniformTissueModel:
    """Test UniformTissueModel."""

    def test_uniform_tissue_model(self):
        """Test UniformTissueModel."""
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )

        # Check volume
        assert tissue.vol.shape == (60, 60, 60 + 2)
        assert tissue.vol.dtype == np.uint8

        # Check properties
        assert len(tissue.prop) == 2
        assert tissue.prop[0] == [0, 0, 1, 1]  # background
        assert tissue.prop[1] == [0.005, 1.0, 0.01, 1.37]


class TestLayeredTissueModel:
    """Test LayeredTissueModel."""

    def test_layered_tissue_model(self):
        """Test LayeredTissueModel."""
        layers = [
            {
                "z_start": 0,
                "z_end": 20,
                "mua": 0.005,
                "mus": 0.5,
                "g": 0.01,
                "n": 1.37,
            },
            {
                "z_start": 20,
                "z_end": 40,
                "mua": 0.01,
                "mus": 1.0,
                "g": 0.01,
                "n": 1.37,
            },
        ]

        tissue = LayeredTissueModel(size=(60, 60, 60), layers=layers)

        # Check volume
        assert tissue.vol.shape == (60, 60, 60 + 2)
        assert np.all(tissue.vol[:, :, 0:20] == 1)
        assert np.all(tissue.vol[:, :, 20:40] == 2)
        assert np.all(tissue.vol[:, :, 40:60] == 0)


class TestTissueModelToCfg:
    """Test tissue model cfg conversion."""

    def test_tissue_model_to_cfg(self):
        """Test tissue model cfg conversion."""
        tissue = UniformTissueModel(
            size=(60, 60, 60),
            mua=0.005,
            mus=1.0,
            g=0.01,
            n=1.37,
        )

        cfg = {}
        tissue.to_cfg(cfg)

        assert "vol" in cfg
        assert "prop" in cfg
        assert cfg["vol"].shape == (60, 60, 60 + 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
