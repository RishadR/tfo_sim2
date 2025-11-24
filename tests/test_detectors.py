"""
Unit tests for detector array functionality.

These tests verify the DetectorArray class and detector positioning.
"""

import pytest

from tfo_sim2 import DetectorArray


class TestDetectorArrayBasic:
    """Test basic DetectorArray operations."""

    def test_add_single_detector(self):
        """Test adding a single detector."""
        detectors = DetectorArray()
        det_id = detectors.add_detector((30, 30, 50), radius=1.0)
        
        assert det_id == 1
        assert len(detectors) == 1

    def test_add_multiple_detectors(self):
        """Test adding multiple detectors."""
        detectors = DetectorArray()
        positions = [(20., 20., 50.), (30., 30., 50.), (40., 40., 50.)]
        ids = detectors.add_detectors_at_positions(positions)
        
        assert len(ids) == 3
        assert len(detectors) == 3


class TestDetectorArrayPatterns:
    """Test detector positioning patterns."""

    def test_add_detector_line(self):
        """Test adding detectors in a line."""
        detectors = DetectorArray()
        ids = detectors.add_detector_line(
            start=(10, 30, 50),
            end=(50, 30, 50),
            num_detectors=5,
        )
        
        assert len(ids) == 5
        assert len(detectors) == 5

    def test_add_detector_grid(self):
        """Test adding detectors in a grid."""
        detectors = DetectorArray()
        ids = detectors.add_detector_grid(
            x_range=(20, 40),
            y_range=(20, 40),
            z=50,
            nx=3,
            ny=3,
        )
        
        assert len(ids) == 9
        assert len(detectors) == 9


class TestDetectorArrayConfig:
    """Test DetectorArray cfg conversion."""

    def test_detector_array_to_cfg(self):
        """Test cfg conversion."""
        detectors = DetectorArray()
        detectors.add_detector((30, 30, 50))
        
        cfg = {}
        detectors.to_cfg(cfg)
        
        assert 'detpos' in cfg
        assert cfg['issavedet'] == 1
        assert len(cfg['detpos']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
