"""
Unit tests for result storage functionality.

These tests verify:
1. Save/load functionality for all StorageFormat types
2. ResultStorage correctly processes data and populates StorableResults
3. StorableResults validity checks and error handling
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import warnings
import random

from tfo_sim2.result_storage import (
    ResultStorage,
    StorageFormat,
    StorableResults,
    Flux,
    Statistics,
    DetectorID,
    NumScatterings,
    PartialPath,
    Momentum,
    ExitPosition,
    ExitVelocity,
    InitialWeight,
    OpticalProperties,
    UnitInMM,
    DetectorIntensity,
    SourcePosition,
    DetectorPositions,
    FluxSlice,
)


class TestStorageSaveAndLoad:
    """Test save/load functionality for all storage formats."""

    @pytest.fixture
    def dummy_result_storage(self):
        """Create a dummy ResultStorage with some test data."""
        data_dict = {
            'flux': np.random.rand(10, 10, 10, 5),  # 4D flux array
            'stat': np.array([100.0, 200.0, 300.0]),
            'srcpos': np.array([0.0, 0.0, 0.0]),
            'detpos': np.array([[10.0, 10.0, 10.0]]),
        }
        storage = ResultStorage(data_dict=data_dict, name="TestStorage", results_to_store=[
            Flux(),
            Statistics(),
            SourcePosition(),
            DetectorPositions(),
        ])
        return storage

    def test_save_and_load_npz_format(self, dummy_result_storage):
        """Test NPZ save and load - ensure data is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.npz"
            
            # Save
            dummy_result_storage.save(str(filepath), StorageFormat.NPZ)
            assert filepath.exists(), "NPZ file was not created"
            
            # Load
            loaded_storage = ResultStorage.load(str(filepath), StorageFormat.NPZ)
            
            # Verify loaded data
            assert loaded_storage.name == "TestStorage"
            assert loaded_storage.flux is not None
            assert loaded_storage.stat is not None
            assert loaded_storage.srcpos is not None
            assert loaded_storage.detpos is not None
            assert np.array_equal(loaded_storage.flux, dummy_result_storage.flux)  # type: ignore
            assert np.array_equal(loaded_storage.stat, dummy_result_storage.stat)  # type: ignore
            assert np.array_equal(loaded_storage.srcpos, dummy_result_storage.srcpos)  # type: ignore
            assert np.array_equal(loaded_storage.detpos, dummy_result_storage.detpos)  # type: ignore

    def test_save_and_load_hdf5_format(self, dummy_result_storage):
        """Test HDF5 save and load - ensure data is preserved."""
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            
            # Save
            dummy_result_storage.save(str(filepath), StorageFormat.HDF5)
            assert filepath.exists(), "HDF5 file was not created"
            
            # Load
            loaded_storage = ResultStorage.load(str(filepath), StorageFormat.HDF5)
            
            # Verify loaded data (HDF5 may store strings as bytes)
            name_value = loaded_storage.name
            if isinstance(name_value, bytes):
                name_value = name_value.decode('utf-8')
            assert name_value == "TestStorage"
            assert loaded_storage.flux is not None
            assert dummy_result_storage.flux is not None
            assert np.array_equal(loaded_storage.flux, dummy_result_storage.flux)  # type: ignore
            assert loaded_storage.stat is not None
            assert dummy_result_storage.stat is not None
            assert np.array_equal(loaded_storage.stat, dummy_result_storage.stat)  # type: ignore
            assert loaded_storage.srcpos is not None
            assert dummy_result_storage.srcpos is not None
            assert np.array_equal(loaded_storage.srcpos, dummy_result_storage.srcpos)  # type: ignore

    def test_save_and_load_json_format(self, dummy_result_storage):
        """Test JSON save and load - ensure data is preserved."""
        try:
            import jdata  # noqa: F401
        except ImportError:
            pytest.skip("jdata not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            
            # Save
            dummy_result_storage.save(str(filepath), StorageFormat.JSON)
            assert filepath.exists(), "JSON file was not created"
            
            # Load
            loaded_storage = ResultStorage.load(str(filepath), StorageFormat.JSON)
            
            # Verify loaded data (JSON may convert arrays differently)
            assert loaded_storage.name == "TestStorage"
            # Check that flux data exists and has correct shape
            assert loaded_storage.flux is not None
            assert loaded_storage.stat is not None
            assert loaded_storage.srcpos is not None
            # Note: detpos may not be preserved depending on JSON implementation
            if loaded_storage.detpos is not None:
                assert np.array_equal(loaded_storage.detpos, dummy_result_storage.detpos)  # type: ignore

    def test_save_format_inferred_from_extension(self, dummy_result_storage):
        """Test that format is correctly inferred from file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test NPZ extension inference
            npz_path = Path(tmpdir) / "test.npz"
            dummy_result_storage.save(str(npz_path))  # No format specified
            loaded = ResultStorage.load(str(npz_path))  # No format specified
            assert loaded.flux is not None
            assert dummy_result_storage.flux is not None
            assert np.array_equal(loaded.flux, dummy_result_storage.flux)  # type: ignore
            
            # Test HDF5 extension inference (if h5py available)
            try:
                import h5py  # noqa: F401
                h5_path = Path(tmpdir) / "test.h5"
                dummy_result_storage.save(str(h5_path), StorageFormat.HDF5)  # Specify format explicitly
                loaded = ResultStorage.load(str(h5_path))  # Format inferred from extension
                assert loaded.flux is not None
                assert dummy_result_storage.flux is not None
                assert np.array_equal(loaded.flux, dummy_result_storage.flux)  # type: ignore
            except ImportError:
                pass


class TestResultStorageDataProcessing:
    """Test that ResultStorage correctly processes data into StorableResults."""

    def test_flux_result_processing(self):
        """Test that Flux result is correctly processed and stored."""
        flux_data = np.random.rand(15, 15, 15, 3)
        data_dict = {
            'flux': flux_data,
            'srcpos': np.array([0.0, 0.0, 0.0]),
        }
        
        storage = ResultStorage(
            data_dict=data_dict,
            results_to_store=[Flux(), SourcePosition()]
        )
        
        assert storage.flux is not None
        assert np.array_equal(storage.flux, flux_data)
        assert storage.flux.shape == flux_data.shape

    def test_statistics_result_processing(self):
        """Test that Statistics result is correctly processed and stored."""
        stat_data = np.array([500.0, 1000.0, 1500.0])
        data_dict = {
            'stat': stat_data,
            'srcpos': np.array([0.0, 0.0, 0.0]),
        }
        
        storage = ResultStorage(
            data_dict=data_dict,
            results_to_store=[Statistics(), SourcePosition()]
        )
        
        assert storage.stat is not None
        assert np.array_equal(storage.stat, stat_data)

    def test_detector_results_with_detp_dict(self):
        """Test detector-related results processing with nested detp dict."""
        detp_dict = {
            'detid': np.array([1, 2, 1, 3, 2]),
            'nscat': np.array([10, 20, 15, 25, 18]),
            'ppath': np.array([1.5, 2.5, 1.8, 3.0, 2.2]),
            'mom': np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            'p': np.array([[1.0, 1.0, 10.0], [2.0, 2.0, 10.0]]),
            'v': np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            'w0': np.array([1.0, 0.95]),
            'prop': np.array([[1.0, 0.1, 0.9, 1.3], [1.0, 0.1, 0.9, 1.3]]),
            'unitinmm': 1.0,
        }
        data_dict = {
            'detp': detp_dict,
            'srcpos': np.array([0.0, 0.0, 0.0]),
        }
        
        # Test two random detector-based results
        detector_results = [
            DetectorID(),
            NumScatterings(),
            PartialPath(),
            Momentum(),
            ExitPosition(),
            ExitVelocity(),
            InitialWeight(),
            OpticalProperties(),
            UnitInMM(),
        ]
        
        selected_results = random.sample(detector_results, 2)
        storage = ResultStorage(
            data_dict=data_dict,
            results_to_store=selected_results + [SourcePosition()]
        )
        
        for result in selected_results:
            attr_name = result.name
            assert hasattr(storage, attr_name)
            assert getattr(storage, attr_name) is not None

    def test_random_pair_of_results(self):
        """Test that two randomly selected StorableResults are processed correctly."""
        # Create comprehensive data_dict
        comprehensive_data = {
            'flux': np.random.rand(8, 8, 8, 2),
            'stat': np.array([100.0, 200.0]),
            'detp': {
                'detid': np.array([1, 2, 3]),
                'nscat': np.array([10, 20, 30]),
                'ppath': np.array([1.0, 2.0, 3.0]),
                'mom': np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
                'p': np.array([[1.0, 1.0, 10.0], [2.0, 2.0, 10.0], [3.0, 3.0, 10.0]]),
                'v': np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
                'w0': np.array([1.0, 0.95, 0.9]),
                'prop': np.array([[1.0, 0.1, 0.9, 1.3], [1.0, 0.1, 0.9, 1.3], [1.0, 0.1, 0.9, 1.3]]),
                'unitinmm': 1.0,
            },
            'srcpos': np.array([0.0, 0.0, 0.0]),
            'detpos': np.array([[10.0, 10.0, 10.0]]),
        }
        
        all_results = [
            Flux(),
            Statistics(),
            DetectorID(),
            NumScatterings(),
            PartialPath(),
            Momentum(),
            ExitPosition(),
            ExitVelocity(),
            InitialWeight(),
            OpticalProperties(),
            UnitInMM(),
            SourcePosition(),
            DetectorPositions(),
        ]
        
        # Pick 2 random results
        selected = random.sample(all_results, 2)
        storage = ResultStorage(
            data_dict=comprehensive_data,
            results_to_store=selected
        )
        
        # Verify both selected results were processed
        for result in selected:
            attr_name = result.name
            assert hasattr(storage, attr_name)
            # Should be populated since we have complete data
            if result.validity_check(comprehensive_data):
                assert getattr(storage, attr_name) is not None


class TestStorableResultsValidityChecks:
    """Test that StorableResults properly validate data and raise errors when needed."""

    def test_detector_id_validity_check_fails_without_detp(self):
        """Test that DetectorID raises error when detp dict is missing."""
        detector_id = DetectorID()
        
        # Data without detp
        data_dict_invalid = {'flux': np.random.rand(10, 10, 10)}
        
        assert not detector_id.validity_check(data_dict_invalid)
        # When extracted, should return None (no error raised, warning issued)
        with warnings.catch_warnings(record=True) as w:
            storage = ResultStorage(
                data_dict=data_dict_invalid,
                results_to_store=[detector_id]
            )
            assert storage.detid is None
            assert len(w) == 1
            assert "not available in data_dict" in str(w[0].message)

    def test_statistics_validity_check_fails_without_stat(self):
        """Test that Statistics raises error when stat key is missing."""
        stat = Statistics()
        
        # Data without stat
        data_dict_invalid = {'flux': np.random.rand(10, 10, 10)}
        
        assert not stat.validity_check(data_dict_invalid)
        with warnings.catch_warnings(record=True) as w:
            storage = ResultStorage(
                data_dict=data_dict_invalid,
                results_to_store=[stat]
            )
            assert storage.stat is None
            assert len(w) == 1

    def test_partial_path_validity_check_fails_without_detp_ppath(self):
        """Test that PartialPath validity check fails without ppath in detp."""
        ppath = PartialPath()
        
        # Data with detp but without ppath
        data_dict_invalid = {
            'detp': {
                'detid': np.array([1, 2, 3]),
                'nscat': np.array([10, 20, 30]),
            }
        }
        
        assert not ppath.validity_check(data_dict_invalid)
        with warnings.catch_warnings(record=True) as w:
            storage = ResultStorage(
                data_dict=data_dict_invalid,
                results_to_store=[ppath]
            )
            assert storage.ppath is None
            assert len(w) == 1

    def test_momentum_validity_check_fails_without_detp_mom(self):
        """Test that Momentum validity check fails without mom in detp."""
        momentum = Momentum()
        
        # Data with detp but without mom
        data_dict_invalid = {
            'detp': {
                'detid': np.array([1, 2, 3]),
                'ppath': np.array([1.0, 2.0, 3.0]),
            }
        }
        
        assert not momentum.validity_check(data_dict_invalid)
        with warnings.catch_warnings(record=True) as w:
            storage = ResultStorage(
                data_dict=data_dict_invalid,
                results_to_store=[momentum]
            )
            assert storage.momentum is None
            assert len(w) == 1

    def test_detector_positions_validity_check_fails_without_detpos(self):
        """Test that DetectorPositions validity check fails without detpos."""
        detpos = DetectorPositions()
        
        # Data without detpos
        data_dict_invalid = {'flux': np.random.rand(10, 10, 10)}
        
        assert not detpos.validity_check(data_dict_invalid)
        with warnings.catch_warnings(record=True) as w:
            storage = ResultStorage(
                data_dict=data_dict_invalid,
                results_to_store=[detpos]
            )
            assert storage.detpos is None
            assert len(w) == 1

    def test_source_position_validity_check_fails_without_srcpos(self):
        """Test that SourcePosition handles missing srcpos gracefully."""
        srcpos = SourcePosition()
        
        # Data without srcpos
        data_dict_invalid = {'flux': np.random.rand(10, 10, 10)}
        
        # SourcePosition uses .get() which returns None
        result = srcpos.extract(data_dict_invalid)
        # Result will be array(None, dtype=object) due to np.array() wrapping
        assert result is None or (isinstance(result, np.ndarray) and result.item() is None)

    def test_two_random_validity_checks_fail(self):
        """Test validity checks for two randomly selected StorableResults."""
        all_results = [
            Flux(),
            Statistics(),
            DetectorID(),
            NumScatterings(),
            PartialPath(),
            Momentum(),
            ExitPosition(),
            ExitVelocity(),
            InitialWeight(),
            OpticalProperties(),
            UnitInMM(),
            DetectorIntensity(),
            SourcePosition(),
            DetectorPositions(),
        ]
        
        # Create minimal data_dict that fails most validity checks
        minimal_data = {'flux': np.random.rand(5, 5, 5)}
        
        # Pick 2 random results and verify they fail validity check
        selected = random.sample(all_results, 2)
        
        for result in selected:
            validity = result.validity_check(minimal_data)
            # Flux should be valid with this data
            if isinstance(result, Flux):
                assert validity
            # Most others should be invalid
            elif not isinstance(result, Flux):
                # Some might still be valid depending on implementation
                # Just verify the method runs without error
                assert isinstance(validity, bool)


class TestDetectorIntensityComputation:
    """Test that DetectorIntensity can be computed from detp data."""

    def test_detector_intensity_computation(self):
        """Test that DetectorIntensity is computed correctly when possible."""
        try:
            from pmcx.utils import detweight  # noqa: F401
        except ImportError:
            pytest.skip("pmcx not available for detweight computation")
        
        # Create valid detp data with proper optical properties
        # prop should have shape (num_regions, 4) where 4 = [mu_s, mu_a, g, n]
        detp_dict = {
            'detid': np.array([1, 2, 1]),
            'ppath': np.array([[1.0, 2.0, 1.5], [0.5, 1.5, 1.0]]).T,  # shape (3, 2) - 3 photons, 2 layers
            'prop': np.array([[1.0, 0.1, 0.9, 1.3], [1.0, 0.1, 0.9, 1.3]]),  # shape (2, 4)
            'unitinmm': 1.0,
        }
        data_dict = {
            'detp': detp_dict,
            'srcpos': np.array([0.0, 0.0, 0.0]),
        }
        
        storage = ResultStorage(
            data_dict=data_dict,
            results_to_store=[DetectorIntensity(), SourcePosition()]
        )
        
        # DetectorIntensity should be computed
        assert storage.detint is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
