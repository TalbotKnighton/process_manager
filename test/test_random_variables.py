import numpy as np
import pytest
from typing import Any
from process_manager.data_handlers import (
    NormalDistribution, 
    CategoricalDistribution,
    NamedValue,
    UniformDistribution
)

class TestNormalDistribution:
    def test_initialization(self):
        # Test basic initialization
        dist = NormalDistribution(name="test", mu=0, sigma=1)
        assert dist.name == "test"
        assert dist.mu == 0
        assert dist.sigma == 1
        
        # Test with specified seed
        dist_seeded = NormalDistribution(name="test", mu=0, sigma=1, seed=42)
        assert dist_seeded.seed == 42

    def test_sampling(self):
        dist = NormalDistribution(name="test", mu=10, sigma=2, seed=42)
        
        # Test single sample with and without squeeze
        sample = dist.sample(squeeze=True)
        assert isinstance(sample, float)  # Should be a scalar with squeeze
        
        sample = dist.sample(squeeze=False)
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (1,)
        
        # Test multiple samples
        samples = dist.sample(1000)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (1000,)
        
        # Test statistical properties (within 3 sigma)
        assert 9 < np.mean(samples) < 11  # mean should be close to 10
        assert 1.5 < np.std(samples) < 2.5  # std should be close to 2
        
        # Test reproducibility with seed
        dist1 = NormalDistribution(name="test", mu=0, sigma=1, seed=42)
        dist2 = NormalDistribution(name="test", mu=0, sigma=1, seed=42)
        assert np.array_equal(dist1.sample(10), dist2.sample(10))

    def test_pdf(self):
        dist = NormalDistribution(name="test", mu=0, sigma=1)
        x = np.array([-1, 0, 1])
        
        # Test with and without squeeze
        pdf_values = dist.pdf(x, squeeze=True)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.shape == (3,)
        
        pdf_values_unsqueezed = dist.pdf(x, squeeze=False)
        assert isinstance(pdf_values_unsqueezed, np.ndarray)
        
        # Test PDF properties
        expected = 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
        np.testing.assert_array_almost_equal(pdf_values, expected)
        
        # Test PDF is non-negative
        assert np.all(pdf_values >= 0)
        
        # Test PDF maximum at mean
        x_dense = np.linspace(-3, 3, 1000)
        pdf_dense = dist.pdf(x_dense)
        assert np.abs(x_dense[np.argmax(pdf_dense)]) < 0.01  # maximum near 0

    def test_cdf(self):
        dist = NormalDistribution(name="test", mu=0, sigma=1)
        x = np.array([-np.inf, -1, 0, 1, np.inf])
        
        # Test with and without squeeze
        cdf_values = dist.cdf(x, squeeze=True)
        assert isinstance(cdf_values, np.ndarray)
        assert cdf_values.shape == (5,)
        
        cdf_values_unsqueezed = dist.cdf(x, squeeze=False)
        assert isinstance(cdf_values_unsqueezed, np.ndarray)
        
        # Test CDF properties
        assert cdf_values[0] == pytest.approx(0)  # CDF(-inf) = 0
        assert cdf_values[-1] == pytest.approx(1)  # CDF(inf) = 1
        assert cdf_values[2] == pytest.approx(0.5)  # CDF(mu) = 0.5
        
        # Test CDF is monotonic
        assert np.all(np.diff(cdf_values) >= 0)

class TestCategoricalDistribution:
    def test_initialization(self):
        # Test with equal probabilities (default)
        categories = np.array([1, 2, 3])
        dist = CategoricalDistribution(name="test", categories=categories)
        np.testing.assert_array_almost_equal(
            dist.probabilities, 
            np.array([1/3, 1/3, 1/3])
        )
        
        # Test with custom probabilities
        probs = np.array([0.2, 0.3, 0.5])
        dist = CategoricalDistribution(
            name="test", 
            categories=categories, 
            probabilities=probs
        )
        np.testing.assert_array_equal(dist.probabilities, probs)

    def test_validation(self):
        categories = np.array([1, 2, 3])
        
        # Test invalid probability sum
        with pytest.raises(ValueError):
            CategoricalDistribution(
                name="test",
                categories=categories,
                probabilities=np.array([0.2, 0.2, 0.2])
            )
        
        # Test negative probabilities
        with pytest.raises(ValueError):
            CategoricalDistribution(
                name="test",
                categories=categories,
                probabilities=np.array([0.5, -0.1, 0.6])
            )
        
        # Test mismatched lengths
        with pytest.raises(ValueError):
            CategoricalDistribution(
                name="test",
                categories=categories,
                probabilities=np.array([0.5, 0.5])
            )

    def test_sampling(self):
        categories = np.array(['A', 'B'])
        probs = np.array([0.7, 0.3])
        dist = CategoricalDistribution(
            name="test",
            categories=categories,
            probabilities=probs,
            seed=42
        )
        
        # Test single sample with and without squeeze
        sample = dist.sample(squeeze=True)
        assert isinstance(sample, str)  # Single sample should be scalar with squeeze
        assert sample in categories
        
        sample = dist.sample(squeeze=False)
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (1,)
        assert sample[0] in categories
        
        # Test multiple samples and frequencies
        n_samples = 10000
        samples = dist.sample(n_samples)
        unique, counts = np.unique(samples, return_counts=True)
        freq = counts / n_samples
        np.testing.assert_array_almost_equal(freq, probs, decimal=2)
        
        # Test reproducibility
        dist1 = CategoricalDistribution(
            name="test",
            categories=categories,
            probabilities=probs,
            seed=42
        )
        dist2 = CategoricalDistribution(
            name="test",
            categories=categories,
            probabilities=probs,
            seed=42
        )
        assert np.array_equal(dist1.sample(10), dist2.sample(10))

    def test_pdf(self):
        categories = np.array([1, 2, 3])
        probs = np.array([0.2, 0.3, 0.5])
        dist = CategoricalDistribution(
            name="test",
            categories=categories,
            probabilities=probs
        )
        
        # Test PDF values with and without squeeze
        x = np.array([1, 2, 3, 4])
        pdf_values = dist.pdf(x, squeeze=True)
        assert isinstance(pdf_values, np.ndarray)
        np.testing.assert_array_equal(
            pdf_values,
            np.array([0.2, 0.3, 0.5, 0.0])
        )
        
        pdf_values_unsqueezed = dist.pdf(x, squeeze=False)
        assert isinstance(pdf_values_unsqueezed, np.ndarray)

    def test_cdf(self):
        categories = np.array([1, 2, 3])
        probs = np.array([0.2, 0.3, 0.5])
        dist = CategoricalDistribution(
            name="test",
            categories=categories,
            probabilities=probs
        )
        
        # Test CDF values with and without squeeze
        x = np.array([0, 1, 2, 3, 4])
        cdf_values = dist.cdf(x, squeeze=True)
        assert isinstance(cdf_values, np.ndarray)
        np.testing.assert_array_equal(
            cdf_values,
            np.array([0.0, 0.2, 0.5, 1.0, 1.0])
        )
        
        cdf_values_unsqueezed = dist.cdf(x, squeeze=False)
        assert isinstance(cdf_values_unsqueezed, np.ndarray)
        
        # Test CDF is monotonic
        assert np.all(np.diff(cdf_values) >= 0)

class TestUniformDistribution:
    def test_init_valid(self):
        """Test valid initialization"""
        dist = UniformDistribution(name="test", low=0, high=1)
        assert dist.low == 0
        assert dist.high == 1
        assert dist.name == "test"

    def test_init_invalid_bounds(self):
        """Test that initialization fails when b <= a"""
        with pytest.raises(ValueError, match="Upper bound .* must be greater than lower bound"):
            UniformDistribution(name="test", low=1, high=0)
        
        with pytest.raises(ValueError, match="Upper bound .* must be greater than lower bound"):
            UniformDistribution(name="test", low=1, high=1)

    def test_sample_shape(self):
        """Test that sample returns correct shapes"""
        dist = UniformDistribution(name="test", low=0, high=1)
        
        # Single sample with and without squeeze
        sample = dist.sample(squeeze=True)
        assert isinstance(sample, float)  # Should be scalar
        
        sample = dist.sample(squeeze=False)
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (1,)
        
        # Multiple samples
        samples = dist.sample(10)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (10,)

    def test_sample_bounds(self):
        """Test that samples fall within specified bounds"""
        dist = UniformDistribution(name="test", low=-2, high=3)
        samples = dist.sample(1000)
        
        assert np.all(samples >= -2)
        assert np.all(samples <= 3)
        
        # Test mean is approximately correct (within 3 standard deviations)
        expected_mean = (-2 + 3) / 2  # 0.5
        assert abs(np.mean(samples) - expected_mean) < 3 * np.std(samples)

    def test_pdf(self):
        """Test probability density function"""
        dist = UniformDistribution(name="test", low=0, high=2)
        x = np.array([-1, 0, 1, 2, 3])
        
        # Test with and without squeeze
        pdf_values = dist.pdf(x, squeeze=True)
        assert isinstance(pdf_values, np.ndarray)
        expected = np.array([0, 0.5, 0.5, 0.5, 0])
        np.testing.assert_allclose(pdf_values, expected)
        
        pdf_values_unsqueezed = dist.pdf(x, squeeze=False)
        assert isinstance(pdf_values_unsqueezed, np.ndarray)
        
        # Test that pdf integrates to approximately 1
        x_dense = np.linspace(-1, 3, 10000)
        pdf_dense = dist.pdf(x_dense)
        integral = np.trapezoid(pdf_dense, x_dense)
        assert abs(integral - 1.0) < 1e-3

    def test_cdf(self):
        """Test cumulative distribution function"""
        dist = UniformDistribution(name="test", low=0, high=2)
        x = np.array([-1, 0, 1, 2, 3])
        
        # Test with and without squeeze
        cdf_values = dist.cdf(x, squeeze=True)
        assert isinstance(cdf_values, np.ndarray)
        expected = np.array([0, 0, 0.5, 1, 1])
        np.testing.assert_allclose(cdf_values, expected)
        
        cdf_values_unsqueezed = dist.cdf(x, squeeze=False)
        assert isinstance(cdf_values_unsqueezed, np.ndarray)

    def test_reproducibility(self):
        """Test that setting seed produces reproducible results"""
        dist1 = UniformDistribution(name="test", low=0, high=1, seed=42)
        dist2 = UniformDistribution(name="test", low=0, high=1, seed=42)
        
        samples1 = dist1.sample(100)
        samples2 = dist2.sample(100)
        
        np.testing.assert_array_equal(samples1, samples2)

    def test_sample_with_different_sizes(self):
        """Test sampling with different size specifications"""
        dist = UniformDistribution(name="test", low=0, high=1)
        
        # Test with tuple size
        samples = dist.sample((2, 3))
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (2, 3)
        
        # Test with integer size
        samples = dist.sample(5)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (5,)