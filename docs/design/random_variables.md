# Random Variable Design

The random variable system provides a flexible and type-safe way to define, sample, and manage random variables in simulations. It implements a hierarchical design pattern where specific distributions inherit from a base random variable class.

## Core Design Principles

### 1. Type Safety and Validation
The system uses Python's type hints and runtime validation to ensure distributions are configured correctly:

```python
# Type-safe parameter definitions
class NormalRandomVariable(RandomVariable[float]):
    mean: float
    std_dev: float = Field(gt=0)  # Validation that std_dev must be positive
```

### 2. Inheritance Hierarchy
A clear inheritance structure ensures consistent behavior across different distributions:

```python
RandomVariable[T]  # Base class with generic type T
    ↳ ContinuousRandomVariable  # For continuous distributions
        ↳ NormalRandomVariable  # Specific normal distribution
        ↳ UniformRandomVariable # Specific uniform distribution
    ↳ DiscreteRandomVariable   # For discrete distributions
        ↳ PoissonRandomVariable # Specific Poisson distribution
```

### 3. Sampling Interface
All random variables implement a consistent sampling interface:

```python
class RandomVariable(Generic[T]):
    def sample(self, size: Optional[int] = None) -> T | NDArray:
        """Sample from the distribution."""
        raise NotImplementedError
        
    def sample_to_list(self, size: int) -> list[T]:
        """Sample multiple values into a list."""
        return list(self.sample(size))
```

### 4. Parameter Validation
Parameters are validated both at instantiation and runtime:

```python
class UniformRandomVariable(ContinuousRandomVariable):
    low: float
    high: float
    
    @field_validator("high")
    def validate_bounds(cls, high: float, info: ValidationInfo) -> float:
        low = info.data.get("low", 0.0)
        if high <= low:
            raise ValueError("high must be greater than low")
        return high
```

## Implementation Details

### 1. Normal Distribution
```python
class NormalRandomVariable(ContinuousRandomVariable):
    """
    Generates normally distributed random values.
    """
    mean: float = 0.0
    std_dev: float = Field(gt=0, default=1.0)
    
    def sample(self, size: Optional[int] = None) -> float | NDArray:
        return np.random.normal(self.mean, self.std_dev, size)
```

### 2. Uniform Distribution
```python
class UniformRandomVariable(ContinuousRandomVariable):
    """
    Generates uniformly distributed random values.
    """
    low: float = 0.0
    high: float = 1.0
    
    def sample(self, size: Optional[int] = None) -> float | NDArray:
        return np.random.uniform(self.low, self.high, size)
```

## Testing Strategy

The testing approach verifies both the statistical properties and error handling of the distributions.

### 1. Statistical Property Tests
```python
def test_normal_distribution_properties():
    # Create normal distribution
    normal = NormalRandomVariable(mean=10, std_dev=2)
    
    # Sample large number of values
    samples = normal.sample(10000)
    
    # Check statistical properties
    assert 9.8 < np.mean(samples) < 10.2  # Mean within range
    assert 1.9 < np.std(samples) < 2.1    # Std dev within range
```

### 2. Parameter Validation Tests
```python
def test_invalid_parameters():
    # Test invalid standard deviation
    with pytest.raises(ValidationError):
        NormalRandomVariable(mean=0, std_dev=-1)
        
    # Test invalid uniform bounds
    with pytest.raises(ValidationError):
        UniformRandomVariable(low=10, high=5)
```

### 3. Sampling Interface Tests
```python
def test_sampling_interface():
    normal = NormalRandomVariable(mean=0, std_dev=1)
    
    # Test single sample
    assert isinstance(normal.sample(), float)
    
    # Test multiple samples
    samples = normal.sample(10)
    assert len(samples) == 10
    
    # Test list conversion
    sample_list = normal.sample_to_list(5)
    assert isinstance(sample_list, list)
    assert len(sample_list) == 5
```

## Usage Examples

### 1. Basic Usage
```python
# Create a normal distribution
normal = NormalRandomVariable(mean=10, std_dev=2)

# Single sample
value = normal.sample()

# Multiple samples
values = normal.sample(100)
```

### 2. Using in Simulations
```python
# Define process variation
process_var = NormalRandomVariable(mean=100, std_dev=5)

# Simulate process
measurements = process_var.sample_to_list(1000)
```

### 3. Combining Distributions
```python
# Process with random failures
base_process = NormalRandomVariable(mean=100, std_dev=2)
failure_rate = PoissonRandomVariable(lambda_=0.1)

def simulate_process(n_steps: int) -> list[float]:
    measurements = base_process.sample_to_list(n_steps)
    failures = failure_rate.sample_to_list(n_steps)
    return [m if f == 0 else 0.0 for m, f in zip(measurements, failures)]
```

## Best Practices

1. Always validate distribution parameters:
```python
# Good
normal = NormalRandomVariable(mean=0, std_dev=1.0)

# Avoid
normal = NormalRandomVariable(mean=0, std_dev=-1.0)  # Will raise error
```

2. Use appropriate distribution types:
```python
# Good - continuous values
process_temp = NormalRandomVariable(mean=350, std_dev=5)

# Good - discrete counts
defects = PoissonRandomVariable(lambda_=2.5)
```

3. Handle sampling errors appropriately:
```python
try:
    samples = distribution.sample(1000)
except ValueError as e:
    # Handle sampling error
    logger.error(f"Sampling failed: {e}")
```

4. Use type hints for clarity:
```python
def simulate_process(
    distribution: RandomVariable[float],
    n_samples: int
) -> NDArray:
    return distribution.sample(n_samples)
```

## Extended Features

### 1. Distribution Composition
The system supports combining distributions:

```python
class CompositeRandomVariable(RandomVariable[float]):
    distributions: list[RandomVariable[float]]
    weights: list[float]
    
    def sample(self, size: Optional[int] = None) -> float | NDArray:
        samples = [d.sample(size) for d in self.distributions]
        return np.average(samples, weights=self.weights, axis=0)
```

### 2. Serialization Support
Random variables can be serialized for storage or transmission:

```python
# Save distribution parameters
normal = NormalRandomVariable(mean=10, std_dev=2)
json_data = normal.model_dump_json()

# Recreate distribution
loaded = NormalRandomVariable.model_validate_json(json_data)
```

This design provides a robust foundation for handling random variables in simulation and statistical applications while maintaining type safety and proper validation.