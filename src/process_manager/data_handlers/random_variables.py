"""Random Variable Implementations for Process Analysis.

This module provides a framework for working with random variables and probability
distributions in a type-safe, numerically stable way. It includes implementations
of common distributions (Normal, Uniform, Categorical) and infrastructure for
managing collections of random variables.

The module uses a metaclass-based approach to ensure consistent handling of array
dimensions across all distributions, making it easy to work with both scalar and
vector-valued random variables.

Key Features:
    - Type-safe implementations of common probability distributions
    - Automatic dimension handling via the squeezable decorator
    - Support for reproducible sampling via seed management
    - Collections for managing groups of random variables
    - Serialization support via pydantic
"""
from __future__ import annotations

# Standard library imports
from typing import Optional, Iterable, TypeVar, Generic, List, Any, Callable, ParamSpec
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from functools import wraps

# External dependencies
import numpy as np
from numpydantic import NDArray
from pydantic import BaseModel, InstanceOf, SerializeAsAny, model_validator, Field
import scipy.special as sp

# Local imports
from process_manager.data_handlers.base import NamedObject, NamedObjectList, NamedObjectHash
from process_manager.data_handlers.values import NamedValue, SerializableValue

__all__ = [
    'RandomVariable',
    'RandomVariableList',
    'RandomVariableHash',
    'NormalDistribution',
    'UniformDistribution',
    'CategoricalDistribution'
]

# Type variables and parameters for generic typing
P = ParamSpec('P')
R = TypeVar('R', bound=np.ndarray)
T = TypeVar('T', bound=SerializableValue)


class RandomVariableList(NamedObjectList):
    """List of random variables.

    This class manages an ordered list of RandomVariable objects, providing methods
    to add, access, and sample from multiple distributions while maintaining their order.

    Attributes:
        _registry_category (str): Category name for object registration
        objects (List[SerializeAsAny[InstanceOf[RandomVariable]]]): List of random variable objects
    """
    _registry_category = "random_variables"
    objects: List[SerializeAsAny[InstanceOf[RandomVariable]]] = Field(default_factory=list)

    def append(self, variable: RandomVariable) -> Self:
        """Append a random variable to the end of the list.

        Args:
            variable (RandomVariable): Random variable to append

        Returns:
            Self: The RandomVariableList instance for method chaining
        """
        return super().append(variable)

    def extend(self, variables: Iterable[RandomVariable]) -> Self:
        """Extend the list with multiple random variables.

        Args:
            variables (Iterable[RandomVariable]): Collection of random variables to add

        Returns:
            Self: The RandomVariableList instance for method chaining
        """
        return super().extend(variables)

    def __getitem__(self, idx: int) -> RandomVariable:
        """Get a random variable by its index in the list.

        Args:
            idx (int): Index of the random variable to retrieve

        Returns:
            RandomVariable: The random variable at the specified index

        Raises:
            IndexError: If the index is out of range
        """
        return super().__getitem__(idx)

    def sample_all(self, size: int = 1) -> dict[str, np.ndarray]:
        """Sample from all variables in the list.

        Args:
            size (int): Number of samples to generate per variable

        Returns:
            dict[str, np.ndarray]: Dictionary mapping variable names to their samples
        """
        return {var.name: var.sample(size) for var in self.objects}

    def register_variable(self, var: RandomVariable) -> Self:
        """Register a random variable to the collection.

        Args:
            var (RandomVariable): Random variable to register

        Returns:
            Self: The RandomVariableList instance
        """
        return self.register_object(var)

    def get_variable(self, name: str) -> RandomVariable:
        """Get a registered random variable by name.

        Args:
            name (str): Name of the random variable

        Returns:
            RandomVariable: The requested random variable

        Raises:
            KeyError: If no variable exists with the given name
        """
        return self.get_object(name)

    def get_variables(self) -> Iterable[RandomVariable]:
        """Get all registered random variables.

        Returns:
            Iterable[RandomVariable]: Iterator over all registered variables
        """
        return self.get_objects()

class RandomVariableHash(NamedObjectHash):
    """Collection of random variables.

    This class manages a collection of RandomVariable objects, providing methods
    to register, retrieve and sample from multiple distributions.

    Attributes:
        _registry_category (str): Category name for object registration
    """
    _registry_category = "random_variables"

    def register_variable(
            self, 
            var: RandomVariable, 
            size: int = 1,
            sample: bool = True,
            squeeze: bool = True,
        ) -> NamedValue[SerializableValue|NDArray[Any,SerializableValue]]:
        """Register a random variable and return its samples wrapped in a NamedValue.

        Args:
            var (RandomVariable): Random variable to register
            size (int): Number of samples to generate

        Returns:
            NamedValue[SerializableValue|NDArray[Any,SerializableValue]]: Named value containing samples

        Raises:
            ValueError: If a variable with the same name already exists
        """
        if var.name in self.objects:
            raise ValueError(f"A variable with name '{var.name}' already exists in the collection")

        self.register_object(var)
        if sample:
            samples = var.sample(size=size)
            if squeeze:
                return NamedValue(name=var.name, value=samples.squeeze())
            else:
                return NamedValue(name=var.name, value=samples)
        else:
            return None

    def get_variables(self) -> Iterable[RandomVariable]:
        """Get all registered random variables.

        Returns:
            Iterable[RandomVariable]: Iterator over all registered variables
        """
        return self.get_objects()

    def sample_all(self, size: int = 1) -> dict[str, np.ndarray]:
        """Sample from all registered distributions.

        Args:
            size (int): Number of samples to generate per distribution

        Returns:
            dict[str, np.ndarray]: Dictionary mapping variable names to their samples
        """
        return {name: var.sample(size) for name, var in self.objects.items()}

from typing import Any, Callable, ParamSpec, TypeVar
from functools import wraps
import inspect
from inspect import Parameter, Signature

def squeezable(func: Callable[P, R], squeeze_by_default: bool = False) -> Callable[P, R]:
    """
    Decorator that makes a function's output array squeezable
    via an added keyword argument `squeeze`.

    Args:
        func (Callable[P, R]): The function to be decorated.
        squeeze_by_default (bool): Whether or not to squeeze by default

    Returns:
        (Callable[P, R]): A new function that 
            squeezes the output of `func` if added keyword `squeeze` is True (default)
    """
    # Get the original signature
    sig = inspect.signature(func)
    
    # Create new parameters list with correct ordering
    parameters = []
    has_var_kwargs = False
    
    # First add all non-variadic parameters
    for param in sig.parameters.values():
        if param.kind != Parameter.VAR_KEYWORD:
            parameters.append(param)
        else:
            has_var_kwargs = True
    
    # Add squeeze parameter as keyword-only
    parameters.append(
        Parameter(
            'squeeze',
            Parameter.KEYWORD_ONLY,
            default=squeeze_by_default,
            annotation=bool
        )
    )
    
    # Add var_kwargs at the end if it exists
    if has_var_kwargs:
        parameters.append(
            Parameter(
                'kwargs',
                Parameter.VAR_KEYWORD
            )
        )
    
    # Create new signature
    new_sig = sig.replace(parameters=parameters)
    
    @wraps(func)
    def wrapper(*args: P.args, squeeze: bool = squeeze_by_default, **kwargs: P.kwargs) -> R:
        result = func(*args, **kwargs)
        if squeeze:
            result = result.squeeze()
            if isinstance(result, np.ndarray) and result.ndim == 0:
                result = result.item()
        return result
    
    # Update the wrapper's signature
    wrapper.__signature__ = new_sig
    return wrapper

class RandomVariableMeta(type(BaseModel)):
    """Metaclass for random variable implementations that automatically adds array handling functionality.
    
    This metaclass inherits from pydantic's model metaclass to maintain compatibility with the 
    BaseModel validation system while adding automatic array handling capabilities to all random 
    variable implementations.
    
    Key Features:
        - Automatically applies the `squeezable` decorator to sample(), pdf(), and cdf() methods
        - Maintains compatibility with pydantic's model validation system
        - Ensures consistent array handling across all random variable implementations
    
    The metaclass processes each new random variable class during its creation by:
        1. Identifying the standard distribution methods (sample, pdf, cdf)
        2. If these methods are defined in the class (not inherited), wrapping them with
           the squeezable decorator
        3. Preserving the original method docstrings while adding squeeze parameter documentation
    
    Example:
        ```python
        class NormalDistribution(RandomVariable[float]):
            def sample(self, size: int = 1) -> NDArray[Any, float]:
                # Method will automatically get squeeze functionality
                # and accept an optional `bool` defaulting to `squeeze=True`
                return rng.normal(self.mu, self.sigma, size=size)
        ```
    
    Technical Details:
        - Inherits from type(BaseModel) to maintain pydantic compatibility
        - Uses __new__ to modify class attributes during class creation
        - Preserves method signatures while adding the squeeze parameter
        - Ensures proper type hints and docstring updates
    
    Notes:
        - The squeezable decorator adds a `squeeze` parameter to wrapped methods
        - When squeeze=True (default), output arrays are squeezed and 0-d arrays
          are converted to scalar values
        - Original method behavior is preserved when squeeze=False
    
    Args:
        mcs: The metaclass instance
        name (str): Name of the class being created
        bases (tuple): Base classes
        namespace (dict): Class namespace dictionary
        **kwargs: Additional keyword arguments passed to type(BaseModel)
    
    Returns:
        (type): The created class with enhanced array handling capabilities
    """
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Wrap distribution methods with squeezable if they're defined in this class
        for method in ['sample', 'pdf', 'cdf']:
            if method in namespace and callable(namespace[method]):
                namespace[method] = squeezable(namespace[method])
        # Pass through all keyword arguments to support pydantic's generic handling
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class RandomVariable(NamedObject, Generic[T], metaclass=RandomVariableMeta):
    """Base class for random variables.
    
    This class provides a common interface for random variable implementations.
    Subclasses must implement sample(), pdf(), and cdf() methods. The metaclass
    ensures these methods support dimension control via the squeeze parameter.
    
    The class is generic over the type of values it produces (T), which must be
    a subtype of SerializableValue to ensure proper serialization behavior.

    Attributes:
        _registry_category (str): Category name for object registration
        seed (Optional[int]): Random seed for reproducible sampling
        name (str): Identifier for this random variable instance
    
    Type Variables:
        T: The type of values produced by this random variable
    """
    _registry_category = "random_variables"
    seed: Optional[int] = None

    def sample(self, size: int = 1, **kwargs) -> NDArray[Any, T]:
        """Generate random samples from the categorical distribution.

        Args:
            size (int): Number of samples to generate. Defaults to 1.
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.

        Returns:
            (NDArray[Any, T]): Array of samples from the categories
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def pdf(self, x: np.ndarray, **kwargs) -> NDArray[Any, T]:
        """Evaluate probability density function at specified points.
        
        Args:
            x (np.ndarray): Points at which to evaluate the PDF
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
            
        Returns:
            (NDArray[Any, T]): PDF values at the input points
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def cdf(self, x: np.ndarray, **kwargs) -> NDArray[Any, T]:
        """Evaluate cumulative distribution function at specified points.
        
        Args:
            x (np.ndarray): Points at which to evaluate the CDF
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
            
        Returns:
            (NDArray[Any, T]): CDF values at the input points
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def register_to_hash(
            self, 
            var_hash: RandomVariableHash, 
            size: int = 1, 
            sample: bool = True,
            squeeze: bool = True,
        ) -> NamedValue[T|NDArray[Any,T]]:
        """Register this random variable to a hash and return sampled values.
        
        This is a convenience method for adding a random variable to a collection
        and immediately sampling from it.

        Args:
            var_hash (RandomVariableHash): Hash object to register to
            size (int): Number of samples to generate

        Returns:
            (NamedValue[T|NDArray[Any,T]]): Named value containing samples
        """
        return var_hash.register_variable(self, size=size, sample=sample, squeeze=squeeze)

class NormalDistribution(RandomVariable[float]):
    """Normal (Gaussian) distribution with mean μ and standard deviation σ.
    
    The normal distribution is a continuous probability distribution that is
    symmetric about its mean, showing the familiar bell-shaped curve.
    
    Key Properties:
        - Symmetric about the mean
        - ~68% of values lie within 1σ of μ
        - ~95% lie within 2σ of μ
        - ~99.7% lie within 3σ of μ
    
    Attributes:
        mu (float): Mean (μ) of the distribution
        sigma (float): Standard deviation (σ) of the distribution
        name (str): Identifier for this distribution instance
        seed (Optional[int]): Random seed for reproducible sampling
    """
    mu: float
    sigma: float

    def sample(self, size: int | tuple[int, ...] = 1, **kwargs) -> NDArray[Any,float]:
        """Generate random samples from the normal distribution.
        
        Args:
            size (int | tuple[int, ...]): Number or shape of samples
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
        
        Returns:
            NDArray[Any,float]: Array of samples from N(μ, σ)
        """
        rng = np.random.default_rng(self.seed)
        return rng.normal(self.mu, self.sigma, size=size)

    def pdf(self, x: np.ndarray, **kwargs) -> NDArray[Any,float]:
        """Evaluate the normal probability density function.
        
        The PDF is given by:
        f(x) = 1/(σ√(2π)) * exp(-(x-μ)²/(2σ²))
        
        Args:
            x (np.ndarray): Points at which to evaluate the PDF
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
        
        Returns:
            NDArray[Any,float]: PDF values at the input points
        """
        return 1/(self.sigma * np.sqrt(2*np.pi)) * np.exp(-(x - self.mu)**2 / (2*self.sigma**2))

    def cdf(self, x: np.ndarray, **kwargs) -> NDArray[Any,float]:
        """Evaluate the normal cumulative distribution function.
        
        The CDF is computed using the error function:
        F(x) = 1/2 * (1 + erf((x-μ)/(σ√2)))
        
        Args:
            x (np.ndarray): Points at which to evaluate the CDF
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
        
        Returns:
            NDArray[Any,float]: CDF values at the input points
        """
        return 0.5 * (1 + sp.erf((x - self.mu)/(self.sigma * np.sqrt(2))))

class UniformDistribution(RandomVariable[float]):
    """Continuous uniform distribution over an interval [low, high].
    
    The uniform distribution describes equal probability over a continuous
    interval. Any value between low and high is equally likely to be drawn.
    
    Key Properties:
        - Mean = (low + high)/2
        - Variance = (high - low)²/12
        - Constant PDF over [low, high]
        - Linear CDF over [low, high]
    
    Attributes:
        low (float): Lower bound of the interval
        high (float): Upper bound of the interval
        name (str): Identifier for this distribution instance
        seed (Optional[int]): Random seed for reproducible sampling
    """
    low: float
    high: float

    @model_validator(mode='after')
    def validate_bounds(self) -> UniformDistribution:
        """Validate that high > low."""
        if self.high <= self.low:
            raise ValueError(f"Upper bound ({self.high}) must be greater than lower bound ({self.low})")
        return self

    def sample(self, size: int = 1, **kwargs) -> NDArray[Any,float]:
        """Generate random samples from the uniform distribution.
        
        Args:
            size (int): Number of samples to generate
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
        
        Returns:
            NDArray[Any,float]: Array of samples from U(low,high)
        """
        rng = np.random.default_rng(self.seed)
        return rng.uniform(self.low, self.high, size=size)

    def pdf(self, x: np.ndarray, **kwargs) -> NDArray[Any,float]:
        """Evaluate the uniform probability density function.
        
        The PDF is 1/(high-low) for x in [low,high] and 0 elsewhere.
        
        Args:
            x (np.ndarray): Points at which to evaluate the PDF
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
        
        Returns:
            NDArray[Any,float]: PDF values at the input points
        """
        return np.where(
            (x >= self.low) & (x <= self.high),
            1.0 / (self.high - self.low),
            0.0
        )

    def cdf(self, x: np.ndarray, **kwargs) -> NDArray[Any,float]:
        """Evaluate the uniform cumulative distribution function.
        
        The CDF is:
        - 0 for x < low
        - (x-low)/(high-low) for low ≤ x ≤ high
        - 1 for x > high
        
        Args:
            x (np.ndarray): Points at which to evaluate the CDF
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
        
        Returns:
            NDArray[Any,float]: CDF values at the input points
        """
        return np.where(
            x < self.low,
            0.0,
            np.where(
                x > self.high,
                1.0,
                (x - self.low) / (self.high - self.low)
            )
        )

class CategoricalDistribution(RandomVariable[T]):
    """Categorical distribution for discrete outcomes with specified probabilities.
    
    A categorical distribution (also called a discrete distribution) describes
    the probability of obtaining one of k possible outcomes. Each outcome has
    a probability between 0 and 1, and all probabilities must sum to 1.
    
    If probabilities are not specified, defaults to equal probabilities for
    all categories (uniform discrete distribution).
    
    Key Properties:
        - Support is finite set of categories
        - PMF gives probability of each category
        - CDF is step function
    
    Attributes:
        categories (np.ndarray): Array of possible outcomes (any type)
        probabilities (np.ndarray): Probability for each category
        name (str): Identifier for this distribution instance
        seed (Optional[int]): Random seed for reproducible sampling
        replace (bool): Whether or not to allow multiple draws of the same
            value (allowed if True)
    
    Raises:
        ValueError: If probabilities don't sum to 1
        ValueError: If lengths of categories and probabilities don't match
        ValueError: If any probability is negative
    """
    categories: NDArray | Iterable
    probabilities: Optional[NDArray | Iterable] = None
    replace: bool = True

    @model_validator(mode='after')
    def validate_and_set_probabilities(self) -> CategoricalDistribution:
        """Validate probability values and set defaults if needed."""
        if self.probabilities is None:
            n_categories = len(self.categories)
            self.probabilities = np.ones(n_categories) / n_categories
            return self

        if len(self.categories) != len(self.probabilities):
            raise ValueError(
                f"Number of categories ({len(self.categories)}) must match "
                f"number of probabilities ({len(self.probabilities)})"
            )
        if not np.all(self.probabilities >= 0):
            raise ValueError("All probabilities must be non-negative")
        if not np.isclose(np.sum(self.probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1")
        return self

    def sample(self, size: int = 1, **kwargs) -> NDArray[Any,T]:
        """Generate random samples from the categorical distribution.
        
        Args:
            size (int, optional): Number of samples to generate. Defaults to 1.
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.

        Notes:
            The squeeze parameter is added automatically by the metaclass and does not appear
            in the function signature, but can be passed as a keyword argument.
        
        Returns:
            NDArray[Any,T]: Array of samples from the categories
        """
        rng = np.random.default_rng(self.seed)
        return rng.choice(self.categories, size=size, p=self.probabilities, replace=self.replace)

    def pdf(self, x: np.ndarray, **kwargs) -> NDArray[Any,float]:
        """Evaluate the probability mass function (PMF).
        
        For categorical distributions, this gives the probability of
        each category occurring.
        
        Args:
            x (np.ndarray): Points at which to evaluate the PMF
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
        
        Returns:
            NDArray[Any,float]: Probability of each input value
        """
        return np.array([
            self.probabilities[self.categories == val].item()
            if val in self.categories else 0.0
            for val in x
        ])

    def cdf(self, x: np.ndarray, **kwargs) -> NDArray[Any,float]:
        """Evaluate the cumulative distribution function.
        
        For categorical distributions, this is a step function that
        increases at each category by that category's probability.
        
        Args:
            x (np.ndarray): Points at which to evaluate the CDF
        
        Other Parameters:
            squeeze (bool, optional): Whether to remove unnecessary dimensions.
                Added by RandomVariableMeta. Defaults is `True`.
        
        Returns:
            NDArray[Any,float]: CDF values at the input points
        """
        return np.array([
            np.sum(self.probabilities[self.categories <= val])
            for val in x
        ])
