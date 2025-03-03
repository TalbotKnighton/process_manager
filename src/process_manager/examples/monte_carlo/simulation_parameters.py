"""Parameter definitions for Monte Carlo simulations using data_handlers framework."""
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from process_manager.data_handlers.values import NamedValue, SerializableValue
from process_manager.data_handlers.random_variables import (
    UniformDistribution,
    RandomVariableHash,
)

class SimulationState(Enum):
    """Tracks the state of individual simulation cases"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class SimulationParams:
    """Generator for simulation parameters using the data_handlers framework."""
    
    def __init__(self, 
                 seed: Optional[int] = None,
                 validate: bool = True):
        """Initialize parameter distributions.
        
        Args:
            seed (Optional[int]): Random seed for reproducibility
            validate (bool): Whether to validate parameters
        """
        self.seed = seed
        self.validate = validate
        self._initialize_distributions()
    
    def _initialize_distributions(self):
        """Setup random variable distributions with physical bounds."""
        self.variables = RandomVariableHash()
        
        # Temperature distribution (273-373K)
        self.temperature = UniformDistribution(
            name="temperature",
            low=273.0,
            high=373.0,
            seed=self.seed
        ).register_to_hash(self.variables)
        
        # Pressure distribution (1e5-5e5 Pa)
        self.pressure = UniformDistribution(
            name="pressure",
            low=1e5,
            high=5e5,
            seed=self.seed
        ).register_to_hash(self.variables)
        
        # Flow rate distribution (0.1-1.0 m³/s)
        self.flow_rate = UniformDistribution(
            name="flow_rate",
            low=0.1,
            high=1.0,
            seed=self.seed
        ).register_to_hash(self.variables)
        
        # Simulation time distribution (10-100s)
        self.sim_time = UniformDistribution(
            name="sim_time",
            low=10.0,
            high=100.0,
            seed=self.seed
        ).register_to_hash(self.variables)

    def generate(self, size: int = 1) -> Dict[str, float]:
        """Generate a set of simulation parameters.
        
        Args:
            size (int): Number of parameter sets to generate
            
        Returns:
            Dict[str, float]: Generated parameter values
        """
        samples = self.variables.sample_all(size=size)
        
        if self.validate:
            self._validate_samples(samples)
            
        return samples
    
    def _validate_samples(self, samples: Dict[str, float]) -> None:
        """Validate generated parameters.
        
        Args:
            samples (Dict[str, float]): Generated parameter values
            
        Raises:
            ValueError: If parameters violate physical constraints
        """
        # High temperature & pressure constraint
        if samples["temperature"] > 373 and samples["pressure"] > 4e5:
            raise ValueError(
                "Invalid combination: High temperature and pressure may cause instability"
            )
        
        # Reynolds number check
        reynolds = self._calculate_reynolds_number(samples)
        if reynolds > 1e5:
            raise ValueError(
                f"Warning: High Reynolds number ({reynolds:.2e}) may cause instability"
            )
    
    def _calculate_reynolds_number(self, params: Dict[str, float]) -> float:
        """Calculate Reynolds number for stability checking."""
        # Physical properties (water)
        density = 1000  # kg/m³
        viscosity = 1e-3  # Pa·s
        char_length = 0.1  # m (example pipe diameter)
        
        velocity = params["flow_rate"] / (3.14159 * (char_length/2)**2)
        return (density * velocity * char_length) / viscosity

@dataclass
class SimulationCase(SerializableValue):
    """Represents a single simulation case.
    
    This class uses the SerializableValue base class from data_handlers
    to enable automatic serialization/deserialization.
    """
    case_id: str
    params: Dict[str, float]
    input_file: Path
    output_file: Optional[Path] = None
    state: SimulationState = SimulationState.PENDING
    error: Optional[str] = None
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert case to dictionary format."""
        return {
            "case_id": self.case_id,
            "params": self.params,
            "input_file": str(self.input_file),
            "output_file": str(self.output_file) if self.output_file else None,
            "state": self.state.value,
            "error": self.error,
            "cache_hit": self.cache_hit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationCase':
        """Create case from dictionary format."""
        return cls(
            case_id=data["case_id"],
            params=data["params"],
            input_file=Path(data["input_file"]),
            output_file=Path(data["output_file"]) if data["output_file"] else None,
            state=SimulationState(data["state"]),
            error=data["error"],
            cache_hit=data["cache_hit"]
        )