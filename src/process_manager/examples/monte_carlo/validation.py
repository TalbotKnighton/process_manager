from typing import Any, Callable, List, Dict, Optional
from dataclasses import dataclass
from process_manager.data_handling.data_validator import DataValidator, ValidationRule
import numpy as np

@dataclass
class PhysicalBounds:
    """
    Defines physical bounds for simulation parameters.
    
    Attributes:
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        units: Physical units for the parameter
        description: Description of the parameter
    """
    min_value: float
    max_value: float
    units: str
    description: str

class SimulationValidator:
    """
    Validates simulation parameters against physical and numerical constraints.
    
    This class provides comprehensive validation for simulation parameters,
    including:
    - Physical bounds checking
    - Consistency validation
    - Numerical stability checks
    """
    
    # Physical bounds for parameters
    PARAMETER_BOUNDS = {
        "temperature": PhysicalBounds(
            min_value=273.0,
            max_value=373.0,
            units="K",
            description="System temperature"
        ),
        "pressure": PhysicalBounds(
            min_value=1e5,
            max_value=5e5,
            units="Pa",
            description="System pressure"
        ),
        "flow_rate": PhysicalBounds(
            min_value=0.1,
            max_value=1.0,
            units="m³/s",
            description="Volumetric flow rate"
        ),
        "sim_time": PhysicalBounds(
            min_value=10.0,
            max_value=100.0,
            units="s",
            description="Simulation duration"
        )
    }
    
    def __init__(self):
        """Initialize validator with all validation rules"""
        self.validators = {
            param: self._create_validator(bounds)
            for param, bounds in self.PARAMETER_BOUNDS.items()
        }
    
    def _create_validator(self, bounds: PhysicalBounds) -> DataValidator:
        """Create a validator for a parameter with given bounds"""
        return DataValidator([
            ValidationRule(
                "range_check",
                lambda x: bounds.min_value <= x <= bounds.max_value,
                f"Value must be between {bounds.min_value} and {bounds.max_value} {bounds.units}"
            ),
            ValidationRule(
                "nan_check",
                lambda x: not np.isnan(x),
                "Value cannot be NaN"
            ),
            ValidationRule(
                "inf_check",
                lambda x: not np.isinf(x),
                "Value cannot be infinite"
            )
        ])
    
    def validate_parameter(self, param_name: str, value: float) -> List[str]:
        """
        Validate a single parameter value.
        
        Args:
            param_name: Name of the parameter to validate
            value: Value to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        if param_name not in self.validators:
            return [f"Unknown parameter: {param_name}"]
        
        return self.validators[param_name].validate({"value": value})
    
    def validate_params(self, params: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Validate all parameters in a parameter set.
        
        Args:
            params: Dictionary of parameter names and values
            
        Returns:
            Dictionary mapping parameter names to lists of validation errors
        """
        errors = {}
        for param_name, value in params.items():
            param_errors = self.validate_parameter(param_name, value)
            if param_errors:
                errors[param_name] = param_errors
        return errors
    
    def validate_simulation_case(self, case: 'SimulationCase') -> List[str]:
        """
        Validate a complete simulation case including consistency checks.
        
        Args:
            case: SimulationCase to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate individual parameters
        param_errors = self.validate_params(case.params.to_dict())
        for param, param_errors in param_errors.items():
            errors.extend(f"{param}: {error}" for error in param_errors)
        
        # Consistency checks
        if case.params.temperature > 373 and case.params.pressure > 4e5:
            errors.append(
                "Invalid combination: High temperature and pressure may cause instability"
            )
        
        # Numerical stability checks
        reynolds_number = self._calculate_reynolds_number(case.params)
        if reynolds_number > 1e5:
            errors.append(
                f"Warning: High Reynolds number ({reynolds_number:.2e}) may cause instability"
            )
        
        return errors
    
    def _calculate_reynolds_number(self, params: 'SimulationParams') -> float:
        """Calculate Reynolds number for stability checking"""
        # Simplified calculation - replace with actual fluid properties
        density = 1000  # kg/m³ (water)
        viscosity = 1e-3  # Pa·s (water)
        characteristic_length = 0.1  # m (example pipe diameter)
        
        velocity = params.flow_rate / (np.pi * (characteristic_length/2)**2)
        return (density * velocity * characteristic_length) / viscosity

