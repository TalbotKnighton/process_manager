from typing import Any, Callable, List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """
    Defines a validation rule for data.
    
    Attributes:
        name: Name of the validation rule
        validator: Function that performs validation
        error_message: Optional custom error message
    """
    name: str
    validator: Callable[[Any], bool]
    error_message: Optional[str] = None

class DataValidator:
    """
    Validates data against a set of rules.
    
    Features:
    - Multiple validation rules
    - Custom error messages
    - Validation reporting
    """
    
    def __init__(self, rules: List[ValidationRule]):
        """
        Initialize validator with rules.
        
        Args:
            rules: List of validation rules to apply
        """
        self.rules = rules
    
    def validate(self, data: Any) -> List[str]:
        """
        Validate data against all rules.
        
        Args:
            data: Data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        for rule in self.rules:
            try:
                if not rule.validator(data):
                    error_msg = (rule.error_message or 
                               f"Validation failed: {rule.name}")
                    errors.append(error_msg)
                    logger.warning(f"Validation error: {error_msg}")
            except Exception as e:
                error_msg = f"Validation error in {rule.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return errors

class NumericValidator(DataValidator):
    """Specialized validator for numeric data"""
    
    @staticmethod
    def create_range_validator(
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> DataValidator:
        """Create validator for numeric range"""
        rules = []
        
        if min_val is not None:
            rules.append(ValidationRule(
                name="minimum_value",
                validator=lambda x: x >= min_val,
                error_message=f"Value must be >= {min_val}"
            ))
        
        if max_val is not None:
            rules.append(ValidationRule(
                name="maximum_value",
                validator=lambda x: x <= max_val,
                error_message=f"Value must be <= {max_val}"
            ))
        
        return DataValidator(rules)
    
    @staticmethod
    def create_statistical_validator(
        mean: Optional[float] = None,
        std_dev: Optional[float] = None,
        n_sigmas: float = 3.0
    ) -> DataValidator:
        """Create validator for statistical bounds"""
        rules = []
        
        if mean is not None and std_dev is not None:
            rules.append(ValidationRule(
                name="statistical_bounds",
                validator=lambda x: (
                    abs(x - mean) <= n_sigmas * std_dev
                ),
                error_message=f"Value outside {n_sigmas} sigma range"
            ))
        
        return DataValidator(rules)

class DictionaryValidator(DataValidator):
    """Specialized validator for dictionary data"""
    
    @staticmethod
    def create_required_keys_validator(
        required_keys: List[str]
    ) -> DataValidator:
        """Create validator for required dictionary keys"""
        return DataValidator([
            ValidationRule(
                name="required_keys",
                validator=lambda d: all(key in d for key in required_keys),
                error_message=f"Missing required keys: {required_keys}"
            )
        ])
    
    @staticmethod
    def create_type_validator(
        key_types: Dict[str, type]
    ) -> DataValidator:
        """Create validator for dictionary value types"""
        return DataValidator([
            ValidationRule(
                name=f"type_{key}",
                validator=lambda d: isinstance(d.get(key), type_),
                error_message=f"{key} must be of type {type_.__name__}"
            )
            for key, type_ in key_types.items()
        ])
