"""
Module for generating, sorting, and managing named values.
This uses pydantic dataclasses for JSON serialization to avoid overloading system memory.

The module provides a robust framework for managing named values with type safety, serialization,
and state management. It includes classes for individual named values, collections of named values
in both list and hash (dictionary) formats, and utilities for type validation and serialization.

Classes:
    NamedValueState: Enum for tracking the state of named values
    NamedValue: Base class for type-safe named values with state management
    NamedValueHash: Dictionary-like container for managing named values
    NamedValueList: List-like container for managing ordered named values

Types:
    SerializableValue: Union type defining all allowed value types
    T: Generic type variable bound to SerializableValue
"""
from __future__ import annotations

# Standard
from dataclasses import Field
from enum import Enum
from typing import Any, Iterable, List, Type, TypeVar, Union, Generic, ClassVar
from typing import get_origin, get_args, Union, Iterable
try:
    from typing import Self
except:
    from typing_extensions import Self
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

# External
import json
from numpydantic import NDArray
import numpy as np
import pandas as pd
from pydantic import (
    ConfigDict, 
    InstanceOf, 
    SerializeAsAny, 
    Field, 
    PrivateAttr,
)
# Local
from data_handlers.base import (
    NamedObject, 
    NamedObjectHash, 
    NamedObjectList,
    ObjectRegistry
)
from data_handlers.custom_serde_definitions.pandantic import PandasDataFrame, PandasSeries
from data_handlers.mixins import ArrayDunders

__all__ = [
    'SerializableValue',
    'NamedValueState',
    'NamedValue',
    'NamedValueList',
    'NamedValueHash',
]

SerializableValue = Union[
    PandasDataFrame,
    PandasSeries,
    NDArray,
    Iterable,
    float,
    int,
    bool,
    str,
    Any,  # TODO does this need to be restricted?
    None,
]

T = TypeVar('T', bound=SerializableValue)

class NamedValueState(str, Enum):
    """
    State enumeration for NamedValue objects.
    
    This enum tracks whether a named value has been set or remains unset. Once set,
    values are typically frozen unless explicitly forced to change.
    
    Attributes:
        UNSET: Indicates no value has been set yet
        SET: Indicates value has been set and is frozen
    """
    UNSET = "unset"
    SET = "set"

    def __str__(self) -> str:
        """
        Convert state to string representation.
        
        Returns:
            str: The state value as a string ("unset" or "set")
        """
        return self.value  # Returns just "unset" or "set"

    def __repr__(self) -> str:
        """
        Get string representation for debugging.
        
        Returns:
            str: The state value as a string ("unset" or "set")
        """
        return self.value  # Returns just "unset" or "set"

@ArrayDunders.mixin
class NamedValue(NamedObject, Generic[T]):
    """
    A named value container with type safety and state management.
    
    NamedValue provides a type-safe way to store and manage values with built-in
    state tracking, serialization, and validation. Values can be frozen after
    initial setting to prevent accidental modification.
    
    Type Parameters:
        T: The type of value to store, must be a SerializableValue
    
    Attributes:
        name (str): Unique identifier for the value
        _stored_value (T | NamedValueState): The actual stored value or UNSET state
        _state (NamedValueState): Current state of the value
        _type (type): Runtime type information for validation
        
    Properties:
        value (T): Access or modify the stored value
        
    Example:
        ```python
        # Create a named integer value
        count = NamedValue[int]("item_count")
        count.value = 42  # Sets and freezes the value
        print(count.value)  # Outputs: 42
        count.value = 50  # Raises ValueError - value is frozen
        count.force_set_value(50)  # Allows value change
        ```
    """
    
    _registry_category: ClassVar[str] = "values"
    
    name: str = Field(..., description="Name of the value")
    _stored_value: T | NamedValueState = PrivateAttr(default=NamedValueState.UNSET)  # Changed this
    _state: NamedValueState = PrivateAttr(default=NamedValueState.UNSET)
    _type: type = PrivateAttr()

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        protected_namespaces=()
    )

    # def __init__(self, name: str, value: T | None = None, **data):
    #     """
    #     Initialize a new NamedValue instance.
        
    #     Args:
    #         name (str): Unique identifier for this value
    #         value (T | None, optional): Initial value to store. Defaults to None.
    #         **data: Additional keyword arguments passed to parent class
            
    #     Note:
    #         If value is provided, it will be validated and set immediately.
    #         The value will be frozen after initial setting.
    #     """
    #     data.pop('stored_value', None)
    #     data.pop('_stored_value', None)
        
    #     super().__init__(name=name, **data)
    #     self._type = self._extract_value_type()
    #     object.__setattr__(self, '_stored_value', NamedValueState.UNSET)  # Explicitly set initial value
        
    #     if value is not None:
    #         self.value = value
    
    # def __init__(self, name: str, value: T | None = None, **data):
    #     print(f"Initializing NamedValue with class: {self.__class__}")
    #     print(f"Has __orig_class__: {hasattr(self, '__orig_class__')}")
    #     if hasattr(self, '__orig_class__'):
    #         print(f"__orig_class__: {self.__orig_class__}")
    #         print(f"__orig_class__.__args__: {self.__orig_class__.__args__}")
    #     print(f"Bases: {self.__class__.__bases__}")
    #     for base in self.__class__.__bases__:
    #         if hasattr(base, '__origin__'):
    #             print(f"Base origin: {base.__origin__}")
    #             if hasattr(base, '__args__'):
    #                 print(f"Base args: {base.__args__}")
        
    #     data.pop('stored_value', None)
    #     data.pop('_stored_value', None)
        
    #     super().__init__(name=name, **data)
    #     self._type = self._extract_value_type()
    #     print(f"Extracted type: {self._type}")
    #     object.__setattr__(self, '_stored_value', NamedValueState.UNSET)
        
    #     if value is not None:
    #         self.value = value
    
    def __init__(self, name: str, value: T | None = None, **data):
        # print(f"Initializing NamedValue with class: {self.__class__}")
        # print(f"Class __args__: {getattr(self.__class__, '__args__', None)}")
        # print(f"Bases: {self.__class__.__bases__}")
        # for base in self.__class__.__bases__:
        #     print(f"Base __args__: {getattr(base, '__args__', None)}")
        
        data.pop('stored_value', None)
        data.pop('_stored_value', None)
        
        super().__init__(name=name, **data)
        self._type = self._extract_value_type()
        # print(f"Extracted type: {self._type}")
        object.__setattr__(self, '_stored_value', NamedValueState.UNSET)
        
        if value is not None:
            self.value = value

    @property
    def value(self) -> T:
        """
        Get the stored value.
        
        Returns:
            T: The currently stored value
            
        Raises:
            ValueError: If attempting to access before a value has been set
            
        Note:
            This property provides read access to the stored value. Once set,
            the value is frozen and can only be changed using force_set_value().
        """
        if self._state == NamedValueState.UNSET or self._stored_value is NamedValueState.UNSET:
            raise ValueError(f"Value '{self.name}' has not been set yet.")
        return self._stored_value

    @value.setter
    def value(self, new_value: T):
        """
        Set the value if it hasn't been set before.
        
        Args:
            new_value (T): Value to store
            
        Raises:
            ValueError: If value has already been set (frozen)
            TypeError: If value doesn't match the expected type T
            
        Note:
            Once set, the value becomes frozen and can only be changed
            using force_set_value().
        """
        # print('\n\n\n\n\n\n\nSETTER CALLED!!!!!!\n\n\n\n\n\n')
        if self._state == NamedValueState.SET:
            raise ValueError(
                f"Value '{self.name}' has already been set and is frozen. "
                "Use force_set_value() if you need to override it."
            )
        
        validated_value = self._validate_type(new_value)
        object.__setattr__(self, '_stored_value', validated_value)
        object.__setattr__(self, '_state', NamedValueState.SET)

    def force_set_value(self, new_value: T) -> None:
        """
        Force set the value regardless of its current state.
        
        This method bypasses the normal freezing mechanism and allows
        changing an already-set value.
        
        Args:
            new_value (T): New value to store
            
        Raises:
            TypeError: If value doesn't match the expected type T
        """
        object.__setattr__(self, '_stored_value', NamedValueState.UNSET)
        object.__setattr__(self, '_state', NamedValueState.UNSET)

        # if new_value == 'not a series':
        #     breakpoint()
        self.value = new_value
    
    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Custom serialization to include value state and stored value.
        
        Extends the parent class serialization to include the value's state
        and stored value (if set) in the serialized data.
        
        Args:
            **kwargs: Additional arguments passed to parent serialization
            
        Returns:
            dict[str, Any]: Dictionary containing serialized state
            
        Example:
            ```python
            value = NamedValue("example", 42)
            data = value.model_dump()
            print(data)  # Contains 'state' and 'stored_value'
            ```
        """
        data = super().model_dump(**kwargs)
        data['state'] = self._state
        if self._state == NamedValueState.SET:
            data['stored_value'] = self._stored_value
        return data

    @classmethod
    def model_validate(cls, data: Any) -> NamedValue:
        """
        Custom deserialization to restore value state and stored value.
        
        Reconstructs a NamedValue instance from serialized data, properly
        restoring both the value state and any stored value.
        
        Args:
            data (Any): Serialized data to deserialize
            
        Returns:
            NamedValue: New instance with restored state
            
        Example:
            ```python
            data = {'name': 'example', 'state': 'set', 'stored_value': 42}
            value = NamedValue.model_validate(data)
            print(value.value)  # Outputs: 42
            ```
        """
        if not isinstance(data, dict):
            return super().model_validate(data)

        data_copy = data.copy()
        state = NamedValueState(data_copy.pop('state', NamedValueState.UNSET))
        stored_value = data_copy.pop('stored_value', None)
        
        instance = super().model_validate(data_copy)
        
        # Only set the value if state was SET
        if state == NamedValueState.SET and stored_value is not None:
            instance.force_set_value(stored_value)
        
        return instance

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Prevent direct modification of protected attributes.
        
        Overrides attribute setting to prevent direct modification of internal
        state attributes. These attributes should only be modified through
        appropriate methods.
        
        Args:
            name (str): Name of the attribute to set
            value (Any): Value to set
            
        Raises:
            AttributeError: If attempting to modify protected attributes directly
            
        Example:
            ```python
            value = NamedValue("example")
            value._stored_value = 42  # Raises AttributeError
            ```
        """
        if name in ('_stored_value', '_state'):
            raise AttributeError(f"Cannot modify {name} directly. Use appropriate methods instead.")
        super().__setattr__(name, value)

    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     This method handles type validation and conversion, with special
    #     handling for numeric types and custom initialization.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated and possibly converted value
            
    #     Raises:
    #         TypeError: If value cannot be converted to type T
    #         ValueError: If value is invalid for custom initialized types
    #     """
    #     # Skip validation for Any type
    #     if self._type is Any:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # Determine if this is a custom subclass with its own __init__
    #     custom_init = self.__class__.__init__ is not NamedValue.__init__
    #     error_type = ValueError if custom_init else TypeError
            
    #     # If already correct type, return as is
    #     if isinstance(value, actual_type):
    #         return value
            
    #     try:
    #         # Handle numeric types explicitly
    #         if actual_type is int:
    #             if isinstance(value, str):
    #                 try:
    #                     # First try direct integer conversion
    #                     return int(value.strip())
    #                 except ValueError:
    #                     # If that fails, try float conversion
    #                     try:
    #                         float_val = float(value.strip())
    #                         # Check if float is actually an integer
    #                         if float_val.is_integer():
    #                             return int(float_val)
    #                     except ValueError:
    #                         raise error_type(
    #                             f"Value '{value}' cannot be converted to integer" if custom_init else
    #                             f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                             f"got {type(value).__name__} with value {value!r}"
    #                         )
    #             return int(value)
                
    #         elif actual_type is float:
    #             if isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise error_type(
    #                         f"Value '{value}' cannot be converted to float" if custom_init else
    #                         f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                         f"got {type(value).__name__} with value {value!r}"
    #                     )
    #             return float(value)
                
    #         # For all other types, try direct conversion
    #         try:
    #             converted = actual_type(value)
    #         except (ValueError, TypeError):
    #             raise error_type(
    #                 f"Value '{value}' cannot be converted to {actual_type.__name__}" if custom_init else
    #                 f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                 f"got {type(value).__name__} with value {value!r}"
    #             )
            
    #         # Verify the conversion worked
    #         if not isinstance(converted, actual_type):
    #             raise error_type(
    #                 f"Conversion failed to produce valid {actual_type.__name__}" if custom_init else
    #                 f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                 f"got {type(converted).__name__}"
    #             )
                
    #         return converted
            
    #     except (ValueError, TypeError) as e:
    #         # Only re-raise if it's already the correct error type
    #         if isinstance(e, error_type):
    #             raise e
    #         raise error_type(
    #             f"Value '{value}' cannot be converted to {actual_type.__name__}" if custom_init else
    #             f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #             f"got {type(value).__name__} with value {value!r}"
    #         )

    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated and possibly converted value
            
    #     Raises:
    #         TypeError: If value doesn't match explicitly specified type T
    #         ValueError: If value cannot be converted to type T
    #     """
    #     # Skip validation for Any type
    #     if self._type is Any:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # Determine if this is a custom subclass with its own __init__
    #     custom_init = self.__class__.__init__ is not NamedValue.__init__
        
    #     # If type was explicitly specified via generic (e.g., NamedValue[int]),
    #     # enforce strict type checking
    #     if hasattr(self, "__orig_class__"):
    #         if not isinstance(value, actual_type):
    #             raise TypeError(
    #                 f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                 f"got {type(value).__name__} with value {value!r}"
    #             )
    #         return value
            
    #     # For non-explicit types, attempt conversion as before
    #     try:
    #         # Handle numeric types explicitly
    #         if actual_type is int:
    #             if isinstance(value, str):
    #                 try:
    #                     return int(value.strip())
    #                 except ValueError:
    #                     try:
    #                         float_val = float(value.strip())
    #                         if float_val.is_integer():
    #                             return int(float_val)
    #                     except ValueError:
    #                         raise ValueError(f"Value '{value}' cannot be converted to integer")
    #             return int(value)
                
    #         elif actual_type is float:
    #             if isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise ValueError(f"Value '{value}' cannot be converted to float")
    #             return float(value)
                
    #         # For all other types, try direct conversion
    #         try:
    #             converted = actual_type(value)
    #         except (ValueError, TypeError):
    #             raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")
            
    #         # Verify the conversion worked
    #         if not isinstance(converted, actual_type):
    #             raise ValueError(f"Conversion failed to produce valid {actual_type.__name__}")
                
    #         return converted
            
    #     except (ValueError, TypeError) as e:
    #         raise ValueError(
    #             f"Value '{value}' cannot be converted to {actual_type.__name__}"
    #         )

    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated value
            
    #     Raises:
    #         TypeError: If value doesn't match explicitly specified type T
    #         ValueError: If value is invalid for the target type
    #     """
    #     # Skip validation for Any type
    #     if self._type is Any:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # Check if type was explicitly specified via generic (e.g., NamedValue[int])
    #     is_explicit_type = hasattr(self, "__orig_class__") or self.__class__ != NamedValue
        
    #     # For explicitly typed values, require exact type match
    #     if is_explicit_type:
    #         if not isinstance(value, actual_type):
    #             raise TypeError(
    #                 f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                 f"got {type(value).__name__} with value {value!r}"
    #             )
    #         return value
            
    #     # For non-explicit types, use the original conversion logic
    #     if actual_type is int:
    #         try:
    #             return int(value)
    #         except (ValueError, TypeError):
    #             raise ValueError(f"Value '{value}' cannot be converted to integer")
                
    #     elif actual_type is float:
    #         try:
    #             return float(value)
    #         except (ValueError, TypeError):
    #             raise ValueError(f"Value '{value}' cannot be converted to float")
                
    #     elif actual_type is str:
    #         return str(value)
            
    #     # For all other types, try direct instantiation
    #     try:
    #         if isinstance(value, actual_type):
    #             return value
    #         return actual_type(value)
    #     except (ValueError, TypeError) as e:
    #         raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")


    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated and possibly converted value
            
    #     Raises:
    #         ValueError: If value cannot be converted to the target type
    #     """
    #     # Skip validation for Any type
    #     if self._type is Any:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # Handle numeric types with conversion
    #     if actual_type is int:
    #         try:
    #             if isinstance(value, str):
    #                 # Try direct integer conversion first
    #                 try:
    #                     return int(value.strip())
    #                 except ValueError:
    #                     # If that fails, try float conversion
    #                     float_val = float(value.strip())
    #                     if float_val.is_integer():
    #                         return int(float_val)
    #                     raise ValueError(f"Value '{value}' is not a valid integer")
    #             return int(value)
    #         except (ValueError, TypeError):
    #             raise ValueError(f"Value '{value}' cannot be converted to integer")
                
    #     elif actual_type is float:
    #         try:
    #             if isinstance(value, str):
    #                 return float(value.strip())
    #             return float(value)
    #         except (ValueError, TypeError):
    #             raise ValueError(f"Value '{value}' cannot be converted to float")
                
    #     elif actual_type is str:
    #         return str(value)
            
    #     # For all other types
    #     if isinstance(value, actual_type):
    #         return value
            
    #     # Try conversion for non-matching types
    #     try:
    #         converted = actual_type(value)
    #         if not isinstance(converted, actual_type):
    #             raise ValueError(
    #                 f"Conversion failed to produce valid {actual_type.__name__}"
    #             )
    #         return converted
    #     except (ValueError, TypeError) as e:
    #         raise ValueError(
    #             f"Value '{value}' cannot be converted to {actual_type.__name__}"
    #         )
    
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated and possibly converted value
            
    #     Raises:
    #         TypeError: If value is of wrong type and no conversion is possible
    #         ValueError: If value is of correct type but invalid, or conversion fails
    #     """
    #     # Skip validation for Any type
    #     if self._type is Any:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # Check if type was explicitly specified via generic (e.g., NamedValue[int])
    #     is_explicit_type = hasattr(self, "__orig_class__") or self.__class__ != NamedValue
        
    #     # For numeric types, attempt conversion
    #     if actual_type in (int, float):
    #         try:
    #             if isinstance(value, (int, float)):
    #                 # Direct numeric conversion
    #                 return actual_type(value)
    #             elif isinstance(value, str) and not is_explicit_type:
    #                 # Try string conversion only for non-explicit types
    #                 try:
    #                     if actual_type is int:
    #                         try:
    #                             return int(value.strip())
    #                         except ValueError:
    #                             float_val = float(value.strip())
    #                             if float_val.is_integer():
    #                                 return int(float_val)
    #                             raise ValueError(f"Value '{value}' is not a valid integer")
    #                     else:
    #                         return float(value.strip())
    #                 except ValueError as e:
    #                     raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")
    #             else:
    #                 # Wrong type and explicit typing
    #                 if is_explicit_type:
    #                     raise TypeError(
    #                         f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                         f"got {type(value).__name__} with value {value!r}"
    #                     )
    #                 raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")
    #         except (ValueError, TypeError) as e:
    #             if isinstance(e, TypeError):
    #                 raise
    #             raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")
                
    #     # For string type
    #     elif actual_type is str:
    #         if isinstance(value, str):
    #             return value
    #         if is_explicit_type:
    #             raise TypeError(
    #                 f"Value for '{self.name}' must be of type str, "
    #                 f"got {type(value).__name__} with value {value!r}"
    #             )
    #         return str(value)
            
    #     # For all other types
    #     else:
    #         if isinstance(value, actual_type):
    #             return value
    #         if is_explicit_type:
    #             raise TypeError(
    #                 f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                 f"got {type(value).__name__} with value {value!r}"
    #             )
            
    #         # Try conversion for non-explicit types
    #         try:
    #             converted = actual_type(value)
    #             if not isinstance(converted, actual_type):
    #                 raise ValueError(f"Conversion failed to produce valid {actual_type.__name__}")
    #             return converted
    #         except (ValueError, TypeError):
    #             raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")

    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated value
            
    #     Raises:
    #         TypeError: If value is of wrong type for explicitly typed values
    #         ValueError: If value cannot be converted for non-explicit types
    #     """
    #     # Skip validation for Any type
    #     if self._type is Any:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # Check if type was explicitly specified (via generic or subclass)
    #     is_explicit_type = hasattr(self, "__orig_class__") or self.__class__ != NamedValue
        
    #     # For explicitly typed values, require exact type match
    #     if is_explicit_type:
    #         if not isinstance(value, actual_type):
    #             raise TypeError(
    #                 f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                 f"got {type(value).__name__} with value {value!r}"
    #             )
    #         return value
        
    #     # For non-explicit types, attempt conversion
    #     try:
    #         if actual_type is int:
    #             return int(float(str(value).strip()))
    #         elif actual_type is float:
    #             return float(str(value).strip())
    #         elif actual_type is str:
    #             return str(value)
    #         elif isinstance(value, actual_type):
    #             return value
    #         else:
    #             return actual_type(value)
    #     except (ValueError, TypeError):
    #         raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated value
            
    #     Raises:
    #         TypeError: If value is of wrong type for explicitly typed values
    #         ValueError: If value cannot be converted for non-explicit types
    #     """
    #     # Skip validation for Any type
    #     if self._type is Any:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # For numeric types, always attempt conversion first
    #     if actual_type in (int, float):
    #         try:
    #             if isinstance(value, (int, float)):
    #                 return actual_type(value)
    #             elif isinstance(value, str):
    #                 # Try string conversion
    #                 try:
    #                     if actual_type is int:
    #                         # For integers, first try direct conversion
    #                         try:
    #                             return int(value.strip())
    #                         except ValueError:
    #                             # Then try through float for cases like "123.0"
    #                             float_val = float(value.strip())
    #                             if float_val.is_integer():
    #                                 return int(float_val)
    #                             raise ValueError(f"Value '{value}' is not a valid integer")
    #                     else:
    #                         return float(value.strip())
    #                 except ValueError:
    #                     raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")
    #             else:
    #                 raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")
    #         except ValueError:
    #             # For explicitly typed values, also raise TypeError for wrong types
    #             if hasattr(self, "__orig_class__") or self.__class__ != NamedValue:
    #                 raise TypeError(
    #                     f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                     f"got {type(value).__name__} with value {value!r}"
    #                 )
    #             raise
                
    #     # For string type
    #     elif actual_type is str:
    #         if isinstance(value, str):
    #             return value
    #         # For explicitly typed strings, don't allow conversion
    #         if hasattr(self, "__orig_class__") or self.__class__ != NamedValue:
    #             raise TypeError(
    #                 f"Value for '{self.name}' must be of type str, "
    #                 f"got {type(value).__name__} with value {value!r}"
    #             )
    #         return str(value)
            
    #     # For all other types
    #     else:
    #         if isinstance(value, actual_type):
    #             return value
                
    #         # For explicitly typed values, don't allow conversion
    #         if hasattr(self, "__orig_class__") or self.__class__ != NamedValue:
    #             raise TypeError(
    #                 f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #                 f"got {type(value).__name__} with value {value!r}"
    #             )
                
    #         # Try conversion for non-explicit types
    #         try:
    #             converted = actual_type(value)
    #             if not isinstance(converted, actual_type):
    #                 raise ValueError(f"Conversion failed to produce valid {actual_type.__name__}")
    #             return converted
    #         except (ValueError, TypeError):
    #             raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")
    
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated value
            
    #     Raises:
    #         TypeError: If value is of wrong type for explicitly typed values
    #         ValueError: If value cannot be converted for non-explicit types
    #     """
    #     # Skip validation for Any type
    #     if self._type is Any:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # Determine if this is explicitly typed
    #     is_explicit_type = hasattr(self, "__orig_class__") or self.__class__ != NamedValue
        
    #     # For non-explicit types, always try conversion
    #     if not is_explicit_type:
    #         try:
    #             if actual_type is int:
    #                 if isinstance(value, str):
    #                     value = value.strip()
    #                 return int(float(value))  # Handles both "123" and "123.0"
    #             elif actual_type is float:
    #                 if isinstance(value, str):
    #                     value = value.strip()
    #                 return float(value)
    #             elif actual_type is str:
    #                 return str(value)
    #             else:
    #                 return actual_type(value)
    #         except (ValueError, TypeError):
    #             raise ValueError(f"Value '{value}' cannot be converted to {actual_type.__name__}")
        
    #     # For explicitly typed values, require exact type match
    #     else:
    #         if isinstance(value, actual_type):
    #             return value
                
    #         # Special case: allow numeric conversions for explicitly typed numerics
    #         if actual_type in (int, float):
    #             try:
    #                 if actual_type is int:
    #                     if isinstance(value, str):
    #                         value = value.strip()
    #                         # Only allow pure integer strings
    #                         if not value.replace("-", "").isdigit():
    #                             raise TypeError
    #                     return int(value)
    #                 else:  # float
    #                     if isinstance(value, (int, float)):
    #                         return float(value)
    #                     raise TypeError
    #             except (ValueError, TypeError):
    #                 pass
                    
    #         raise TypeError(
    #             f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #             f"got {type(value).__name__} with value {value!r}"
    #         )

    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     For explicitly typed NamedValue[T], attempts to cast the value to type T.
    #     For untyped NamedValue, accepts any SerializableValue without conversion.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated and potentially converted value
            
    #     Raises:
    #         TypeError: If value cannot be cast to the explicit type T
    #     """
    #     # Skip validation for Any type or untyped NamedValue
    #     if self._type is Any or self.__class__ is NamedValue:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # Handle explicit typing (NamedValue[T])
    #     try:
    #         # Special handling for numeric types
    #         if actual_type is int:
    #             if isinstance(value, float):
    #                 if value.is_integer():
    #                     return int(value)
    #             return int(value)
    #         elif actual_type is float:
    #             return float(value)
    #         elif actual_type is str:
    #             return str(value)
    #         elif isinstance(value, actual_type):
    #             return value
    #         else:
    #             # Attempt general conversion
    #             converted = actual_type(value)
    #             if not isinstance(converted, actual_type):
    #                 raise TypeError
    #             return converted
    #     except (ValueError, TypeError):
    #         raise TypeError(
    #             f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #             f"got {type(value).__name__} with value {value!r}"
    #         )
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     For explicitly typed NamedValue[T], attempts to cast the value to type T.
    #     For untyped NamedValue, accepts any SerializableValue without conversion.
        
    #     Args:
    #         value (Any): Value to validate/convert
            
    #     Returns:
    #         T: Validated and potentially converted value
            
    #     Raises:
    #         TypeError: If value cannot be cast to the explicit type T
    #     """
    #     breakpoint()
    #     # Skip validation for Any type or untyped NamedValue
    #     if self._type is Any:
    #         return value

    #     actual_type = getattr(self._type, "__origin__", self._type)
        
    #     # Check if we should do type conversion (explicit type given)
    #     is_explicit_type = hasattr(self, "__orig_class__") or self.__class__ != NamedValue
        
    #     # For untyped NamedValue, accept any SerializableValue
    #     if not is_explicit_type:
    #         return value
            
    #     # Handle explicit typing (NamedValue[T])
    #     try:
    #         # Special handling for numeric types
    #         if actual_type is int:
    #             if isinstance(value, (int, float)):
    #                 if isinstance(value, float) and not value.is_integer():
    #                     raise TypeError("Float value has decimal part")
    #                 return int(value)
    #             elif isinstance(value, str):
    #                 value = value.strip()
    #                 try:
    #                     return int(value)
    #                 except ValueError:
    #                     # Try converting through float for scientific notation
    #                     float_val = float(value)
    #                     if float_val.is_integer():
    #                         return int(float_val)
    #                     raise TypeError("Float value has decimal part")
    #             raise TypeError
                
    #         elif actual_type is float:
    #             if isinstance(value, (int, float)):
    #                 return float(value)
    #             elif isinstance(value, str):
    #                 return float(value.strip())
    #             raise TypeError
                
    #         elif actual_type is str:
    #             return str(value)
                
    #         elif isinstance(value, actual_type):
    #             return value
                
    #         # Attempt general conversion
    #         converted = actual_type(value)
    #         if not isinstance(converted, actual_type):
    #             raise TypeError
    #         return converted
                
    #     except (ValueError, TypeError) as e:
    #         if isinstance(e, TypeError) and str(e):
    #             raise TypeError(str(e))
    #         raise TypeError(
    #             f"Value for '{self.name}' must be of type {actual_type.__name__}, "
    #             f"got {type(value).__name__} with value {value!r}"
    #         )
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     For explicitly typed NamedValue[T], attempts to cast the value to type T.
    #     For untyped NamedValue, accepts any SerializableValue without conversion.
    #     """
    #     expected_type = self._type
        
    #     # Skip validation for Any type
    #     if expected_type is Any:
    #         return value

    #     try:
    #         # Handle numeric types
    #         if expected_type is int:
    #             if isinstance(value, str):
    #                 return int(value.strip())
    #             return int(value)
    #         elif expected_type is float:
    #             if isinstance(value, str):
    #                 return float(value.strip())
    #             return float(value)
    #         elif expected_type is str:
    #             return str(value)
    #         elif isinstance(value, expected_type):
    #             return value
    #         else:
    #             # Try general conversion
    #             return expected_type(value)
    #     except (ValueError, TypeError):
    #         raise TypeError(
    #             f"Value for '{self.name}' must be of type {expected_type.__name__}, "
    #             f"got {type(value).__name__} with value {value!r}"
    #         )
        
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     For explicitly typed NamedValue[T], attempts to cast the value to type T.
    #     For untyped NamedValue, accepts any SerializableValue without conversion.
    #     """
    #     expected_type = self._type
        
    #     # Skip validation for Any type
    #     if expected_type is Any:
    #         return value

    #     try:
    #         # Handle numeric types
    #         if expected_type is int:
    #             if isinstance(value, float):
    #                 if not value.is_integer():
    #                     raise TypeError("Float value has decimal part")
    #                 return int(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     float_val = float(value.strip())
    #                     if not float_val.is_integer():
    #                         raise TypeError("Float value has decimal part")
    #                     return int(float_val)
    #                 except ValueError:
    #                     raise TypeError(f"Value '{value}' cannot be converted to integer")
    #             elif isinstance(value, int):
    #                 return value
    #             else:
    #                 raise TypeError(f"Cannot convert {type(value).__name__} to integer")
                    
    #         elif expected_type is float:
    #             if isinstance(value, (int, float)):
    #                 return float(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise TypeError(f"Value '{value}' cannot be converted to float")
    #             else:
    #                 raise TypeError(f"Cannot convert {type(value).__name__} to float")
                    
    #         elif expected_type is str:
    #             return str(value)
                
    #         elif isinstance(value, expected_type):
    #             return value
    #         else:
    #             # Try general conversion
    #             try:
    #                 converted = expected_type(value)
    #                 if not isinstance(converted, expected_type):
    #                     raise TypeError
    #                 return converted
    #             except (ValueError, TypeError):
    #                 raise TypeError(
    #                     f"Value for '{self.name}' must be of type {expected_type.__name__}, "
    #                     f"got {type(value).__name__} with value {value!r}"
    #                 )
    #     except TypeError as e:
    #         # Preserve specific error messages
    #         raise TypeError(str(e))
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     For explicitly typed NamedValue[T], attempts to cast the value to type T.
    #     For untyped NamedValue, accepts any SerializableValue without conversion.
    #     """
    #     expected_type = self._type
        
    #     # Skip validation for Any type
    #     if expected_type is Any:
    #         return value

    #     # Standard error message format
    #     type_error_msg = lambda got_type, got_value: (
    #         f"Value for '{self.name}' must be of type {expected_type.__name__}, "
    #         f"got {got_type.__name__} with value {got_value!r}"
    #     )

    #     try:
    #         # Handle numeric types
    #         if expected_type is int:
    #             if isinstance(value, float):
    #                 if not value.is_integer():
    #                     raise TypeError(type_error_msg(float, value))
    #                 return int(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     float_val = float(value.strip())
    #                     if not float_val.is_integer():
    #                         raise TypeError(type_error_msg(float, float_val))
    #                     return int(float_val)
    #                 except ValueError:
    #                     raise TypeError(type_error_msg(str, value))
    #             elif isinstance(value, int):
    #                 return value
    #             else:
    #                 raise TypeError(type_error_msg(type(value), value))
                    
    #         elif expected_type is float:
    #             if isinstance(value, (int, float)):
    #                 return float(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise TypeError(type_error_msg(str, value))
    #             else:
    #                 raise TypeError(type_error_msg(type(value), value))
                    
    #         elif expected_type is str:
    #             if isinstance(value, str):
    #                 return value
    #             raise TypeError(type_error_msg(type(value), value))
                
    #         elif isinstance(value, expected_type):
    #             return value
                
    #         # Try general conversion
    #         try:
    #             converted = expected_type(value)
    #             if not isinstance(converted, expected_type):
    #                 raise TypeError
    #             return converted
    #         except (ValueError, TypeError):
    #             raise TypeError(type_error_msg(type(value), value))
                
    #     except TypeError as e:
    #         # Always use the standard error message format
    #         if "must be of type" not in str(e):
    #             raise TypeError(type_error_msg(type(value), value))
    #         raise

    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
        
    #     For explicitly typed NamedValue[T], attempts to cast the value to type T.
    #     For untyped NamedValue, accepts any SerializableValue without conversion.
    #     """
    #     expected_type = self._type
        
    #     # Skip validation for Any type
    #     if expected_type is Any:
    #         return value

    #     # Standard error message format
    #     type_error_msg = lambda got_type, got_value: (
    #         f"Value for '{self.name}' must be of type {expected_type.__name__}, "
    #         f"got {got_type.__name__} with value {got_value!r}"
    #     )

    #     try:
    #         # Handle numeric types
    #         if expected_type is int:
    #             if isinstance(value, float):
    #                 if not value.is_integer():
    #                     raise TypeError("Float value has decimal part")  # Special case message
    #                 return int(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     float_val = float(value.strip())
    #                     if not float_val.is_integer():
    #                         raise TypeError("Float value has decimal part")  # Special case message
    #                     return int(float_val)
    #                 except ValueError:
    #                     raise TypeError(type_error_msg(str, value))
    #             elif isinstance(value, int):
    #                 return value
    #             else:
    #                 raise TypeError(type_error_msg(type(value), value))
                    
    #         elif expected_type is float:
    #             if isinstance(value, (int, float)):
    #                 return float(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise TypeError(type_error_msg(str, value))
    #             else:
    #                 raise TypeError(type_error_msg(type(value), value))
                    
    #         elif expected_type is str:
    #             if isinstance(value, str):
    #                 return value
    #             raise TypeError(type_error_msg(type(value), value))
                
    #         elif isinstance(value, expected_type):
    #             return value
                
    #         # Try general conversion
    #         try:
    #             converted = expected_type(value)
    #             if not isinstance(converted, expected_type):
    #                 raise TypeError
    #             return converted
    #         except (ValueError, TypeError):
    #             raise TypeError(type_error_msg(type(value), value))
                
    #     except TypeError as e:
    #         # Pass through the special case message
    #         if "Float value has decimal part" in str(e):
    #             raise
    #         # Use standard format for other type errors
    #         if "must be of type" not in str(e):
    #             raise TypeError(type_error_msg(type(value), value))
    #         raise
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type.
    #     """
    #     expected_type = self._type
        
    #     # Skip validation for Any type
    #     if expected_type is Any:
    #         return value

    #     # Standard error message format
    #     type_error_msg = lambda got_type, got_value: (
    #         f"Value for '{self.name}' must be of type {expected_type.__name__}, "
    #         f"got {got_type.__name__} with value {got_value!r}"
    #     )

    #     try:
    #         # Handle numeric types
    #         if expected_type is int:
    #             if isinstance(value, float):
    #                 if not value.is_integer():
    #                     raise TypeError("Float value has decimal part")  # Special case message
    #                 return int(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     float_val = float(value.strip())
    #                     if not float_val.is_integer():
    #                         raise TypeError("Float value has decimal part")  # Special case message
    #                     return int(float_val)
    #                 except ValueError:
    #                     raise TypeError(type_error_msg(str, value))
    #             elif isinstance(value, int):
    #                 return value
    #             else:
    #                 raise TypeError(type_error_msg(type(value), value))
                    
    #         elif expected_type is float:
    #             if isinstance(value, (int, float)):
    #                 return float(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise TypeError(type_error_msg(str, value))
    #             else:
    #                 raise TypeError(type_error_msg(type(value), value))
                    
    #         elif expected_type is str:
    #             # Always allow conversion to string
    #             return str(value)
                
    #         elif isinstance(value, expected_type):
    #             return value
                
    #         # Try general conversion
    #         try:
    #             converted = expected_type(value)
    #             if not isinstance(converted, expected_type):
    #                 raise TypeError
    #             return converted
    #         except (ValueError, TypeError):
    #             raise TypeError(type_error_msg(type(value), value))
                
    #     except TypeError as e:
    #         # Pass through the special case message
    #         if "Float value has decimal part" in str(e):
    #             raise
    #         # Use standard format for other type errors
    #         if "must be of type" not in str(e):
    #             raise TypeError(type_error_msg(type(value), value))
    #         raise
    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class generic definition.
        
    #     This method introspects the class definition to determine the concrete
    #     type that should be used for value validation. It handles both simple
    #     types and complex generic types.
        
    #     Returns:
    #         type: The extracted type parameter, or Any if not explicitly specified
            
    #     Example:
    #         ```python
    #         class IntValue(NamedValue[int]): pass
    #         value = IntValue("example")
    #         value._extract_value_type()  # Returns int
    #         ```
    #     """
    #     cls = self.__class__
        
    #     # Type mapping for common types
    #     TYPE_MAP = {
    #         'int': int,
    #         'float': float,
    #         'bool': bool,
    #         'str': str,
    #         'list': list,
    #         'dict': dict,
    #         'set': set,
    #         'tuple': tuple,
    #         # Add more types as needed
    #     }
        
    #     # First try to get from the class bases
    #     bases = cls.__bases__
    #     for base in bases:
    #         base_str = str(base)
    #         if 'NamedValue[' in base_str:
    #             # Extract the type from NamedValue[type]
    #             type_str = base_str.split('NamedValue[')[1].split(']')[0]
    #             if type_str in TYPE_MAP:
    #                 return TYPE_MAP[type_str]
                
    #             # Handle more complex types if needed
    #             # For example: List[int], Dict[str, int], etc.
        
    #     # Fallback to checking generic type parameters
    #     if hasattr(cls, '__orig_bases__'):
    #         for base in cls.__orig_bases__:
    #             if (hasattr(base, '__origin__') and 
    #                 base.__origin__ is NamedValue and 
    #                 len(base.__args__) > 0):
    #                 return base.__args__[0]
        
    #     return Any
    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class generic definition.
    #     """
    #     # First check if we have a generic type parameter
    #     if hasattr(self, "__orig_class__"):
    #         return self.__orig_class__.__args__[0]
        
    #     # Then check if we're a subclass with a type parameter
    #     if self.__class__ != NamedValue:
    #         for base in self.__class__.__bases__:
    #             if (hasattr(base, "__origin__") and 
    #                 base.__origin__ is NamedValue and 
    #                 hasattr(base, "__args__")):
    #                 return base.__args__[0]
        
    #     return Any

    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class generic definition.
    #     """
    #     # Get the class itself
    #     cls = self.__class__
        
    #     # If the class has __args__, it's a generic instantiation (NamedValue[int])
    #     if hasattr(cls, '__args__'):
    #         return cls.__args__[0]
        
    #     # Check the bases for type information
    #     for base in cls.__bases__:
    #         if hasattr(base, '__args__'):
    #             return base.__args__[0]
        
    #     # Default to Any if no type information found
    #     return Any

    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class generic definition.
    #     """
    #     # Get the class itself
    #     cls = self.__class__
        
    #     print(f"Debug - Class: {cls}")
    #     print(f"Debug - Class qualname: {cls.__qualname__}")
    #     print(f"Debug - Class origin: {getattr(cls, '__origin__', None)}")
    #     print(f"Debug - Class parameters: {getattr(cls, '__parameters__', None)}")
        
    #     # First try to get it from the class name for NamedValue[int]
    #     if 'NamedValue[int]' in cls.__qualname__:
    #         return int
    #     if 'NamedValue[float]' in cls.__qualname__:
    #         return float
    #     if 'NamedValue[str]' in cls.__qualname__:
    #         return str
        
    #     # Check bases for inherited classes
    #     for base in cls.__bases__:
    #         print(f"Debug - Base: {base}")
    #         print(f"Debug - Base qualname: {base.__qualname__}")
    #         print(f"Debug - Base origin: {getattr(base, '__origin__', None)}")
    #         print(f"Debug - Base parameters: {getattr(base, '__parameters__', None)}")
            
    #         if 'NamedValue[int]' in base.__qualname__:
    #             return int
    #         if 'NamedValue[float]' in base.__qualname__:
    #             return float
    #         if 'NamedValue[str]' in base.__qualname__:
    #             return str
        
    #     return Any
    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class's generic type information.
        
    #     This method determines the expected type for value validation by examining:
    #     1. The class name for direct generic instantiation (e.g., NamedValue[int])
    #     2. The base classes for inherited types (e.g., class IntegerValue(NamedValue[int]))
        
    #     The type information is extracted from the class's qualname because Python's generic
    #     type parameters are not directly accessible at runtime in a reliable way. Instead,
    #     we look for the type in the generated class name which includes the type parameter.
        
    #     Returns:
    #         type: The extracted type (int, float, str, etc.) or Any if no type is specified
            
    #     Examples:
    #         ```python
    #         v1 = NamedValue[int]("count")  # Creates class 'NamedValue[int]'
    #         v1._extract_value_type()  # Returns int

    #         class IntValue(NamedValue[int]): pass
    #         v2 = IntValue("count")  # Has base class 'NamedValue[int]'
    #         v2._extract_value_type()  # Returns int

    #         v3 = NamedValue("any")  # No type parameter
    #         v3._extract_value_type()  # Returns Any
    #         ```
    #     """
    #     # Get the class itself for type checking
    #     cls = self.__class__
        
    #     # # Debug information about the class structure
    #     # print(f"Debug - Class: {cls}")
    #     # print(f"Debug - Class qualname: {cls.__qualname__}")
    #     # print(f"Debug - Class origin: {getattr(cls, '__origin__', None)}")
    #     # print(f"Debug - Class parameters: {getattr(cls, '__parameters__', None)}")
        
    #     # First look for type in the class name (direct generic use)
    #     if 'NamedValue[int]' in cls.__qualname__:
    #         return int
    #     if 'NamedValue[float]' in cls.__qualname__:
    #         return float
    #     if 'NamedValue[str]' in cls.__qualname__:
    #         return str
        
    #     # Then check base classes for inherited types
    #     for base in cls.__bases__:
    #         # print(f"Debug - Base: {base}")
    #         # print(f"Debug - Base qualname: {base.__qualname__}")
    #         # print(f"Debug - Base origin: {getattr(base, '__origin__', None)}")
    #         # print(f"Debug - Base parameters: {getattr(base, '__parameters__', None)}")
            
    #         # Look for type in base class name
    #         if 'NamedValue[int]' in base.__qualname__:
    #             return int
    #         if 'NamedValue[float]' in base.__qualname__:
    #             return float
    #         if 'NamedValue[str]' in base.__qualname__:
    #             return str
        
    #     # Default to Any if no type specification found
    #     return Any

    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class's generic type information.
    #     """
    #     # Get the class itself
    #     cls = self.__class__
        
    #     # Map of built-in types
    #     TYPE_MAP = {
    #         'int': int,
    #         'float': float,
    #         'str': str,
    #         'Series': PandasSeries,  # Add pandas types
    #         'DataFrame': PandasDataFrame,
    #     }
        
    #     # Handle direct generic use (NamedValue[T])
    #     if '[' in cls.__qualname__:
    #         type_str = cls.__qualname__.split('[')[1].rstrip(']')
    #         if type_str in TYPE_MAP:
    #             return TYPE_MAP[type_str]
    #         # Try to find the type in the module namespace
    #         import sys
    #         for module_name in sys.modules:
    #             module = sys.modules[module_name]
    #             if hasattr(module, type_str):
    #                 return getattr(module, type_str)
        
    #     # Handle inherited classes
    #     for base in cls.__bases__:
    #         if '[' in base.__qualname__:
    #             type_str = base.__qualname__.split('[')[1].rstrip(']')
    #             if type_str in TYPE_MAP:
    #                 return TYPE_MAP[type_str]
    #             # Try to find the type in the module namespace
    #             import sys
    #             for module_name in sys.modules:
    #                 module = sys.modules[module_name]
    #                 if hasattr(module, type_str):
    #                     return getattr(module, type_str)
        
    #     return Any
    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class's generic type information.
    #     """
    #     # Get the class itself
    #     cls = self.__class__
        
    #     def find_annotated_type(type_str: str) -> type:
    #         """Find a type in imported modules, handling Annotated types"""
    #         # Try built-in types first
    #         TYPE_MAP = {
    #             'int': int,
    #             'float': float,
    #             'str': str,
    #             'Series': PandasSeries,
    #             'DataFrame': PandasDataFrame,
    #         }
    #         if type_str in TYPE_MAP:
    #             return TYPE_MAP[type_str]
                
    #         # Look in modules
    #         import sys
    #         for module_name in sys.modules:
    #             module = sys.modules[module_name]
    #             if hasattr(module, type_str):
    #                 found_type = getattr(module, type_str)
    #                 # If it's an Annotated type, return as is
    #                 if (hasattr(found_type, '__origin__') and 
    #                     found_type.__origin__ is Annotated):
    #                     return found_type
    #                 # Otherwise wrap in Annotated if needed
    #                 return found_type
    #         return None

    #     # Handle direct generic use (NamedValue[T])
    #     if '[' in cls.__qualname__:
    #         type_str = cls.__qualname__.split('[')[1].rstrip(']')
    #         found_type = find_annotated_type(type_str)
    #         if found_type is not None:
    #             return found_type

    #     # Handle inherited classes
    #     for base in cls.__bases__:
    #         if '[' in base.__qualname__:
    #             type_str = base.__qualname__.split('[')[1].rstrip(']')
    #             found_type = find_annotated_type(type_str)
    #             if found_type is not None:
    #                 return found_type

    #     return Any

    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class's generic type information.
        
    #     This method handles:
    #     1. Direct generic types (NamedValue[T])
    #     2. Inherited classes (class IntValue(NamedValue[int]))
    #     3. Annotated types and their variations
    #     4. Complex generic types
        
    #     Returns:
    #         type: The extracted type, with Annotated types resolved to their base type
    #     """

    #     def resolve_type_str(type_str: str) -> type | None:
    #         """
    #         Resolve a type string to an actual type, checking built-ins and modules.
    #         """
    #         # Check built-in types first
    #         builtins = {
    #             'int': int,
    #             'float': float,
    #             'str': str,
    #             'bool': bool,
    #             'list': list,
    #             'dict': dict,
    #             'tuple': tuple,
    #             'set': set,
    #         }
    #         if type_str in builtins:
    #             return builtins[type_str]
                
    #         # Look through all loaded modules
    #         import sys
    #         for module in sys.modules.values():
    #             if hasattr(module, type_str):
    #                 return getattr(module, type_str)
    #         return None

    #     def extract_type_from_annotation(cls_or_type: Any) -> type:
    #         """
    #         Extract the base type from any form of type annotation.
    #         Handles: Annotated, Generic, Union, and other typing constructs.
    #         """
    #         # Handle None type
    #         if cls_or_type is None:
    #             return type(None)
                
    #         # Handle Any type
    #         if cls_or_type is Any:
    #             return Any
                
    #         # If it's already a basic type, return it
    #         if isinstance(cls_or_type, type):
    #             return cls_or_type
            
    #         # Handle Annotated types
    #         if hasattr(cls_or_type, '__origin__'):
    #             if cls_or_type.__origin__ is Annotated:
    #                 # Get the first argument, which is the actual type
    #                 return extract_type_from_annotation(cls_or_type.__args__[0])
    #             # Handle other typing constructs (Union, Generic, etc)
    #             return extract_type_from_annotation(cls_or_type.__origin__)
                
    #         # Handle string representation (from class qualname)
    #         if isinstance(cls_or_type, str):
    #             resolved = resolve_type_str(cls_or_type)
    #             if resolved is not None:
    #                 return extract_type_from_annotation(resolved)
                    
    #         return Any

    #     def get_type_from_class(cls: type) -> type:
    #         """Extract type from a class's generic parameters or qualname."""
    #         breakpoint()
    #         # Check for direct generic parameters
    #         if hasattr(cls, '__orig_bases__'):
    #             for base in cls.__orig_bases__:
    #                 if (hasattr(base, '__origin__') and 
    #                     base.__origin__ is NamedValue and 
    #                     hasattr(base, '__args__')):
    #                     return extract_type_from_annotation(base.__args__[0])
            
    #         # Check qualname for type information
    #         if '[' in cls.__qualname__:
    #             type_str = cls.__qualname__.split('[')[1].rstrip(']')
    #             return extract_type_from_annotation(type_str)
                
    #         # Check base classes
    #         for base in cls.__bases__:
    #             if '[' in base.__qualname__:
    #                 type_str = base.__qualname__.split('[')[1].rstrip(']')
    #                 return extract_type_from_annotation(type_str)
                    
    #         return Any

    #     # Start type extraction process
    #     extracted_type = get_type_from_class(self.__class__)
    #     print(f"Debug - Original extracted type: {extracted_type}")
        
    #     # Resolve any remaining type annotations
    #     final_type = extract_type_from_annotation(extracted_type)
    #     print(f"Debug - Final resolved type: {final_type}")
        
    #     # breakpoint()
    #     return final_type

    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class's generic type information.
        
    #     This method handles:
    #     1. Direct generic types (NamedValue[T])
    #     2. Inherited classes (class IntValue(NamedValue[int]))
    #     3. Annotated types and their variations
    #     4. Complex generic types
    #     """
    #     def extract_type_from_annotation(ann_type: Any) -> type:
    #         """
    #         Extract the base type from any form of type annotation.
    #         """
    #         # Handle None or Any
    #         if ann_type is None:
    #             return type(None)
    #         if ann_type is Any:
    #             return Any
                
    #         # Handle Annotated types
    #         if hasattr(ann_type, '__origin__') and ann_type.__origin__ is Annotated:
    #             # Get the first argument, which is the actual type
    #             return extract_type_from_annotation(ann_type.__args__[0])
                
    #         # If it's already a type, return it
    #         if isinstance(ann_type, type):
    #             return ann_type
                
    #         # Handle other typing constructs
    #         if hasattr(ann_type, '__origin__'):
    #             return extract_type_from_annotation(ann_type.__origin__)
                
    #         return ann_type

    #     def get_generic_type(cls: type) -> type:
    #         """Get the generic type parameter from a class."""
    #         # First check if we have __orig_class__ (direct generic instantiation)
    #         if hasattr(cls, '__orig_class__'):
    #             orig = cls.__orig_class__
    #             if hasattr(orig, '__args__') and orig.__args__:
    #                 return extract_type_from_annotation(orig.__args__[0])
            
    #         # Then check __orig_bases__ (inherited or direct generic)
    #         if hasattr(cls, '__orig_bases__'):
    #             for base in cls.__orig_bases__:
    #                 if (hasattr(base, '__origin__') and 
    #                     base.__origin__ is NamedValue and 
    #                     hasattr(base, '__args__')):
    #                     return extract_type_from_annotation(base.__args__[0])
                        
    #         # Finally check regular bases
    #         for base in cls.__bases__:
    #             if hasattr(base, '__args__') and base.__args__:
    #                 return extract_type_from_annotation(base.__args__[0])
                    
    #         return Any

    #     # Get the type from generic parameters
    #     extracted_type = get_generic_type(self.__class__)
    #     print(f"Debug - Extracted type: {extracted_type}")
        
    #     # Resolve any remaining type annotations
    #     final_type = extract_type_from_annotation(extracted_type)
    #     print(f"Debug - Final resolved type: {final_type}")
        
    #     return final_type

    # def _extract_value_type(self) -> type:
    #     """
    #     Extract the type parameter T from the class's generic type information.
    #     """
    #     cls = self.__class__
    #     print(f"\nDebug - Class: {cls}")
    #     print(f"Debug - Class name: {cls.__name__}")
    #     print(f"Debug - Class qualname: {cls.__qualname__}")
    #     print(f"Debug - Has __orig_class__: {hasattr(cls, '__orig_class__')}")
    #     if hasattr(cls, '__orig_class__'):
    #         print(f"Debug - __orig_class__: {cls.__orig_class__}")
    #     print(f"Debug - Has __orig_bases__: {hasattr(cls, '__orig_bases__')}")
    #     if hasattr(cls, '__orig_bases__'):
    #         print(f"Debug - __orig_bases__: {cls.__orig_bases__}")
    #     print(f"Debug - Bases: {cls.__bases__}")
        
    #     def get_type_from_annotated(ann_type: Any) -> type:
    #         """Extract the actual type from an Annotated type."""
    #         if hasattr(ann_type, '__origin__') and ann_type.__origin__ is Annotated:
    #             return ann_type.__args__[0]
    #         return ann_type

    #     # Check for direct generic parameters (NamedValue[T])
    #     if hasattr(cls, '_name'):  # Generic alias
    #         print(f"Debug - Generic alias name: {cls._name}")
    #         print(f"Debug - Generic alias args: {cls.__args__}")
    #         return get_type_from_annotated(cls.__args__[0])

    #     # Check original class for inherited types
    #     if hasattr(cls, '__orig_bases__'):
    #         for base in cls.__orig_bases__:
    #             print(f"Debug - Checking base: {base}")
    #             if hasattr(base, '__origin__') and base.__origin__ is NamedValue:
    #                 if hasattr(base, '__args__'):
    #                     return get_type_from_annotated(base.__args__[0])

    #     # Check base classes
    #     for base in cls.__bases__:
    #         print(f"Debug - Checking base class: {base}")
    #         if hasattr(base, '__args__'):
    #             print(f"Debug - Base args: {base.__args__}")
    #             return get_type_from_annotated(base.__args__[0])
    #         # Check if it's a string representation
    #         if '[' in base.__qualname__:
    #             type_str = base.__qualname__.split('[')[1].rstrip(']')
    #             print(f"Debug - Found type string in base: {type_str}")
    #             # Handle Annotated types in string form
    #             if type_str.startswith('Annotated['):
    #                 # Extract the first type argument
    #                 actual_type = type_str[len('Annotated['):].split(',')[0]
    #                 print(f"Debug - Extracted from Annotated: {actual_type}")
    #                 if actual_type == 'DataFrame':
    #                     return pd.DataFrame
    #                 if actual_type == 'Series':
    #                     return pd.Series
    #                 if actual_type == 'int':
    #                     return int
    #                 if actual_type == 'float':
    #                     return float
    #                 if actual_type == 'str':
    #                     return str
    #             # Handle direct types
    #             elif type_str == 'int':
    #                 return int
    #             elif type_str == 'float':
    #                 return float
    #             elif type_str == 'str':
    #                 return str
    #             elif type_str == 'DataFrame':
    #                 return pd.DataFrame
    #             elif type_str == 'Series':
    #                 return pd.Series
    #             elif type_str == 'NDArray':
    #                 return np.ndarray

    #     print("Debug - Falling back to Any")
    #     return Any

    def _extract_value_type(self) -> type:
        """
        Extract the type parameter T from the class's generic type information.
        """
        cls = self.__class__

        def parse_type_str(type_str: str) -> type:
            """Parse a type string into an actual type."""
            # Handle Annotated types
            if type_str.startswith('Annotated['):
                # Extract the first type from Annotated[Type, ...]
                inner_types = type_str[len('Annotated['):].split(',')[0].strip()
                return parse_type_str(inner_types)
                
            # Map of type strings to actual types
            TYPE_MAP = {
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'DataFrame': pd.DataFrame,
                'Series': pd.Series,
                'NDArray': np.ndarray,
                'ndarray': np.ndarray,
                'PandasDataFrame': pd.DataFrame,
                'PandasSeries': pd.Series,
            }
            
            return TYPE_MAP.get(type_str, Any)

        # Parse the class name for type information
        name = cls.__qualname__
        if '[' in name:
            # Extract everything between the first [ and the last ]
            type_part = name.split('[', 1)[1].rsplit(']', 1)[0]
            # print(f"Debug - Extracted type part: {type_part}")
            return parse_type_str(type_part)
        
        # Check base classes for inherited types
        for base in cls.__bases__:
            if '[' in base.__qualname__:
                type_part = base.__qualname__.split('[', 1)[1].rsplit(']', 1)[0]
                # print(f"Debug - Extracted type part from base: {type_part}")
                return parse_type_str(type_part)

        return Any
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type T.
        
    #     This method implements type validation and conversion logic with different
    #     behavior depending on whether the NamedValue has an explicit type parameter:
        
    #     1. For explicitly typed values (NamedValue[T]):
    #         - Enforces strict type checking for most types
    #         - Allows specific conversions for numeric types:
    #             * Integers: Allows conversion from strings and integer-valued floats
    #             * Floats: Allows conversion from integers and numeric strings
    #         - Always allows conversion to string (matches Python's str() behavior)
            
    #     2. For untyped values (plain NamedValue):
    #         - Accepts any value that matches SerializableValue union
    #         - No type conversion is performed
            
    #     Special Cases:
    #         - Float to int conversion only works if float has no decimal part
    #         - String to numeric conversion only works for valid numeric strings
    #         - Any value can be converted to string
            
    #     Args:
    #         value (Any): The value to validate and potentially convert
            
    #     Returns:
    #         T: The validated and possibly converted value
            
    #     Raises:
    #         TypeError: When value cannot be converted to the expected type, with two formats:
    #             - "Float value has decimal part" for float->int conversion failures
    #             - "Value for '{name}' must be of type {type}" for other type mismatches
            
    #     Examples:
    #         ```python
    #         int_val = NamedValue[int]("count")
    #         int_val.value = "123"  # OK: converts to 123
    #         int_val.value = 45.0   # OK: converts to 45
    #         int_val.value = 45.7   # TypeError: Float value has decimal part

    #         str_val = NamedValue[str]("name")
    #         str_val.value = 123    # OK: converts to "123"

    #         any_val = NamedValue("any")
    #         any_val.value = "123"  # OK: keeps as string
    #         ```
    #     """
    #     expected_type = self._type
        
    #     # Skip validation for Any type (untyped NamedValue)
    #     if expected_type is Any:
    #         return value

    #     # Standard error message format for type mismatches
    #     type_error_msg = lambda got_type, got_value: (
    #         f"Value for '{self.name}' must be of type {expected_type.__name__}, "
    #         f"got {got_type.__name__} with value {got_value!r}"
    #     )

    #     try:
    #         # Handle numeric types with special conversion rules
    #         if expected_type is int:
    #             if isinstance(value, float):
    #                 # Only allow floats with no decimal part
    #                 if not value.is_integer():
    #                     raise TypeError("Float value has decimal part")
    #                 return int(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     # Try converting through float to handle scientific notation
    #                     float_val = float(value.strip())
    #                     if not float_val.is_integer():
    #                         raise TypeError("Float value has decimal part")
    #                     return int(float_val)
    #                 except ValueError:
    #                     raise TypeError(type_error_msg(str, value))
    #             elif isinstance(value, int):
    #                 return value
    #             else:
    #                 raise TypeError(type_error_msg(type(value), value))
                    
    #         elif expected_type is float:
    #             if isinstance(value, (int, float)):
    #                 # Integers can always convert to float
    #                 return float(value)
    #             elif isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise TypeError(type_error_msg(str, value))
    #             else:
    #                 raise TypeError(type_error_msg(type(value), value))
                    
    #         elif expected_type is str:
    #             # Special case: any value can be converted to string
    #             # This matches Python's built-in str() behavior
    #             return str(value)
                
    #         elif isinstance(value, expected_type):
    #             # Value is already the correct type
    #             return value
                
    #         # For other types, try general conversion
    #         try:
    #             converted = expected_type(value)
    #             if not isinstance(converted, expected_type):
    #                 raise TypeError
    #             return converted
    #         except (ValueError, TypeError):
    #             raise TypeError(type_error_msg(type(value), value))
                
    #     except TypeError as e:
    #         # Preserve the special case message for float->int conversion
    #         if "Float value has decimal part" in str(e):
    #             raise
    #         # Use standard format for other type errors
    #         if "must be of type" not in str(e):
    #             raise TypeError(type_error_msg(type(value), value))
    #         raise
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type T.
    #     """
    #     expected_type = self._type
        
    #     # Skip validation for Any type
    #     if expected_type is Any:
    #         return value

    #     def type_error(message: str = None) -> TypeError:
    #         if message:
    #             return TypeError(message)
    #         return TypeError(f"Cannot convert {type(value).__name__} to {expected_type.__name__}")

    #     try:
    #         # Special handling for built-in numeric types
    #         if expected_type is int:
    #             if isinstance(value, float):
    #                 if not value.is_integer():
    #                     raise type_error("Cannot convert decimal value to integer")
    #                 return int(value)
    #             if isinstance(value, str):
    #                 try:
    #                     float_val = float(value.strip())
    #                     if not float_val.is_integer():
    #                         raise type_error("Cannot convert decimal value to integer")
    #                     return int(float_val)
    #                 except ValueError:
    #                     raise type_error()
    #             if isinstance(value, int):
    #                 return value
    #             raise type_error()

    #         if expected_type is float:
    #             if isinstance(value, (int, float)):
    #                 return float(value)
    #             if isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise type_error()
    #             raise type_error()

    #         # Special case: anything can convert to string
    #         if expected_type is str:
    #             return str(value)

    #         # For all other types:
    #         if isinstance(value, expected_type):
    #             return value
                
    #         # Try conversion if type has constructor that takes this value
    #         try:
    #             result = expected_type(value)
    #             if not isinstance(result, expected_type):
    #                 raise type_error()
    #             return result
    #         except (ValueError, TypeError):
    #             raise type_error()

    #     except TypeError as e:
    #         raise type_error(str(e))
    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type T.
    #     """
    #     expected_type = self._type
        
    #     # Skip validation for Any type
    #     if expected_type is Any:
    #         return value

    #     def type_error(message: str = None) -> TypeError:
    #         if message:
    #             return TypeError(message)
    #         return TypeError(f"Cannot convert {type(value).__name__} to {get_type_name(expected_type)}")

    #     def get_type_name(t) -> str:
    #         """Get a readable name for a type, handling Annotated types"""
    #         # Handle Annotated[T, ...] types
    #         if hasattr(t, '__origin__') and t.__origin__ is Annotated:
    #             return t.__args__[0].__name__
    #         return getattr(t, '__name__', str(t))

    #     def is_instance_of_type(value: Any, expected: type) -> bool:
    #         """Check if value is an instance of expected type, handling Annotated types"""
    #         # Handle Annotated[T, ...] types
    #         if hasattr(expected, '__origin__') and expected.__origin__ is Annotated:
    #             actual_type = expected.__args__[0]
    #             return isinstance(value, actual_type)
    #         return isinstance(value, expected)

    #     try:
    #         # Special handling for built-in numeric types
    #         if expected_type is int:
    #             if isinstance(value, float):
    #                 if not value.is_integer():
    #                     raise type_error("Cannot convert decimal value to integer")
    #                 return int(value)
    #             if isinstance(value, str):
    #                 try:
    #                     float_val = float(value.strip())
    #                     if not float_val.is_integer():
    #                         raise type_error("Cannot convert decimal value to integer")
    #                     return int(float_val)
    #                 except ValueError:
    #                     raise type_error()
    #             if isinstance(value, int):
    #                 return value
    #             raise type_error()

    #         if expected_type is float:
    #             if isinstance(value, (int, float)):
    #                 return float(value)
    #             if isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise type_error()
    #             raise type_error()

    #         # Special case: anything can convert to string
    #         if expected_type is str:
    #             return str(value)

    #         # For all other types:
    #         if is_instance_of_type(value, expected_type):
    #             return value

    #         # Try conversion if type has constructor that takes this value
    #         try:
    #             # Get the actual type for conversion
    #             conversion_type = (expected_type.__args__[0] 
    #                             if hasattr(expected_type, '__origin__') 
    #                             and expected_type.__origin__ is Annotated 
    #                             else expected_type)
                
    #             result = conversion_type(value)
    #             if not is_instance_of_type(result, expected_type):
    #                 raise type_error()
    #             return result
    #         except (ValueError, TypeError):
    #             raise type_error()

    #     except TypeError as e:
    #         if "Subscripted generics" in str(e):
    #             # Handle the special case of generic type checking
    #             if is_instance_of_type(value, expected_type):
    #                 return value
    #             raise type_error()
    #         raise type_error(str(e))


    # def _validate_type(self, value: Any) -> T:
    #     """
    #     Validate and potentially convert a value to the expected type T.
        
    #     Handles all types in SerializableValue union:
    #     - PandasDataFrame (custom type and pd.DataFrame)
    #     - PandasSeries (custom type and pd.Series)
    #     - NDArray (and np.ndarray)
    #     - Iterable
    #     - float
    #     - int
    #     - bool
    #     - str
    #     - Any
    #     - None
    #     """
    #     expected_type = self._type
        
    #     # Skip validation for Any type or None
    #     if expected_type is Any:
    #         return value
    #     if expected_type is type(None) and value is None:
    #         return value

    #     def type_error(message: str = None) -> TypeError:
    #         if message:
    #             return TypeError(message)
    #         return TypeError(f"Cannot convert {type(value).__name__} to {get_base_type(expected_type).__name__}")

    #     def get_base_type(t: type) -> type:
    #         """Get the base type from an Annotated or other complex type"""
    #         if get_origin(t) is Annotated:
    #             return get_args(t)[0]
    #         return t

    #     def is_valid_type(value: Any, expected: type) -> bool:
    #         """Check if value matches the expected type, handling complex types"""
    #         base_type = get_base_type(expected)
            
    #         # Handle None
    #         if base_type is type(None):
    #             return value is None

    #         # Handle pandas DataFrame
    #         if base_type in (PandasDataFrame, pd.DataFrame):
    #             return isinstance(value, pd.DataFrame)
                
    #         # Handle pandas Series
    #         if base_type in (PandasSeries, pd.Series):
    #             return isinstance(value, pd.Series)
                
    #         # Handle numpy arrays
    #         if base_type in (NDArray, np.ndarray):
    #             return isinstance(value, np.ndarray)
                
    #         # Handle Iterable (but not strings)
    #         if base_type is Iterable and not isinstance(value, str):
    #             return hasattr(value, '__iter__')
                
    #         # Handle basic types
    #         try:
    #             return isinstance(value, base_type)
    #         except TypeError:
    #             # Fall back to type name comparison if isinstance fails
    #             return type(value).__name__ == base_type.__name__

    #     try:
    #         # Handle None type
    #         if expected_type is type(None):
    #             if value is not None:
    #                 raise type_error("Expected None value")
    #             return value

    #         # Special handling for built-in numeric types
    #         if expected_type is int:
    #             if isinstance(value, float):
    #                 if not value.is_integer():
    #                     raise type_error("Cannot convert decimal value to integer")
    #                 return int(value)
    #             if isinstance(value, (str, np.integer)):
    #                 try:
    #                     float_val = float(str(value).strip())
    #                     if not float_val.is_integer():
    #                         raise type_error("Cannot convert decimal value to integer")
    #                     return int(float_val)
    #                 except ValueError:
    #                     raise type_error()
    #             if isinstance(value, int):
    #                 return value
    #             raise type_error()

    #         if expected_type is float:
    #             if isinstance(value, (int, float, np.number)):
    #                 return float(value)
    #             if isinstance(value, str):
    #                 try:
    #                     return float(value.strip())
    #                 except ValueError:
    #                     raise type_error()
    #             raise type_error()

    #         if expected_type is bool:
    #             if isinstance(value, (bool, np.bool_)):
    #                 return bool(value)
    #             raise type_error()

    #         # Special case: anything can convert to string
    #         if expected_type is str:
    #             return str(value)

    #         # Handle pandas DataFrame
    #         base_type = get_base_type(expected_type)
    #         if base_type in (PandasDataFrame, pd.DataFrame):
    #             if isinstance(value, pd.DataFrame):
    #                 return value
    #             try:
    #                 return pd.DataFrame(value)
    #             except:
    #                 raise type_error()

    #         # Handle pandas Series
    #         if base_type in (PandasSeries, pd.Series):
    #             if isinstance(value, pd.Series):
    #                 return value
    #             try:
    #                 return pd.Series(value)
    #             except:
    #                 raise type_error()

    #         # Handle numpy arrays
    #         if base_type in (NDArray, np.ndarray):
    #             if isinstance(value, np.ndarray):
    #                 return value
    #             try:
    #                 return np.array(value)
    #             except:
    #                 raise type_error()

    #         # Handle Iterable
    #         if base_type is Iterable:
    #             if hasattr(value, '__iter__') and not isinstance(value, str):
    #                 return value
    #             raise type_error()

    #         # For all other types:
    #         if is_valid_type(value, expected_type):
    #             return value

    #         # Try general conversion
    #         try:
    #             result = base_type(value)
    #             if not is_valid_type(result, expected_type):
    #                 raise type_error()
    #             return result
    #         except (ValueError, TypeError):
    #             raise type_error()

    #     except TypeError as e:
    #         if "Subscripted generics" in str(e):
    #             if is_valid_type(value, expected_type):
    #                 return value
    #         raise type_error()

    def _validate_type(self, value: Any) -> T:
        """
        Validate and potentially convert a value to the expected type T.
        """
        expected_type = self._type
        
        # Skip validation for Any type or None
        if expected_type is Any:
            return value
        if expected_type is type(None) and value is None:
            return value

        def type_error(message: str = None) -> TypeError:
            if message:
                return TypeError(message)
            base_type = get_base_type(expected_type)
            type_name = getattr(base_type, '__name__', str(base_type))
            return TypeError(f"Cannot convert {type(value).__name__} to {type_name}")

        def get_base_type(t: type) -> type:
            """Get the base type from an Annotated or other complex type"""
            # Handle Annotated types
            if get_origin(t) is Annotated:
                return get_args(t)[0]
            # Handle other type origins (List, Union, etc)
            if hasattr(t, '__origin__'):
                return t.__origin__
            return t

        def get_actual_type(t: type) -> type:
            """Get the actual type to use for isinstance checks"""
            base = get_base_type(t)
            if base is PandasDataFrame:
                return pd.DataFrame
            if base is PandasSeries:
                return pd.Series
            if base is NDArray:
                return np.ndarray
            return base

        def is_valid_type(value: Any, expected: type) -> bool:
            """Check if value matches the expected type"""
            actual_type = get_actual_type(expected)
            
            # Handle None
            if actual_type is type(None):
                return value is None

            # Handle pandas DataFrame
            if actual_type is pd.DataFrame:
                return isinstance(value, pd.DataFrame)
                
            # Handle pandas Series
            if actual_type is pd.Series:
                return isinstance(value, pd.Series)
                
            # Handle numpy arrays
            if actual_type is np.ndarray:
                return isinstance(value, np.ndarray)
                
            # Handle Iterable (but not strings)
            if actual_type is Iterable and not isinstance(value, str):
                return hasattr(value, '__iter__')
                
            # Handle basic types
            try:
                return isinstance(value, actual_type)
            except TypeError:
                return False

        try:
            actual_type = get_actual_type(expected_type)

            # Handle basic validations first
            if is_valid_type(value, expected_type):
                return value

            # Handle type conversions
            if actual_type is pd.DataFrame:
                try:
                    return pd.DataFrame(value)
                except:
                    raise type_error()
                    
            if actual_type is pd.Series:
                try:
                    return pd.Series(value)
                except:
                    raise type_error()
                    
            if actual_type is np.ndarray:
                try:
                    return np.array(value)
                except:
                    raise type_error()

            if actual_type is int:
                try:
                    if isinstance(value, float):
                        if not value.is_integer():
                            raise type_error("Cannot convert decimal value to integer")
                        return int(value)
                    if isinstance(value, (str, np.integer)):
                        float_val = float(str(value).strip())
                        if not float_val.is_integer():
                            raise type_error("Cannot convert decimal value to integer")
                        return int(float_val)
                    return int(value)
                except (ValueError, TypeError):
                    raise type_error()

            if actual_type is float:
                try:
                    if isinstance(value, (int, float, np.number)):
                        return float(value)
                    if isinstance(value, str):
                        return float(value.strip())
                    raise type_error()
                except ValueError:
                    raise type_error()

            if actual_type is bool:
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
                raise type_error()

            if actual_type is str:
                return str(value)

            # Try general conversion as last resort
            try:
                result = actual_type(value)
                if is_valid_type(result, expected_type):
                    return result
                raise type_error()
            except (ValueError, TypeError):
                raise type_error()

        except TypeError as e:
            if "Subscripted generics" in str(e):
                # Handle generic type checking errors
                if is_valid_type(value, expected_type):
                    return value
            raise type_error()

    def append_to_value_list(self, l: NamedValueList) -> Self:
        """
        Appends this value instance to a NamedValueList.
        
        Convenience method for adding this value to a list while enabling
        method chaining.
        
        Args:
            l (NamedValueList): The list to append this value to
            
        Returns:
            Self: This instance for method chaining
            
        Example:
            ```python
            value_list = NamedValueList()
            value = NamedValue("example", 42)
            value.append_to_value_list(value_list).force_set_value(43)
            ```
        """
        l.append(self)
        return self

    def register_to_value_hash(self, h: NamedValueHash) -> Self:
        """
        Registers this value instance in a NamedValueHash.
        
        Registers this value in the provided hash container. If the hash contains
        value overrides, this value's current value may be overridden during
        registration.
        
        Args:
            h (NamedValueHash): The hash to register this value in
            
        Returns:
            Self: This instance for method chaining
            
        Example:
            ```python
            value_hash = NamedValueHash()
            value = NamedValue("example", 42)
            value.register_to_value_hash(value_hash).force_set_value(43)
            ```
        """
        h.register_value(self)
        return self

    def model_dump_json(self, **kwargs) -> str:
        """
        Custom JSON serialization of the named value.
        
        Serializes the named value instance to a JSON string, including
        all state information and stored value.
        
        Args:
            **kwargs: JSON serialization options like indent, ensure_ascii, etc.
            
        Returns:
            str: JSON string representation
            
        Example:
            ```python
            value = NamedValue("example", 42)
            json_str = value.model_dump_json(indent=2)
            print(json_str)  # Pretty-printed JSON
            ```
        """
        # Separate JSON-specific kwargs from model_dump kwargs
        json_kwargs = {k: v for k, v in kwargs.items() if k in {'indent', 'ensure_ascii', 'separators'}}
        dump_kwargs = {k: v for k, v in kwargs.items() if k not in json_kwargs}
        
        # Get model data with stored value
        data = self.model_dump(**dump_kwargs)
        return json.dumps(data, **json_kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs) -> NamedValue:
        """
        Custom JSON deserialization to NamedValue instance.
        
        Reconstructs a NamedValue instance from a JSON string representation,
        restoring all state and stored value information.
        
        Args:
            json_data (str): JSON string to deserialize
            **kwargs: Additional validation options
            
        Returns:
            NamedValue: New instance with restored state
            
        Example:
            ```python
            json_str = '{"name": "example", "state": "set", "stored_value": 42}'
            value = NamedValue.model_validate_json(json_str)
            print(value.value)  # Outputs: 42
            ```
        """
        data = json.loads(json_data)
        return cls.model_validate(data, **kwargs)

class NamedValueHash(NamedObjectHash):
    """
    A type-safe dictionary for storing and managing NamedValue objects.
    
    NamedValueHash provides a dictionary-like interface for managing a collection
    of NamedValue instances, using their names as keys. It ensures type safety
    and provides convenient methods for accessing and managing the stored values.
    
    The hash maintains unique naming across all stored values and supports
    serialization/deserialization of the entire collection.
    
    Attributes:
        _registry_category (str): Category identifier for object registration
        model_config (ConfigDict): Pydantic configuration for model behavior
        
    Example:
        ```python
        value_hash = NamedValueHash()
        value_hash.register_value(NamedValue("count", 42))
        print(value_hash.get_raw_value("count"))  # Outputs: 42
        ```
    """
    _registry_category = "values"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        protected_namespaces=()
    )

    def register_value(self, value: NamedValue) -> Self:
        """
        Register a named value in the hash.
        
        Args:
            value (NamedValue): The value to register
            
        Returns:
            Self: Returns self for method chaining
            
        Raises:
            ValueError: If a value with the same name already exists
            
        Example:
            ```python
            hash = NamedValueHash()
            value = NamedValue("price", 10.99)
            hash.register_value(value).register_value(NamedValue("qty", 5))
            ```
        """
        return self.register_object(value)
    
    def get_value(self, name: str) -> NamedValue:
        """
        Retrieve a named value by its name.
        
        Args:
            name (str): Name of the value to retrieve
            
        Returns:
            NamedValue: The requested named value
            
        Raises:
            KeyError: If no value exists with the given name
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("price", 10.99))
            price = hash.get_value("price")
            print(price.value)  # Outputs: 10.99
            ```
        """
        return self.get_object(name)
    
    def get_values(self) -> Iterable[NamedValue]:
        """
        Get all registered named values.
        
        Returns:
            Iterable[NamedValue]: An iterator over all stored named values
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            hash.register_value(NamedValue("y", 2))
            for value in hash.get_values():
                print(f"{value.name}: {value.value}")
            ```
        """
        return self.get_objects()
    
    def get_value_names(self) -> Iterable[str]:
        """
        Get names of all registered values.
        
        Returns:
            Iterable[str]: An iterator over all value names
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            hash.register_value(NamedValue("y", 2))
            print(list(hash.get_value_names()))  # Outputs: ['x', 'y']
            ```
        """
        return self.get_object_names()
    
    def get_value_by_type(self, value_type: Type) -> Iterable[NamedValue]:
        """
        Get all values of a specific type.
        
        Args:
            value_type (Type): Type to filter values by
            
        Returns:
            Iterable[NamedValue]: Values matching the specified type
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            hash.register_value(NamedValue("name", "test"))
            integers = list(hash.get_value_by_type(int))
            ```
        """
        return [val for val in self.get_values() if isinstance(val, value_type)]
    
    def get_raw_values(self) -> Iterable[Any]:
        """
        Get the underlying values of all named values.
        
        Returns:
            Iterable[Any]: Iterator over the actual values stored in each NamedValue
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            hash.register_value(NamedValue("y", 2))
            print(list(hash.get_raw_values()))  # Outputs: [1, 2]
            ```
        """
        return (val.value for val in self.get_values())
    
    def get_raw_value(self, name: str) -> Any:
        """
        Get the underlying value by name.
        
        Args:
            name (str): Name of the value to retrieve
            
        Returns:
            Any: The actual value stored in the named value
            
        Raises:
            KeyError: If no value exists with the given name
            ValueError: If the value hasn't been set yet
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("price", 10.99))
            print(hash.get_raw_value("price"))  # Outputs: 10.99
            ```
        """
        return self.get_value(name).value
    
    def set_raw_value(self, name: str, value: Any) -> None:
        """
        Set the underlying value for a named value.
        
        Args:
            name (str): Name of the value to update
            value (Any): New value to set
            
        Raises:
            KeyError: If no value exists with the given name
            TypeError: If value type doesn't match the expected type
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("price", 10.99))
            hash.set_raw_value("price", 11.99)
            ```
        """
        self.get_value(name).value = value

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Custom serialization to preserve stored values and their states.
        
        Creates a dictionary representation of the hash that includes full
        serialization of all contained NamedValue objects, preserving their
        values and states.
        
        Args:
            **kwargs: Additional serialization options passed to all nested objects
            
        Returns:
            dict[str, Any]: Dictionary containing the complete hash state
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            data = hash.model_dump()
            print(data['objects']['x']['stored_value'])  # Outputs: 1
            ```
        """
        data = super().model_dump(**kwargs)
        # Ensure each object's stored value is included
        if 'objects' in data:
            for name, obj in self.objects.items():
                if isinstance(obj, NamedValue):
                    # Get the full dump including stored value
                    obj_data = obj.model_dump(**kwargs)
                    data['objects'][name] = obj_data
        return data

    @classmethod
    def model_validate(cls, data: Any) -> NamedValueHash:
        """
        Custom validation to restore hash state from serialized data.
        
        Reconstructs a NamedValueHash instance from serialized data, including
        all contained NamedValue objects with their values and states.
        
        Args:
            data (Any): Serialized data to deserialize. Should be a dictionary
                containing an 'objects' key with serialized NamedValue instances
            
        Returns:
            NamedValueHash: New instance with all values restored
            
        Example:
            ```python
            data = {
                'objects': {
                    'x': {'name': 'x', 'type': 'NamedValue', 'stored_value': 1}
                }
            }
            hash = NamedValueHash.model_validate(data)
            print(hash.get_raw_value('x'))  # Outputs: 1
            ```
        """
        if not isinstance(data, dict):
            return super().model_validate(data)
        
        instance = cls()
        
        # Process each object in the data
        for name, obj_data in data.get('objects', {}).items():
            if isinstance(obj_data, dict):
                obj_type = obj_data.get('type')
                if obj_type:
                    # Get the appropriate class from registry
                    value_class = ObjectRegistry.get(cls._registry_category, obj_type)
                    # Create and validate the object with its stored value
                    value_obj = value_class.model_validate(obj_data)
                    instance.register_value(value_obj)
        
        return instance

    def model_dump_json(self, **kwargs) -> str:
        """
        Custom JSON serialization of the entire hash.
        
        Serializes the NamedValueHash instance and all contained NamedValue
        objects to a JSON string representation. Handles both the hash structure
        and the nested value serialization.
        
        Args:
            **kwargs: JSON serialization options such as:
                - indent: Number of spaces for pretty printing
                - ensure_ascii: Escape non-ASCII characters
                - separators: Tuple of (item_sep, key_sep) for custom formatting
            
        Returns:
            str: JSON string representation of the hash
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            json_str = hash.model_dump_json(indent=2)
            print(json_str)  # Pretty-printed JSON with nested values
            ```
        """
        # Separate JSON-specific kwargs from model_dump kwargs
        json_kwargs = {k: v for k, v in kwargs.items() if k in {'indent', 'ensure_ascii', 'separators'}}
        dump_kwargs = {k: v for k, v in kwargs.items() if k not in json_kwargs}
        
        # Get model data
        data = self.model_dump(**dump_kwargs)
        # Serialize to JSON
        return json.dumps(data, **json_kwargs)
    
    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs) -> NamedValueHash:
        """
        Custom JSON deserialization to NamedValueHash instance.
        
        Reconstructs a NamedValueHash instance from a JSON string representation,
        including all contained NamedValue objects with their complete state.
        
        Args:
            json_data (str): JSON string containing serialized hash data
            **kwargs: Additional validation options for nested objects
            
        Returns:
            NamedValueHash: New instance with all values restored
            
        Example:
            ```python
            json_str = '''
            {
                "objects": {
                    "x": {"name": "x", "type": "NamedValue", "stored_value": 1}
                }
            }
            '''
            hash = NamedValueHash.model_validate_json(json_str)
            print(hash.get_raw_value('x'))  # Outputs: 1
            ```
        """
        data = json.loads(json_data)
        return cls.model_validate(data, **kwargs)

class NamedValueList(NamedObjectList):
    """
    An ordered list container for managing NamedValue objects.
    
    NamedValueList maintains an ordered collection of NamedValue objects while
    providing type safety and convenient access methods. It preserves insertion
    order while also allowing access by name.
    
    Attributes:
        _registry_category (str): Category identifier for object registration
        objects (List[SerializeAsAny[InstanceOf[NamedValue]]]): The list of stored values
        
    Example:
        ```python
        value_list = NamedValueList()
        value_list.append(NamedValue("first", 1))
        value_list.append(NamedValue("second", 2))
        print([v.value for v in value_list])  # Outputs: [1, 2]
        ```
    """
    _registry_category = "values"
    
    # Attributes
    objects: List[SerializeAsAny[InstanceOf[NamedValue]]] = Field(default_factory=list)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        protected_namespaces=()
    )

    def append(self, value: NamedValue) -> Self:
        """
        Append a named value to the end of the list.
        
        Args:
            value (NamedValue): Named value to append
            
        Returns:
            Self: The list instance for method chaining
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue("x", 1)).append(NamedValue("y", 2))
            ```
        """
        return super().append(value)

    def extend(self, values: Iterable[NamedValue]) -> Self:
        """
        Extend the list with multiple named values.
        
        Args:
            values (Iterable[NamedValue]): Collection of named values to add
            
        Returns:
            Self: The list instance for method chaining
            
        Example:
            ```python
            value_list = NamedValueList()
            new_values = [NamedValue("x", 1), NamedValue("y", 2)]
            value_list.extend(new_values)
            ```
        """
        return super().extend(values)

    def __getitem__(self, idx: int) -> NamedValue:
        """Get a named value by its index in the list.

        Args:
            idx (int): Index of the named value to retrieve

        Returns:
            NamedValue: The named value at the specified index

        Raises:
            IndexError: If the index is out of range

        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue(name="price", value=10.5))
            first_value = value_list[0] # Get first named value
            ```
        """
        return super().__getitem__(idx)

    def register_value(self, value: NamedValue) -> Self:
        """
        Register a named value to the list.
        
        Similar to append but uses the register_object method internally,
        which may perform additional validation.
        
        Args:
            value (NamedValue): Named value to register
            
        Returns:
            Self: The list instance for method chaining
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.register_value(NamedValue("x", 1))
            ```
        """
        return self.register_object(value)

    def get_value(self, name: str) -> NamedValue:
        """
        Get a registered named value by name.
        
        Args:
            name (str): Name of the value to retrieve
            
        Returns:
            NamedValue: The requested named value
            
        Raises:
            KeyError: If no value exists with the given name
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue("x", 1))
            x = value_list.get_value("x")
            print(x.value)  # Outputs: 1
            ```
        """
        return self.get_object(name)

    def get_values(self) -> Iterable[NamedValue]:
        """
        Get all registered named values.
        
        Returns:
            Iterable[NamedValue]: Iterator over all stored named values
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.extend([NamedValue("x", 1), NamedValue("y", 2)])
            for value in value_list.get_values():
                print(f"{value.name}: {value.value}")
            ```
        """
        return self.get_objects()

    def get_value_by_type(self, value_type: Type) -> Iterable[NamedValue]:
        """
        Get all values whose stored value is of a specific type.
        
        Args:
            value_type (Type): Type to filter values by
            
        Returns:
            Iterable[NamedValue]: Values whose stored value matches the specified type
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue("x", 1))
            value_list.append(NamedValue("name", "test"))
            integers = list(value_list.get_value_by_type(int))
            ```
        """
        for value in self.get_values():
            try:
                if isinstance(value.value, value_type):
                    yield value
            except ValueError:
                # Skip unset values
                continue
    
    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Custom serialization to preserve stored values.
        
        Extends the parent class serialization to ensure proper serialization
        of all stored named values and their states.
        
        Args:
            **kwargs: Additional serialization options
            
        Returns:
            dict[str, Any]: Dictionary containing serialized state
        """
        data = super().model_dump(**kwargs)
        if 'objects' in data:
            # Ensure each object's stored value is included
            data['objects'] = [
                obj.model_dump(**kwargs) if isinstance(obj, NamedValue) else obj
                for obj in self.objects
            ]
        return data

    @classmethod
    def model_validate(cls, data: Any) -> NamedValueList:
        """
        Custom validation to restore stored values.
        
        Reconstructs a NamedValueList instance from serialized data,
        properly restoring all contained named values and their states.
        
        Args:
            data (Any): Serialized data to deserialize
            
        Returns:
            NamedValueList: New instance with restored values
        """
        if not isinstance(data, dict):
            return super().model_validate(data)
        
        instance = cls()
        
        # Process each object in the data
        for obj_data in data.get('objects', []):
            if isinstance(obj_data, dict):
                obj_type = obj_data.get('type')
                if obj_type:
                    # Get the appropriate class from registry
                    value_class = ObjectRegistry.get(cls._registry_category, obj_type)
                    # Create and validate the object
                    value_obj = value_class.model_validate(obj_data)
                    instance.append(value_obj)
        
        return instance

    def model_dump_json(self, **kwargs) -> str:
        """
        Custom JSON serialization of the value list.
        
        Serializes the NamedValueList instance and all contained NamedValue
        objects to a JSON string representation. Preserves the order of values
        and their complete state.
        
        Args:
            **kwargs: JSON serialization options such as:
                - indent: Number of spaces for pretty printing
                - ensure_ascii: Escape non-ASCII characters
                - separators: Tuple of (item_sep, key_sep) for custom formatting
            
        Returns:
            str: JSON string representation of the list
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue("x", 1))
            value_list.append(NamedValue("y", 2))
            json_str = value_list.model_dump_json(indent=2)
            print(json_str)  # Pretty-printed JSON with ordered values
            ```
        """
        # Separate JSON-specific kwargs from model_dump kwargs
        json_kwargs = {k: v for k, v in kwargs.items() if k in {'indent', 'ensure_ascii', 'separators'}}
        dump_kwargs = {k: v for k, v in kwargs.items() if k not in json_kwargs}
        
        # Get model data
        data = self.model_dump(**dump_kwargs)
        # Serialize to JSON
        return json.dumps(data, **json_kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs) -> NamedValueList:
        """
        Custom JSON deserialization to NamedValueList instance.
        
        Reconstructs a NamedValueList instance from a JSON string representation,
        preserving the order of values and restoring their complete state.
        
        Args:
            json_data (str): JSON string containing serialized list data
            **kwargs: Additional validation options for nested objects
            
        Returns:
            NamedValueList: New instance with all values restored in order
            
        Example:
            ```python
            json_str = '''
            {
                "objects": [
                    {"name": "x", "type": "NamedValue", "stored_value": 1},
                    {"name": "y", "type": "NamedValue", "stored_value": 2}
                ]
            }
            '''
            value_list = NamedValueList.model_validate_json(json_str)
            print([v.value for v in value_list])  # Outputs: [1, 2]
            ```
        """
        data = json.loads(json_data)
        return cls.model_validate(data, **kwargs)
