"""
Module for generating, sorting, and managing named values.
This uses pydantic dataclasses for JSON serialization to avoid overloading system memory.
"""
from __future__ import annotations

from dataclasses import Field
from enum import Enum
from typing import Any, Iterable, List, Type, TypeVar, Union, Generic, ClassVar
from numpydantic import NDArray
from process_manager.data_handlers.custom_serde_definitions.pandantic import PandasDataFrame, PandasSeries
try:
    from typing import Self
except:
    from typing_extensions import Self

from pydantic import (
    ConfigDict, 
    InstanceOf, 
    SerializeAsAny, 
    Field, 
    PrivateAttr, 
    model_validator
)
import json

from process_manager.data_handlers.base import (
    NamedObject, 
    NamedObjectHash, 
    NamedObjectList,
    ObjectRegistry
)
from process_manager.data_handlers.mixins import ArrayDunders

__all__ = [
    'SerializableValue',
    'NamedValue',
    'NamedValueList',
    'NamedValueHash',
    'UNSET'
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

class UNSET(Enum):
    """Sentinel value to indicate an unset value state."""
    token = object()

    def __str__(self) -> str:
        return "<UNSET>"
    
    def __repr__(self) -> str:
        return "<UNSET>"

@ArrayDunders.mixin
class NamedValue(NamedObject, Generic[T]):
    # Define as a ClassVar to ensure it's treated as a class-level attribute
    _registry_category: ClassVar[str] = "values"
    
    name: str = Field(..., description="Name of the value")
    _stored_value: T | UNSET = PrivateAttr(default=UNSET.token)
    _type: type = PrivateAttr()

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra='allow',
        protected_namespaces=()  # Allow fields starting with underscore
    )

    def __init__(self, name: str, value: T | None = None, **data):
        """
        Initialize a new NamedValue instance.

        Args:
            name (str): The name identifier for this value
            value (T, optional): Initial value to set. If provided,
                this value becomes frozen after initialization. Defaults to None.
            **data: Additional keyword arguments passed to parent class
        """
        # Remove stored_value from data if present to prevent backdoor setting
        data.pop('stored_value', None)
        data.pop('_stored_value', None)
        
        super().__init__(name=name, **data)
        self._type = self._extract_value_type()
        
        if value is not None:
            self.value = value

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent direct modification of _stored_value"""
        if name == '_stored_value':
            raise AttributeError("Cannot modify _stored_value directly. Use force_set_value() instead.")
        super().__setattr__(name, value)

    @property
    def value(self) -> T:
        """
        Get the stored value.

        Returns:
            T: The currently stored value

        Raises:
            ValueError: If attempting to access the value before it has been set
        """
        if self._stored_value is UNSET.token:
            raise ValueError(f"Value '{self.name}' has not been set yet.")
        return self._stored_value

    @value.setter
    def value(self, new_value: T):
        """Set the value if it hasn't been set before."""
        if self._stored_value is not UNSET.token:
            raise ValueError(
                f"Value '{self.name}' has already been set and is frozen. "
                "Use force_set_value() if you need to override it."
            )
            
        validated_value = self._validate_type(new_value)
        object.__setattr__(self, '_stored_value', validated_value)

    def force_set_value(self, new_value: T) -> None:
        """Force set the value regardless of whether it was previously set."""
        object.__setattr__(self, '_stored_value', UNSET.token)
        self.value = new_value

    def _validate_type(self, value: Any) -> T:
        """
        Validate that the value matches the expected type and cast if necessary.
        """
        # Skip validation for Any type
        if self._type is Any:
            return value

        actual_type = getattr(self._type, "__origin__", self._type)
        
        # Determine if this is a custom subclass with its own __init__
        custom_init = self.__class__.__init__ is not NamedValue.__init__
        error_type = ValueError if custom_init else TypeError
            
        # If already correct type, return as is
        if isinstance(value, actual_type):
            return value
            
        try:
            # Handle numeric types explicitly
            if actual_type is int:
                if isinstance(value, str):
                    try:
                        # First try direct integer conversion
                        return int(value.strip())
                    except ValueError:
                        # If that fails, try float conversion
                        try:
                            float_val = float(value.strip())
                            # Check if float is actually an integer
                            if float_val.is_integer():
                                return int(float_val)
                        except ValueError:
                            raise error_type(
                                f"Value '{value}' cannot be converted to integer" if custom_init else
                                f"Value for '{self.name}' must be of type {actual_type.__name__}, "
                                f"got {type(value).__name__} with value {value!r}"
                            )
                return int(value)
                
            elif actual_type is float:
                if isinstance(value, str):
                    try:
                        return float(value.strip())
                    except ValueError:
                        raise error_type(
                            f"Value '{value}' cannot be converted to float" if custom_init else
                            f"Value for '{self.name}' must be of type {actual_type.__name__}, "
                            f"got {type(value).__name__} with value {value!r}"
                        )
                return float(value)
                
            # For all other types, try direct conversion
            try:
                converted = actual_type(value)
            except (ValueError, TypeError):
                raise error_type(
                    f"Value '{value}' cannot be converted to {actual_type.__name__}" if custom_init else
                    f"Value for '{self.name}' must be of type {actual_type.__name__}, "
                    f"got {type(value).__name__} with value {value!r}"
                )
            
            # Verify the conversion worked
            if not isinstance(converted, actual_type):
                raise error_type(
                    f"Conversion failed to produce valid {actual_type.__name__}" if custom_init else
                    f"Value for '{self.name}' must be of type {actual_type.__name__}, "
                    f"got {type(converted).__name__}"
                )
                
            return converted
            
        except (ValueError, TypeError) as e:
            # Only re-raise if it's already the correct error type
            if isinstance(e, error_type):
                raise e
            raise error_type(
                f"Value '{value}' cannot be converted to {actual_type.__name__}" if custom_init else
                f"Value for '{self.name}' must be of type {actual_type.__name__}, "
                f"got {type(value).__name__} with value {value!r}"
            )
    
    def _extract_value_type(self) -> type:
        """
        Extract the type parameter T from the class generic.
        
        Returns:
            type: The type parameter, or Any if not specified
        """
        cls = self.__class__
        
        # Type mapping for common types
        TYPE_MAP = {
            'int': int,
            'float': float,
            'bool': bool,
            'str': str,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            # Add more types as needed
        }
        
        # First try to get from the class bases
        bases = cls.__bases__
        for base in bases:
            base_str = str(base)
            if 'NamedValue[' in base_str:
                # Extract the type from NamedValue[type]
                type_str = base_str.split('NamedValue[')[1].split(']')[0]
                if type_str in TYPE_MAP:
                    return TYPE_MAP[type_str]
                
                # Handle more complex types if needed
                # For example: List[int], Dict[str, int], etc.
        
        # Fallback to checking generic type parameters
        if hasattr(cls, '__orig_bases__'):
            for base in cls.__orig_bases__:
                if (hasattr(base, '__origin__') and 
                    base.__origin__ is NamedValue and 
                    len(base.__args__) > 0):
                    return base.__args__[0]
        
        return Any
    
    def append_to_value_list(self, l: NamedValueList) -> Self:
        """
        Appends self to given list.

        Args:
            l (NamedValueList): list to which self will be appended
        
        Returns:
            Self: returns instance of self for method chaining
        """
        l.append(self)
        return self

    def register_to_value_hash(self, h: NamedValueHash) -> Self:
        """
        Registers self to the NamedValueHash object.
        
        The hash may override this value's current value if there are
        value overrides defined in the hash. This is done using force_set_value()
        internally.
        
        Args:
            h (NamedValueHash): Hash to which self will be registered
            
        Returns:
            Self: returns instance of self for method chaining
        """
        h.register_value(self)
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization to include stored value"""
        data = super().model_dump(**kwargs)
        # Include stored_value, using "<UNSET>" string for UNSET token
        data['stored_value'] = "<UNSET>" if self._stored_value is UNSET.token else self._stored_value
        return data

    @classmethod
    def model_validate(cls, data: Any) -> 'NamedValue':
        """Custom validation to restore stored value"""
        if not isinstance(data, dict):
            return super().model_validate(data)

        # Make a copy to avoid modifying the input
        data_copy = data.copy()
        
        # Extract stored value
        stored_value = data_copy.pop('stored_value', "<UNSET>")
        
        # Create instance
        instance = super().model_validate(data_copy)
        
        # Set the value only if it's not "<UNSET>"
        if stored_value != "<UNSET>":
            instance.force_set_value(stored_value)
        
        return instance
    
    def model_dump_json(self, **kwargs) -> str:
        """Custom JSON serialization"""
        # Separate JSON-specific kwargs from model_dump kwargs
        json_kwargs = {k: v for k, v in kwargs.items() if k in {'indent', 'ensure_ascii', 'separators'}}
        dump_kwargs = {k: v for k, v in kwargs.items() if k not in json_kwargs}
        
        # Get model data with stored value
        data = self.model_dump(**dump_kwargs)
        return json.dumps(data, **json_kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs) -> 'NamedValue':
        """Custom JSON deserialization"""
        data = json.loads(json_data)
        return cls.model_validate(data, **kwargs)

class NamedValueHash(NamedObjectHash):
    """
    Dictionary of named value instances.
    
    A type-safe dictionary for storing and managing NamedValue objects,
    using the name as the key of the value instance.
    """
    _registry_category = "values"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        protected_namespaces=()
    )

    def register_value(self, value: NamedValue) -> Self:
        """
        Register a named value. Checks for naming conflicts.
        
        Args:
            value (NamedValue): The value to register
            
        Returns:
            Self: Returns self for chaining
            
        Raises:
            ValueError: If a value with the same name exists
        """
        return self.register_object(value)
    
    def get_value(self, name: str) -> NamedValue:
        """Get value by name."""
        return self.get_object(name)
    
    def get_values(self) -> Iterable[NamedValue]:
        """Get all values."""
        return self.get_objects()
    
    def get_value_names(self) -> Iterable[str]:
        """Get names of all values."""
        return self.get_object_names()
    
    def get_value_by_type(self, value_type: Type) -> Iterable[NamedValue]:
        """
        Get all values of a specific type.
        
        Args:
            value_type (Type): Type to filter values by
            
        Returns:
            Iterable[NamedValue]: Values matching the specified type
        """
        return [val for val in self.get_values() if isinstance(val, value_type)]
    
    def get_raw_values(self) -> Iterable[Any]:
        """
        Get the underlying values of all named values.
        
        Returns:
            Iterable[Any]: The actual values stored in each NamedValue
        """
        return (val.value for val in self.get_values())
    
    def get_raw_value(self, name: str) -> Any:
        """
        Get the underlying value by name.
        
        Args:
            name (str): Name of the value to retrieve
            
        Returns:
            Any: The actual value stored in the named value
        """
        return self.get_value(name).value
    
    def set_raw_value(self, name: str, value: Any) -> None:
        """
        Set the underlying value.
        
        Args:
            name (str): Name of the value to update
            value (Any): New value to set
        """
        self.get_value(name).value = value

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization to preserve stored values"""
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
    def model_validate(cls, data: Any) -> 'NamedValueHash':
        """Custom validation to restore stored values"""
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
        """Custom JSON serialization"""
        # Separate JSON-specific kwargs from model_dump kwargs
        json_kwargs = {k: v for k, v in kwargs.items() if k in {'indent', 'ensure_ascii', 'separators'}}
        dump_kwargs = {k: v for k, v in kwargs.items() if k not in json_kwargs}
        
        # Get model data
        data = self.model_dump(**dump_kwargs)
        # Serialize to JSON
        return json.dumps(data, **json_kwargs)
    
    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs) -> 'NamedValueHash':
        """Custom JSON deserialization"""
        data = json.loads(json_data)
        return cls.model_validate(data, **kwargs)

class NamedValueList(NamedObjectList):
    """List of named value instances.
    This class manages an ordered list of NamedValue objects, providing methods
    to add, access, and manage multiple named values while maintaining their order.

    Attributes:
        _registry_category (str): Category name for object registration
        objects (List[SerializeAsAny[InstanceOf[NamedValue]]]): List of named value objects
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
        """Append a named value to the end of the list.

        Args:
            value (NamedValue): Named value to append

        Returns:
            Self: The NamedValueList instance for method chaining

        Example:
            >>> value_list = NamedValueList()
            >>> named_value = NamedValue(name="price", value=10.5)
            >>> value_list.append(named_value)
        """
        return super().append(value)

    def extend(self, values: Iterable[NamedValue]) -> Self:
        """Extend the list with multiple named values.

        Args:
            values (Iterable[NamedValue]): Collection of named values to add

        Returns:
            Self: The NamedValueList instance for method chaining

        Example:
            >>> value_list = NamedValueList()
            >>> new_values = [
            ...     NamedValue(name="price", value=10.5),
            ...     NamedValue(name="quantity", value=5)
            ... ]
            >>> value_list.extend(new_values)
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
            >>> value_list = NamedValueList()
            >>> value_list.append(NamedValue(name="price", value=10.5))
            >>> first_value = value_list[0] # Get first named value
        """
        return super().__getitem__(idx)

    def register_value(self, value: NamedValue) -> Self:
        """Register a named value to the collection.

        Args:
            value (NamedValue): Named value to register

        Returns:
            Self: The NamedValueList instance
        """
        return self.register_object(value)

    def get_value(self, name: str) -> NamedValue:
        """Get a registered named value by name.

        Args:
            name (str): Name of the named value

        Returns:
            NamedValue: The requested named value

        Raises:
            KeyError: If no value exists with the given name
        """
        return self.get_object(name)

    def get_values(self) -> Iterable[NamedValue]:
        """Get all registered named values.

        Returns:
            Iterable[NamedValue]: Iterator over all registered values
        """
        return self.get_objects()

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization to preserve stored values"""
        data = super().model_dump(**kwargs)
        if 'objects' in data:
            # Ensure each object's stored value is included
            data['objects'] = [
                obj.model_dump(**kwargs) if isinstance(obj, NamedValue) else obj
                for obj in self.objects
            ]
        return data

    @classmethod
    def model_validate(cls, data: Any) -> 'NamedValueList':
        """Custom validation to restore stored values"""
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
        """Custom JSON serialization"""
        # Separate JSON-specific kwargs from model_dump kwargs
        json_kwargs = {k: v for k, v in kwargs.items() if k in {'indent', 'ensure_ascii', 'separators'}}
        dump_kwargs = {k: v for k, v in kwargs.items() if k not in json_kwargs}
        
        # Get model data
        data = self.model_dump(**dump_kwargs)
        # Serialize to JSON
        return json.dumps(data, **json_kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs) -> 'NamedValueList':
        """Custom JSON deserialization"""
        data = json.loads(json_data)
        return cls.model_validate(data, **kwargs)

def test_value_serialization():
    print("\n=== Testing Value Serialization ===")
    
    # Create test value
    value = NamedValue(name="test", value=42)
    print("\nOriginal value:")
    print(f"Name: {value.name}")
    print(f"Value: {value.value}")
    print(f"Stored value: {value._stored_value}")
    
    # Serialize
    dump_data = value.model_dump()
    print("\nDump data:")
    print(dump_data)
    
    json_str = value.model_dump_json(indent=2)
    print("\nJSON string:")
    print(json_str)
    
    # Deserialize
    new_value = NamedValue.model_validate_json(json_str)
    print("\nDeserialized value:")
    print(f"Name: {new_value.name}")
    print(f"Stored value: {new_value._stored_value}")
    print(f"Value: {new_value.value}")
    
    # Test list serialization
    print("\n=== Testing List Serialization ===")
    value_list = NamedValueList()
    value_list.append(NamedValue(name="first", value=1))
    value_list.append(NamedValue(name="second", value="test"))
    
    # Serialize list
    list_json = value_list.model_dump_json(indent=2)
    print("\nList JSON:")
    print(list_json)
    
    # Deserialize list
    new_list = NamedValueList.model_validate_json(list_json)
    print("\nDeserialized list values:")
    for idx, value in enumerate(new_list):
        print(f"\nValue {idx}:")
        print(f"Name: {value.name}")
        print(f"Stored value: {value._stored_value}")
        try:
            print(f"Value: {value.value}")
        except Exception as e:
            print(f"Error accessing value: {e}")

if __name__ == "__main__":
    test_value_serialization()