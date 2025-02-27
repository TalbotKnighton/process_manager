Here's a comprehensive example showing both practical usage and potential extensions:

First, basic usage:
test_named_values.py

```python
from process_manager.data_handlers import (
    NamedValue,
    NamedValueState,
    NamedValueList,
    NamedValueHash
)

def test_basic_usage():
    print("\n=== Basic Usage ===")
    
    # Create a value and check its state
    value = NamedValue(name="temperature")
    print(f"Initial state: {value._state}")
    
    # Set the value and verify state change
    value.value = 72.5
    print(f"After setting: {value._state}")
    print(f"Value: {value.value}")
    
    # Try to modify (should fail)
    try:
        value.value = 73.0
        print("Warning: Value was modified!")
    except ValueError as e:
        print(f"Protected: {e}")
    
    # Force set and verify state
    value.force_set_value(73.0)
    print(f"After force set: {value._state}")
    print(f"New value: {value.value}")

def test_serialization():
    print("\n=== Serialization ===")
    
    # Create values in different states
    unset = NamedValue(name="unset")
    set_value = NamedValue(name="set", value=42)
    
    # Serialize both
    print("\nUnset value serialized:")
    print(unset.model_dump_json(indent=2))
    
    print("\nSet value serialized:")
    print(set_value.model_dump_json(indent=2))
    
    # Deserialize and verify states
    json_str = set_value.model_dump_json()
    restored = NamedValue.model_validate_json(json_str)
    print(f"\nRestored value state: {restored._state}")
    print(f"Restored value: {restored.value}")

def test_collections():
    print("\n=== Collections Usage ===")
    
    # Create a list with mixed states
    value_list = NamedValueList()
    value_list.append(NamedValue(name="unset"))
    value_list.append(NamedValue(name="set", value="hello"))
    
    # Create a hash with mixed states
    value_hash = NamedValueHash()
    value_hash.register_value(NamedValue(name="temp", value=98.6))
    value_hash.register_value(NamedValue(name="pressure"))
    
    # Serialize both
    print("\nList serialized:")
    print(value_list.model_dump_json(indent=2))
    
    print("\nHash serialized:")
    print(value_hash.model_dump_json(indent=2))

if __name__ == "__main__":
    test_basic_usage()
    test_serialization()
    test_collections()
```

Extended version with more complex state management:

```python
from enum import Enum
from typing import Any, Optional
from datetime import datetime

class NamedValueState(str, Enum):
    """Extended state enum for NamedValue objects."""
    UNSET = "unset"
    SET = "set"
    PENDING = "pending"  # For async operations
    INVALID = "invalid"  # For validation failures
    EXPIRED = "expired"  # For time-sensitive values

class NamedValueMetadata:
    """Metadata for tracking value state changes."""
    def __init__(self):
        self.created_at: datetime = datetime.now()
        self.last_modified: Optional[datetime] = None
        self.set_count: int = 0
        self.previous_value: Any = None
        self.validation_error: Optional[str] = None

class NamedValue(NamedObject, Generic[T]):
    """Enhanced NamedValue with extended state management."""
    
    _registry_category: ClassVar[str] = "values"
    
    name: str = Field(..., description="Name of the value")
    _stored_value: T | UNSET = PrivateAttr(default=UNSET.token)
    _state: NamedValueState = PrivateAttr(default=NamedValueState.UNSET)
    _type: type = PrivateAttr()
    _metadata: NamedValueMetadata = PrivateAttr()
    _expiry: Optional[datetime] = PrivateAttr(default=None)

    def __init__(self, name: str, value: T | None = None, expires_in: Optional[float] = None, **data):
        super().__init__(name=name, **data)
        self._type = self._extract_value_type()
        self._metadata = NamedValueMetadata()
        
        if expires_in is not None:
            self._expiry = datetime.now().timestamp() + expires_in
            
        if value is not None:
            self.value = value

    @property
    def value(self) -> T:
        """Get the stored value with additional state checks."""
        if self._state == NamedValueState.UNSET:
            raise ValueError(f"Value '{self.name}' has not been set yet.")
        
        if self._state == NamedValueState.INVALID:
            raise ValueError(f"Value '{self.name}' is invalid: {self._metadata.validation_error}")
            
        if self._state == NamedValueState.PENDING:
            raise ValueError(f"Value '{self.name}' is pending.")
            
        if self._state == NamedValueState.EXPIRED:
            raise ValueError(f"Value '{self.name}' has expired.")
            
        return self._stored_value

    @value.setter
    def value(self, new_value: T):
        """Set value with metadata tracking."""
        if self._state == NamedValueState.SET:
            raise ValueError(
                f"Value '{self.name}' has already been set and is frozen. "
                "Use force_set_value() if you need to override it."
            )
        
        try:
            validated_value = self._validate_type(new_value)
            
            # Update metadata
            self._metadata.last_modified = datetime.now()
            self._metadata.set_count += 1
            self._metadata.previous_value = self._stored_value
            
            # Set the new value
            object.__setattr__(self, '_stored_value', validated_value)
            object.__setattr__(self, '_state', NamedValueState.SET)
            
        except Exception as e:
            # Handle validation failure
            self._metadata.validation_error = str(e)
            object.__setattr__(self, '_state', NamedValueState.INVALID)
            raise

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Enhanced serialization with metadata."""
        data = super().model_dump(**kwargs)
        data['state'] = self._state
        
        if self._state == NamedValueState.SET:
            data['stored_value'] = self._stored_value
            
        # Include metadata
        data['metadata'] = {
            'created_at': self._metadata.created_at.isoformat(),
            'last_modified': self._metadata.last_modified.isoformat() if self._metadata.last_modified else None,
            'set_count': self._metadata.set_count,
        }
        
        if self._expiry:
            data['expires_at'] = datetime.fromtimestamp(self._expiry).isoformat()
            
        return data

    def check_expiry(self) -> bool:
        """Check if the value has expired."""
        if self._expiry and datetime.now().timestamp() > self._expiry:
            object.__setattr__(self, '_state', NamedValueState.EXPIRED)
            return True
        return False

    def mark_pending(self) -> None:
        """Mark value as pending (for async operations)."""
        object.__setattr__(self, '_state', NamedValueState.PENDING)

    @property
    def state(self) -> NamedValueState:
        """Get current state."""
        return self._state

    @property
    def metadata(self) -> dict:
        """Get value metadata."""
        return {
            'created_at': self._metadata.created_at,
            'last_modified': self._metadata.last_modified,
            'set_count': self._metadata.set_count,
            'has_previous_value': self._metadata.previous_value is not None
        }
```
Example usage of the extended version:
test_extended_values.py

```python
from process_manager.data_handlers import NamedValue, NamedValueState
from time import sleep

def test_extended_features():
    print("\n=== Testing Extended Features ===")
    
    # Test expiring value
    print("\nTesting expiring value:")
    temp = NamedValue(name="temperature", value=98.6, expires_in=2)
    print(f"Initial value: {temp.value}")
    print(f"Initial state: {temp.state}")
    
    print("Waiting for expiration...")
    sleep(2.1)
    temp.check_expiry()
    print(f"State after expiry: {temp.state}")
    
    try:
        print(f"Value after expiry: {temp.value}")
    except ValueError as e:
        print(f"Expected error: {e}")

    # Test pending state
    print("\nTesting pending state:")
    async_value = NamedValue(name="async_data")
    async_value.mark_pending()
    print(f"State: {async_value.state}")
    
    try:
        print(f"Value: {async_value.value}")
    except ValueError as e:
        print(f"Expected error: {e}")

    # Test metadata
    print("\nTesting metadata:")
    value = NamedValue(name="tracked", value=42)
    print("Initial metadata:", value.metadata)
    
    value.force_set_value(43)
    print("Metadata after update:", value.metadata)

    # Test serialization with metadata
    print("\nTesting enhanced serialization:")
    print(value.model_dump_json(indent=2))

if __name__ == "__main__":
    test_extended_features()
```
This extended version shows:

Additional states for more complex scenarios
Metadata tracking for value changes
Expiring values
Pending state for async operations
Enhanced error handling
Rich serialization format
