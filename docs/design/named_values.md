# NamedValue Design

The `NamedValue` class implements a type-safe container for named values with built-in validation and serialization support. It follows several key design principles to ensure robust and flexible value handling.

## Core Design Principles

### 1. Type Safety
The `NamedValue` class uses Python's generic typing system to enforce type safety at both runtime and static analysis time:

```python
# Type-safe value container
array_value = NamedValue[np.ndarray]("my_array", value=np.array([1, 2, 3]))
int_value = NamedValue[int]("my_int", value=42)
```

Type safety is enforced through:
- Generic type parameters (`NamedValue[T]`)
- Runtime type validation
- Automatic type conversion when possible
- Clear error messages when types don't match

### 2. Value Immutability
Values are immutable by default after initial setting to prevent accidental modifications:

```python
value = NamedValue[int]("counter", 1)
value.value = 2  # Raises ValueError - value is frozen
value.force_set_value(2)  # Explicit override when needed
```

### 3. Flexible Type Conversion
The system attempts to convert values to the correct type when possible:

```python
# String to int conversion
int_value = NamedValue[int]("count", "123")  # Automatically converts to int(123)

# Float to int conversion
int_value = NamedValue[int]("count", 123.0)  # Automatically converts to int(123)
```

### 4. Inheritance Support
Two different patterns are supported for extending `NamedValue`:

1. Direct generic usage:
```python
value = NamedValue[int]("my_int", 42)
```

2. Subclass with custom behavior:
```python
class IntegerValue(NamedValue[int]):
    def __init__(self, name: str, value: int = None):
        super().__init__(name, value)
```

## Error Handling Design

The class implements a sophisticated error handling system that distinguishes between different usage patterns:

1. Direct Usage Errors (TypeError):
```python
# Raises TypeError with detailed type information
value = NamedValue[int]("test", "not an integer")
```

2. Subclass Usage Errors (ValueError):
```python
# Raises ValueError with user-friendly message
class IntegerValue(NamedValue[int]):
    pass
value = IntegerValue("test", "not an integer")
```

## Testing Strategy

The design principles are verified through comprehensive testing:

### 1. Type Safety Tests
```python
def test_type_checking(self):
    # Test with explicit type parameter
    class IntValue(NamedValue[int]):
        pass
    
    # Valid integer assignment
    int_value = IntValue("test", 42)
    assert int_value.value == 42
    
    # Valid string that can be cast to int
    str_int_value = IntValue("test2", "123")
    assert str_int_value.value == 123
    
    # Invalid type that can't be cast
    with pytest.raises(TypeError):
        IntValue("test3", "not an integer")
```

### 2. Inheritance Tests
```python
def test_type_casting_inheritance(self):
    class IntegerValue(NamedValue[int]):
        def __init__(self, name: str, value: int = None):
            super().__init__(name, value)
    
    # Test valid assignment
    int_value = IntegerValue("test", 42)
    assert isinstance(int_value.value, int)
    
    # Test type hints are preserved
    with pytest.raises(ValueError):
        IntegerValue("test", "not an integer")
```

### 3. Value Immutability Tests
```python
def test_value_immutability(self):
    value = NamedValue[int]("test", 42)
    
    # Cannot change value after setting
    with pytest.raises(ValueError):
        value.value = 43
        
    # Can force change value when needed
    value.force_set_value(43)
    assert value.value == 43
```

## Serialization Support

The class implements JSON serialization support through pydantic:

```python
value = NamedValue[int]("counter", 42)
serialized = value.model_dump_json()
deserialized = NamedValue.model_validate_json(serialized)
```

## Usage Guidelines

1. Use direct generic syntax for simple value containers:
```python
value = NamedValue[int]("simple_counter", 0)
```

2. Create subclasses for custom validation or behavior:
```python
class PositiveInteger(NamedValue[int]):
    def _validate_type(self, value: Any) -> int:
        value = super()._validate_type(value)
        if value <= 0:
            raise ValueError("Value must be positive")
        return value
```

3. Use `force_set_value()` only when value mutability is explicitly needed:
```python
value.force_set_value(new_value)  # Use with caution
```

## Best Practices

1. Always specify the type parameter for clarity:
```python
# Good
value = NamedValue[int]("count", 42)

# Avoid
value = NamedValue("count", 42)  # Type defaults to Any
```

2. Use meaningful names that describe the value's purpose:
```python
# Good
count = NamedValue[int]("iteration_count", 0)

# Avoid
x = NamedValue[int]("x", 0)  # Name is not descriptive
```

3. Handle type conversion errors appropriately:
```python
try:
    value = NamedValue[int]("count", user_input)
except (TypeError, ValueError) as e:
    # Handle invalid input
    pass
```

By following these design principles and usage patterns, `NamedValue` provides a robust and type-safe way to manage named values in your application.