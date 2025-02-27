# Data Handling Quickstart Guide

This guide demonstrates how to use the [`data_handlers`](../../../process_manager/reference/process_manager/data_handlers/) package for handling random variables and named values.

## Basic Usage of Named Values

```python
from process_manager import data_handlers as dh

# Create named values with automatic state management
name = dh.NamedValue(name="name", value="John Doe")  # Value is set immediately
age = dh.NamedValue(name="age")  # Value is initially unset

# Check and set values
print(name.value)  # Output: "John Doe"
try:
    print(age.value)  # Raises ValueError: Value 'age' has not been set yet
except ValueError as e:
    print(e)

# Set the value for age
age.value = 25  # Sets and freezes the value
try:
    age.value = 26  # Raises ValueError - value is frozen
except ValueError as e:
    print(e)  # Output: Value 'age' has already been set and is frozen...

# Use force_set_value to change a frozen value
age.force_set_value(26)  # Successfully changes the value
print(age.value)  # Output: 26
```

## Managing Named Value Collections

```python
# Create a hash to store named values
nv_hash = dh.NamedValueHash()

# Register named values
nv_hash.register_value(name)
nv_hash.register_value(age)

# Access values through the hash
print(nv_hash.get_raw_value("name"))  # Output: "John Doe"
print(nv_hash.get_raw_value("age"))   # Output: 26

# Create ordered lists of named values
nv_list = dh.NamedValueList()
nv_list.append(name)
nv_list.append(age)

# Access values by index or name
print(nv_list[0].value)  # Output: "John Doe"
print(nv_list.get_value("age").value)  # Output: 26
```

## Working with Random Variables

```python
import numpy as np

# Create random variable distributions
normal_dist = dh.NormalDistribution(name="height", mu=170, sigma=10)
uniform_dist = dh.UniformDistribution(name="weight", low=60, high=90)
categories = np.array(["A", "B", "C"])
cat_dist = dh.CategoricalDistribution(
    name="blood_type",
    categories=categories,
    probabilities=np.array([0.4, 0.1, 0.5])
)

# Create a hash to store random variables
rv_hash = dh.RandomVariableHash()

# Register variables and get samples
height = normal_dist.register_to_hash(rv_hash, size=5)
weight = uniform_dist.register_to_hash(rv_hash, size=5)
blood_type = cat_dist.register_to_hash(rv_hash, size=5)

# View the samples
print(f"Heights: {height.value}")      # e.g., [168.3, 175.2, 162.1, 171.8, 169.5]
print(f"Weights: {weight.value}")      # e.g., [75.3, 82.1, 68.4, 71.2, 88.9]
print(f"Blood Types: {blood_type.value}")  # e.g., ['A', 'C', 'A', 'C', 'C']
```

## Serialization and Deserialization

```python
# Serialize named values hash
nv_json = nv_hash.model_dump_json(indent=2)
print(nv_json)
```

Example output:
```json
{
  "objects": {
    "name": {
      "name": "name",
      "type": "NamedValue",
      "state": "set",
      "stored_value": "John Doe"
    },
    "age": {
      "name": "age",
      "type": "NamedValue",
      "state": "set",
      "stored_value": 26
    }
  }
}
```

```python
# Serialize random variables hash
rv_json = rv_hash.model_dump_json(indent=2)
print(rv_json)
```

Example output:
```json
{
  "objects": {
    "height": {
      "name": "height",
      "type": "NormalDistribution",
      "mu": 170,
      "sigma": 10,
      "seed": null
    },
    "weight": {
      "name": "weight",
      "type": "UniformDistribution",
      "low": 60,
      "high": 90,
      "seed": null
    },
    "blood_type": {
      "name": "blood_type",
      "type": "CategoricalDistribution",
      "categories": ["A", "B", "C"],
      "probabilities": [0.4, 0.1, 0.5],
      "seed": null
    }
  }
}
```

```python
# Load from serialized data
new_nv_hash = dh.NamedValueHash.model_validate_json(nv_json)
new_rv_hash = dh.RandomVariableHash.model_validate_json(rv_json)

# Save to and load from files
with open("nv_hash.json", "w") as f:
    f.write(nv_hash.model_dump_json(indent=2))

with open("nv_hash.json", "r") as f:
    loaded_nv_hash = dh.NamedValueHash.model_validate_json(f.read())
```

## Advanced Named Value Features

```python
# Type-safe value handling
integer_value = dh.NamedValue[int](name="count")  # Explicitly typed as int
integer_value.value = 42

try:
    integer_value.value = "not an integer"  # Raises TypeError
except TypeError as e:
    print(e)

# Working with collections
values_list = dh.NamedValueList()
integer_value.append_to_value_list(values_list)  # Method chaining
values_list.append(dh.NamedValue("price", 10.99))

# Iterate over values
for value in values_list.get_values():
    print(f"{value.name}: {value.value}")

# Filter values by type
number_values = values_list.get_value_by_type(int)
```

## Best Practices

1. Always use type hints with NamedValue for better type safety:
```python
temperature = dh.NamedValue[float](name="temp")
name = dh.NamedValue[str](name="user_name")
```

2. Handle unset values appropriately:
```python
value = dh.NamedValue(name="example")
if value._state == dh.NamedValueState.UNSET:
    # Handle unset value case
    value.value = default_value
```

3. Use force_set_value() sparingly and only when you need to override frozen values:
```python
# Prefer setting values once
config = dh.NamedValue(name="config", value=initial_config)

# Only use force_set_value when absolutely necessary
if needs_update:
    config.force_set_value(new_config)
```

4. Leverage the built-in serialization for persistence:
```python
# Save state
saved_state = value.model_dump_json()

# Restore state
restored_value = dh.NamedValue.model_validate_json(saved_state)
```