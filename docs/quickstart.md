# Quickstart Guide

This guide demonstrates how to use the process_manager package for handling random variables and named values.

## Basic Usage

```python
from process_manager import data_handlers as dh
import numpy as np

# Create some random variables
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

print("Samples:")
print(f"Heights: {height.value}")  # e.g., [168.3, 175.2, 162.1, 171.8, 169.5]
print(f"Weights: {weight.value}")  # e.g., [75.3, 82.1, 68.4, 71.2, 88.9]
print(f"Blood Types: {blood_type.value}")  # e.g., ['A', 'C', 'A', 'C', 'C']

# Create named values
age = dh.NamedValue(name="age", value=25)
name = dh.NamedValue(name="name", value="John Doe")

# Create a hash to store named values
nv_hash = dh.NamedValueHash()

# Register named values
nv_hash.register_value(age)
nv_hash.register_value(name)

# Serialization - using standard Pydantic methods
rv_json = rv_hash.model_dump_json(indent=4)
nv_json = nv_hash.model_dump_json(indent=4)

print("\nSerialized Random Variables Hash:")
print(rv_json)
# Output will look like:
# {
#     "objects": {
#         "height": {
#             "name": "height",
#             "mu": 170,
#             "sigma": 10,
#             "seed": null
#         },
#         "weight": {
#             "name": "weight",
#             "low": 60,
#             "high": 90,
#             "seed": null
#         },
#         "blood_type": {
#             "name": "blood_type",
#             "categories": ["A", "B", "C"],
#             "probabilities": [0.4, 0.1, 0.5],
#             "seed": null
#         }
#     }
# }

print("\nSerialized Named Values Hash:")
print(nv_json)
# Output will look like:
# {
#     "objects": {
#         "age": {
#             "name": "age",
#             "value": 25
#         },
#         "name": {
#             "name": "name",
#             "value": "John Doe"
#         }
#     }
# }

# Loading back from serialized data - using standard Pydantic methods
new_rv_hash = dh.RandomVariableHash.model_validate_json(rv_json)
new_nv_hash = dh.NamedValueHash.model_validate_json(nv_json)

# File operations using standard Pydantic methods
# Save to file
with open("rv_hash.json", "w") as f:
    f.write(rv_hash.model_dump_json(indent=4))

# Load from file
with open("rv_hash.json", "r") as f:
    loaded_rv_hash = dh.RandomVariableHash.model_validate_json(f.read())
```

[Rest of the guide remains the same...]