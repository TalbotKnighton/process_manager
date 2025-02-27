from process_manager import data_handlers as dh
import numpy as np

# Create named values with debugging
age = dh.NamedValue(name="age", value=25)
name = dh.NamedValue(name="name", value="John Doe")

# Create a hash to store named values
nv_hash = dh.NamedValueHash()

# Register named values
nv_hash.register_value(age)
nv_hash.register_value(name)

print("\nBefore serialization:")
print(f"Age value: {nv_hash.get_value('age').value}")  # Access .value property
print(f"Name value: {nv_hash.get_value('name').value}")  # Access .value property

# Debug the internal state
print("\nInternal state:")
for key, value in nv_hash.objects.items():
    print(f"{key}: _stored_value = {getattr(value, '_stored_value', None)}")

# Serialization
nv_json = nv_hash.model_dump_json(indent=2)
print("\nSerialized JSON:")
print(nv_json)

# Loading back from serialized data
new_nv_hash = dh.NamedValueHash.model_validate_json(nv_json)

print("\nAfter deserialization:")
try:
    print(f"Age value: {new_nv_hash.get_value('age').value}")  # Access .value property
    print(f"Name value: {new_nv_hash.get_value('name').value}")  # Access .value property
except Exception as e:
    print(f"Error: {e}")

print("\nDeserialized internal state:")
for key, value in new_nv_hash.objects.items():
    print(f"{key}: _stored_value = {getattr(value, '_stored_value', None)}")