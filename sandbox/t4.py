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
print(f"Age value: {nv_hash.get_value('age')}")
print(f"Name value: {nv_hash.get_value('name')}")

# Debug the internal state
print("\nInternal state:")
for key, value in nv_hash.objects.items():
    print(f"{key}: _stored_value = {getattr(value, '_stored_value', None)}")

# Serialization
nv_json = nv_hash.model_dump_json(indent=2)
print("\nSerialized JSON:")
print(nv_json)

# # We might need to add a model_validator to NamedValue that looks like:
# """
# @model_validator(mode='after')
# def set_stored_value(cls, values):
#     if '_stored_value' in values:
#         values._stored_value = values.value
#     return values
# """

# # Or potentially use a custom serializer:
# """
# class NamedValue:
#     def model_dump(self, **kwargs):
#         data = super().model_dump(**kwargs)
#         data['_stored_value'] = self._stored_value
#         return data
# """

# Loading back from serialized data
new_nv_hash = dh.NamedValueHash.model_validate_json(nv_json)

print("\nAfter deserialization:")
try:
    print(f"Age value: {new_nv_hash.get_value('age')}")
    print(f"Name value: {new_nv_hash.get_value('name')}")
except Exception as e:
    print(f"Error: {e}")

print("\nDeserialized internal state:")
for key, value in new_nv_hash.objects.items():
    print(f"{key}: _stored_value = {getattr(value, '_stored_value', None)}")