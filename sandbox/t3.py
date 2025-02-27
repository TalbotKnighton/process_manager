from process_manager import data_handlers as dh
from pydantic import BaseModel, Field
import numpy as np

class NamedValueModel(BaseModel):
    name: str
    value: object  # This will store the actual value
    type: str = "NamedValue"

# Create a hash to store named values
nv_hash = dh.NamedValueHash()

# Create and register named values
age = dh.NamedValue(name="age", value=25)
name = dh.NamedValue(name="name", value="John Doe")

nv_hash.register_value(age)
nv_hash.register_value(name)

# Create the serialization structure manually to include values
serialization_dict = {
    "objects": {
        k: NamedValueModel(
            name=v.name if hasattr(v, 'name') else k,
            value=v.value if hasattr(v, 'value') else v,
            type="NamedValue"
        ).model_dump()
        for k, v in nv_hash.objects.items()
    }
}

# Convert to JSON
import json
nv_json = json.dumps(serialization_dict, indent=2)
print("\nModified Serialized Named Values Hash:")
print(nv_json)

# Test deserialization
deserialized = json.loads(nv_json)
new_nv_hash = dh.NamedValueHash()
for key, value_data in deserialized["objects"].items():
    value_obj = dh.NamedValue(name=value_data["name"], value=value_data["value"])
    new_nv_hash.register_value(value_obj)

# Verify the deserialized data
print("\nVerifying deserialized data:")
print(f"Age value: {new_nv_hash.get_value('age')}")
print(f"Name value: {new_nv_hash.get_value('name')}")