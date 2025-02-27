from process_manager import data_handlers as dh
import json

# Create and set up test values
age = dh.NamedValue(name="age", value=25)
name = dh.NamedValue(name="name", value="John Doe")

# Debug original values
print("\nOriginal objects:")
print(f"Age object dump: {age.model_dump()}")
print(f"Name object dump: {name.model_dump()}")

# Create and populate hash
nv_hash = dh.NamedValueHash()
nv_hash.register_value(age)
nv_hash.register_value(name)

print("\nBefore serialization:")
print(f"Age value: {nv_hash.get_value('age').value}")
print(f"Name value: {nv_hash.get_value('name').value}")

# Debug internal state
print("\nInternal state before serialization:")
for key, value in nv_hash.objects.items():
    print(f"{key}: _stored_value = {getattr(value, '_stored_value', None)}")
    print(f"{key} model dump: {value.model_dump()}")

# Serialize to JSON
nv_json = nv_hash.model_dump_json(indent=2)
print("\nSerialized JSON:")
print(nv_json)

# Debug the raw JSON data
print("\nParsed JSON data:")
json_data = json.loads(nv_json)
print(json.dumps(json_data, indent=2))

# Deserialize
new_nv_hash = dh.NamedValueHash.model_validate_json(nv_json)

print("\nAfter deserialization:")
try:
    print(f"Age value: {new_nv_hash.get_value('age').value}")
    print(f"Name value: {new_nv_hash.get_value('name').value}")
except Exception as e:
    print(f"Error during value access: {str(e)}")
    
print("\nDeserialized internal state:")
for key, value in new_nv_hash.objects.items():
    print(f"{key}: _stored_value = {getattr(value, '_stored_value', None)}")
    print(f"{key} model dump: {value.model_dump()}")