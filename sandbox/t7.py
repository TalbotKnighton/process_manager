from process_manager import data_handlers as dh
import json

def debug_value(prefix: str, value) -> None:
    print(f"\n{prefix}:")
    print(f"  Type: {type(value)}")
    print(f"  Dict: {vars(value)}")
    print(f"  Model dump: {value.model_dump()}")
    print(f"  _stored_value: {getattr(value, '_stored_value', '<not found>')}")

# Create test values
age = dh.NamedValue(name="age", value=25)
debug_value("Original age value", age)

# Create hash
nv_hash = dh.NamedValueHash()
nv_hash.register_value(age)

# Debug before serialization
print("\nBefore serialization:")
debug_value("Age from hash", nv_hash.get_value('age'))

# Get raw dump
raw_dump = nv_hash.model_dump()
print("\nRaw model dump:")
print(json.dumps(raw_dump, indent=2))

# Serialize
nv_json = nv_hash.model_dump_json(indent=2)
print("\nJSON string:")
print(nv_json)

# Parse JSON back to dict for inspection
parsed_json = json.loads(nv_json)
print("\nParsed JSON:")
print(json.dumps(parsed_json, indent=2))

# Deserialize
new_nv_hash = dh.NamedValueHash.model_validate_json(nv_json)

# Debug after deserialization
print("\nAfter deserialization:")
try:
    age_value = new_nv_hash.get_value('age')
    debug_value("Deserialized age", age_value)
except Exception as e:
    print(f"Error accessing deserialized value: {e}")