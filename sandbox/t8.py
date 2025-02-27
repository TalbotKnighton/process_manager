from process_manager import data_handlers as dh
import json

def test_value_protection_and_serialization():
    print("\n=== Testing Value Protection and Serialization ===")
    
    # Create initial value
    value = dh.NamedValue(name="test", value=42)
    print(f"Initial value: {value.value}")
    
    # Test direct modification protection
    print("\nTesting direct modification protection:")
    try:
        value._stored_value = 100
        print("WARNING: Direct modification succeeded!")
    except AttributeError as e:
        print(f"Protected: {e}")
    
    # Test normal value setter protection
    print("\nTesting value setter protection:")
    try:
        value.value = 100
        print("WARNING: Value modification succeeded!")
    except ValueError as e:
        print(f"Protected: {e}")
    
    # Test serialization
    print("\nTesting serialization:")
    nv_hash = dh.NamedValueHash()
    nv_hash.register_value(value)
    
    # Debug the hash before serialization
    print("\nBefore serialization:")
    print(f"Raw value in hash: {nv_hash.get_value('test').value}")
    print(f"Hash dump: {nv_hash.model_dump()}")
    
    # Serialize
    json_data = nv_hash.model_dump_json(indent=2)
    print("\nSerialized JSON:")
    print(json_data)
    
    # Debug parsed JSON
    print("\nParsed JSON data:")
    parsed = json.loads(json_data)
    print(json.dumps(parsed, indent=2))
    
    # Deserialize
    print("\nTesting deserialization:")
    new_hash = dh.NamedValueHash.model_validate_json(json_data)
    new_value = new_hash.get_value("test")
    
    try:
        print(f"Deserialized value: {new_value.value}")
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Deserialized object state:", vars(new_value))
    
    # Test force_set_value still works
    print("\nTesting force_set_value:")
    try:
        new_value.force_set_value(100)
        print(f"Force set succeeded. New value: {new_value.value}")
    except Exception as e:
        print(f"ERROR: Force set failed: {e}")

if __name__ == "__main__":
    test_value_protection_and_serialization()