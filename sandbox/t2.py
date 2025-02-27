from process_manager import data_handlers as dh
import numpy as np

# Create named values
age = dh.NamedValue(name="age", value=25)
name = dh.NamedValue(name="name", value="John Doe")

print("\nInitial NamedValue objects:")
print("Age type:", type(age))
print("Age dict:", vars(age))
print("Age value:", getattr(age, 'value', None))
print("Age dir:", dir(age))

# Create a hash to store named values
nv_hash = dh.NamedValueHash()

# Register named values and verify storage
nv_hash.register_value(age)
print("\nAfter registering age:")
print("Hash dict:", vars(nv_hash))
print("Hash objects type:", type(nv_hash.objects))
print("Hash objects:", nv_hash.objects)

# Let's try to directly access the stored value
try:
    stored_age = nv_hash.get_object("age")
    print("\nStored age object:")
    print("Type:", type(stored_age))
    print("Dict:", vars(stored_age))
    print("Value:", getattr(stored_age, 'value', None))
    
    # Try different ways to access the value
    print("\nTrying different access methods:")
    print("Direct _value:", getattr(stored_age, '_value', None))
    print("Through property:", stored_age.value if hasattr(stored_age, 'value') else None)
except Exception as e:
    print(f"Error accessing stored value: {str(e)}")

# Before serialization, let's inspect the full structure
print("\nFull hash structure:")
print("Hash type:", type(nv_hash))
print("Hash dict:", vars(nv_hash))
print("Hash objects:", nv_hash.objects)

# Now try serialization
nv_json = nv_hash.model_dump_json(indent=2)
print("\nSerialized JSON:")
print(nv_json)