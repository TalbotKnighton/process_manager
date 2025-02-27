from process_manager import data_handlers as dh

# Type-safe value handling
integer_value = dh.NamedValue[int](name="count")  # Explicitly typed as int
try:
    integer_value.value = "not an integer"  # Raises TypeError
except TypeError as e:
    print(e)
integer_value.value = 42

# Working with collections
values_list = dh.NamedValueList()
integer_value.append_to_value_list(values_list)  # Method chaining
values_list.append(dh.NamedValue("price", 10.99))

# Iterate over values
for value in values_list.get_values():
    print(f"{value.name}: {value.value}")

# Filter values by type
number_values = values_list.get_value_by_type(int)