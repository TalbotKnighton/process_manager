import pytest
from process_manager.data_handlers import NamedValue, UNSET
from pydantic import BaseModel
from typing import List

class TestNamedValues:
    def test_named_value_basic(self):
        # Test basic initialization and value access
        value = NamedValue("test", 42)
        assert value.name == "test"
        assert value.value == 42

        # Test initialization without value
        empty_value = NamedValue("empty")
        with pytest.raises(ValueError, match="Value 'empty' has not been set yet"):
            _ = empty_value.value

    def test_value_immutability(self):
        # Test that values are frozen after initial set
        value = NamedValue("test")
        value.value = 42
        
        with pytest.raises(ValueError, match=".*already been set and is frozen.*"):
            value.value = 100
        
        # Test force_set_value
        value.force_set_value(100)
        assert value.value == 100

    def test_math_operations(self):
        # Test various mathematical operations
        v1 = NamedValue("num1", 10)
        v2 = NamedValue("num2", 5)
        
        # Test addition
        result = v1.value + v2.value
        assert result == 15
        
        # Test multiplication
        result = v1.value * v2.value
        assert result == 50
        
        # Test division
        result = v1.value / v2.value
        assert result == 2
        
        # Test modulo
        result = v1.value % v2.value
        assert result == 0

    def test_array_operations(self):
        # Test array operations (since it uses ArrayDunders.mixin)
        value = NamedValue("array", [1, 2, 3])
        
        # Test length
        assert len(value.value) == 3
        
        # Test iteration
        assert list(value.value) == [1, 2, 3]
        
        # Test indexing
        assert value.value[0] == 1

    def test_complex_types(self):
        # Test with more complex data types
        class Point(BaseModel):
            x: float
            y: float
        
        point = Point(x=1.0, y=2.0)
        value = NamedValue("point", point)
        
        assert value.value.x == 1.0
        assert value.value.y == 2.0

    def test_serialization(self):
        # Test serialization capabilities
        value = NamedValue("test", 42)
        
        # Test model dump
        serialized = value.model_dump()
        assert serialized["name"] == "test"
        
        # Test model dump JSON
        serialized_json = value.model_dump_json()
        assert "test" in serialized_json
        
        # Test deserialization
        deserialized = NamedValue.model_validate({"name": "test"})
        assert deserialized.name == "test"

    def test_list_and_hash_operations(self):
        class MockNamedValueList:
            def __init__(self):
                self.values: List[NamedValue] = []
            
            def append(self, value):
                self.values.append(value)
        
        class MockNamedValueHash:
            def __init__(self):
                self.values = {}
            
            def register_value(self, value):
                self.values[value.name] = value
        
        # Test append_to_value_list
        value_list = MockNamedValueList()
        value1 = NamedValue("test1", 42)
        value1.append_to_value_list(value_list)
        assert value_list.values[0] == value1
        
        # Test register_to_value_hash
        value_hash = MockNamedValueHash()
        value2 = NamedValue("test2", 100)
        value2.register_to_value_hash(value_hash)
        assert value_hash.values["test2"] == value2

    def test_type_handling(self):
        # Test different types of values
        str_value = NamedValue("str_test", "hello")
        assert isinstance(str_value.value, str)
        
        float_value = NamedValue("float_test", 3.14)
        assert isinstance(float_value.value, float)
        
        list_value = NamedValue("list_test", [1, 2, 3])
        assert isinstance(list_value.value, list)
        
        dict_value = NamedValue("dict_test", {"a": 1, "b": 2})
        assert isinstance(dict_value.value, dict)

    def test_error_cases(self):
        # Test various error cases
        with pytest.raises(ValueError):
            value = NamedValue("test")
            _ = value.value  # Accessing unset value
        
        with pytest.raises(ValueError):
            value = NamedValue("test", 42)
            value.value = 100  # Attempting to modify frozen value

    def test_type_casting(self):
        # Test numeric type casting
        int_value = NamedValue("int_test", 42)
        assert float(int_value.value) == 42.0
        assert str(int_value.value) == "42"
        assert bool(int_value.value) == True
        
        # Test float to int casting
        float_value = NamedValue("float_test", 3.14)
        assert int(float_value.value) == 3
        assert str(float_value.value) == "3.14"
        assert bool(float_value.value) == True
        
        # Test string casting
        str_value = NamedValue("str_test", "123")
        assert int(str_value.value) == 123
        assert float(str_value.value) == 123.0
        assert bool(str_value.value) == True
        
        # Test boolean casting
        bool_value = NamedValue("bool_test", True)
        assert int(bool_value.value) == 1
        assert float(bool_value.value) == 1.0
        assert str(bool_value.value) == "True"
        
        # Test zero/empty values casting
        zero_value = NamedValue("zero_test", 0)
        assert bool(zero_value.value) == False
        assert str(zero_value.value) == "0"
        
        empty_str = NamedValue("empty_str_test", "")
        assert bool(empty_str.value) == False

    def test_type_casting_collections(self):
        # Test list casting
        list_value = NamedValue("list_test", [1, 2, 3])
        assert str(list_value.value) == "[1, 2, 3]"
        assert bool(list_value.value) == True
        assert tuple(list_value.value) == (1, 2, 3)
        assert set(list_value.value) == {1, 2, 3}
        
        # Test tuple casting
        tuple_value = NamedValue("tuple_test", (1, 2, 3))
        assert list(tuple_value.value) == [1, 2, 3]
        assert str(tuple_value.value) == "(1, 2, 3)"
        assert bool(tuple_value.value) == True
        
        # Test set casting
        set_value = NamedValue("set_test", {1, 2, 3})
        assert list(sorted(set_value.value)) == [1, 2, 3]
        assert bool(set_value.value) == True

    def test_type_casting_errors(self):
        # Test invalid casting scenarios
        str_value = NamedValue("invalid_str_test", "not_a_number")
        with pytest.raises(ValueError):
            int(str_value.value)
        
        with pytest.raises(ValueError):
            float(str_value.value)
        
        # Test casting of complex types that shouldn't work
        dict_value = NamedValue("dict_test", {"a": 1})
        with pytest.raises(TypeError):
            int(dict_value.value)
        
        # Test casting of None
        none_value = NamedValue("none_test")
        none_value.value = None
        with pytest.raises(TypeError):
            int(none_value.value)

    def test_type_casting_edge_cases(self):
        # Test casting with whitespace strings
        space_str = NamedValue("space_test", "  42  ")
        assert int(space_str.value.strip()) == 42
        assert float(space_str.value.strip()) == 42.0
        
        # Test casting with boolean strings
        bool_str = NamedValue("bool_str_test", "True")
        assert bool(bool_str.value) == True
        
        # Test casting with hexadecimal
        hex_str = NamedValue("hex_test", "0x1F")
        assert int(hex_str.value, 16) == 31
        
        # Test casting with scientific notation
        sci_str = NamedValue("sci_test", "1e-10")
        assert float(sci_str.value) == 1e-10

    def test_type_casting_custom_objects(self):
        class CustomClass:
            def __init__(self, value):
                self._value = value
            
            def __int__(self):
                return int(self._value)
            
            def __float__(self):
                return float(self._value)
            
            def __str__(self):
                return str(self._value)
            
            def __bool__(self):
                return bool(self._value)
        
        # Test casting with custom object that implements casting methods
        custom_obj = CustomClass(42)
        value = NamedValue("custom_test", custom_obj)
        
        assert int(value.value) == 42
        assert float(value.value) == 42.0
        assert str(value.value) == "42"
        assert bool(value.value) == True

    def test_type_casting_inheritance(self):
        # Test casting with inheritance
        class IntegerValue(NamedValue[int]):
            def __init__(self, name: str, value: int = None):
                super().__init__(name, value)
        
        int_value = IntegerValue("test", 42)
        assert isinstance(int_value.value, int)
        assert float(int_value.value) == 42.0
        
        # Test that type hints are preserved
        with pytest.raises(ValueError):
            IntegerValue("test", "not an integer")

    def test_type_checking(self):
        # Test with explicit type parameter
        class IntValue(NamedValue[int]):
            pass
        
        # Valid integer assignment
        int_value = IntValue("test", 42)
        assert int_value.value == 42
        
        # Valid string that can be cast to int
        str_int_value = IntValue("test2", "123")
        assert str_int_value.value == 123
        
        # Invalid type that can't be cast
        with pytest.raises(TypeError, match="must be of type int"):
            IntValue("test3", "not an integer")
        
        # Test with float type
        class FloatValue(NamedValue[float]):
            pass
        
        float_value = FloatValue("test", 3.14)
        assert float_value.value == 3.14
        
        # Test casting int to float
        int_float_value = FloatValue("test2", 42)
        assert int_float_value.value == 42.0
        
        # Test invalid float
        with pytest.raises(TypeError, match="must be of type float"):
            FloatValue("test3", "not a float")
    
    def test_unset_value(self):
        value = NamedValue("test")
        
        # Check that value starts as UNSET
        assert value._stored_value is UNSET.token
        
        # Verify error message for unset value
        with pytest.raises(ValueError, match="Value 'test' has not been set yet"):
            _ = value.value
        
        # Set value and verify state change
        value.value = 42
        assert value._stored_value == 42
        
        # Verify UNSET string representations
        assert str(UNSET.token) == "<UNSET>"
        assert repr(UNSET.token) == "<UNSET>"
        
        # Alternative test using the enum member
        assert str(UNSET.token) == str(UNSET.token)
        assert repr(UNSET.token) == repr(UNSET.token)
    
    def test_any_type_value(self):
        # Test with no type parameter (should accept any type)
        value = NamedValue("test")
        
        # Should accept any type
        value.value = 42
        value.force_set_value("string")
        value.force_set_value([1, 2, 3])
        value.force_set_value({"a": 1})
        
        # All these should work without type errors
        assert isinstance(value.value, dict)
