def test_type_validation(self):
    # Create specifically typed NamedValue instances
    int_value = dh.NamedValue[int](name="count")
    float_value = dh.NamedValue[float](name="price")
    str_value = dh.NamedValue[str](name="name")

    # Test correct type assignments
    int_value.value = 42
    float_value.value = 10.99
    str_value.value = "John"

    # Test that wrong types raise TypeError
    with self.assertRaises(TypeError) as ctx:
        int_value.value = "not an integer"
    self.assertIn("must be of type int", str(ctx.exception))

    with self.assertRaises(TypeError) as ctx:
        float_value.value = "not a float"
    self.assertIn("must be of type float", str(ctx.exception))

    with self.assertRaises(TypeError) as ctx:
        str_value.value = 42
    self.assertIn("must be of type str", str(ctx.exception))

    # Test valid type conversions
    int_value.value = 123  # Direct integer
    self.assertEqual(int_value.value, 123)
    
    int_value.force_set_value("456")  # String that's a valid integer
    self.assertEqual(int_value.value, 456)
    
    float_value.value = 42.0  # Direct float
    self.assertEqual(float_value.value, 42.0)
    
    float_value.force_set_value("123.45")  # String that's a valid float
    self.assertEqual(float_value.value, 123.45)

if __name__ == '__main__':
    test_type_validation()