from typing import Iterable
import unittest

import pytest
import numpy as np

from process_manager.data_handlers.custom_serde_definitions.pandantic import PandasDataFrame, PandasSeries
from process_manager.data_handlers.values import NamedValue
from process_manager.data_handlers.custom_serde_definitions import PandasSeries as Series
# from pandas import Series


class TestNamedValueTypes(unittest.TestCase):
    def test_custom_type_validation(self):
        """Test validation with all SerializableValue types"""
        import numpy as np
        import pandas as pd
        from numpydantic import NDArray
        
        # Test pandas DataFrame
        for t in (pd.DataFrame, PandasDataFrame):
            df_value = NamedValue[t](name="df")
            valid_df = pd.DataFrame({'a': [1, 2, 3]})
            
            # Valid DataFrame assignments
            df_value.value = valid_df
            assert isinstance(df_value.value, pd.DataFrame)
            
            # Valid DataFrame conversions
            df_value.force_set_value({'b': [4, 5, 6]})  # Dict
            assert isinstance(df_value.value, pd.DataFrame)
            
            df_value.force_set_value([[1, 2], [3, 4]])  # List of lists
            assert isinstance(df_value.value, pd.DataFrame)
            
            df_value.force_set_value(['not a dataframe'])  # List of strings
            assert isinstance(df_value.value, pd.DataFrame)
            
            # Invalid DataFrame conversions
            with pytest.raises((TypeError, ValueError)):
                df_value.force_set_value('not a dataframe')  # Plain string should fail
            
            df_value.force_set_value(None)
            assert df_value.value.empty
        
        # Test pandas Series
        for t in (pd.Series, PandasSeries):
            series_value = NamedValue[t](name="series")
            valid_series = pd.Series([1, 2, 3])
            
            # Valid Series assignments
            series_value.value = valid_series
            assert isinstance(series_value.value, pd.Series)
            
            # Valid Series conversions
            series_value.force_set_value([4, 5, 6])  # List
            assert isinstance(series_value.value, pd.Series)
            assert list(series_value.value) == [4, 5, 6]
            
            series_value.force_set_value("test string")  # String becomes single-element Series
            assert isinstance(series_value.value, pd.Series)
            assert len(series_value.value) == 1
            assert series_value.value[0] == "test string"
            
            # Only None should raise TypeError for Series
            series_value.force_set_value(None)
            assert series_value.value.empty
        
        for t in (np.ndarray, NDArray):
            # Test numpy array
            array_value = NamedValue[t](name="array")
            valid_array = np.array([1, 2, 3])
            
            # Valid array assignments
            array_value.value = valid_array
            assert isinstance(array_value.value, np.ndarray)
            
            # Valid array conversions
            array_value.force_set_value([4, 5, 6])  # List
            assert isinstance(array_value.value, np.ndarray)
            assert list(array_value.value) == [4, 5, 6]
            
            array_value.force_set_value("test string")  # String becomes array of characters
            assert isinstance(array_value.value, np.ndarray)
            assert array_value.value.dtype.kind in ['U', 'S']  # Unicode or byte string
            
            array_value.force_set_value(42)  # Scalar becomes 0-d array
            assert isinstance(array_value.value, np.ndarray)
            assert array_value.value.item() == 42
            
            array_value.force_set_value(None)
            assert isinstance(array_value.value, np.ndarray)
            assert array_value.value.ndim == 0
            assert array_value.value.dtype == np.dtype('object')

    def test_type_validation_errors(self):
        """Test type validation for basic types where we can enforce stricter rules"""
        
        # Integer validation
        int_value = NamedValue[int](name="num")
        
        # Valid integer conversions
        int_value.value = 42
        assert isinstance(int_value.value, int)
        assert int_value.value == 42
        
        int_value.force_set_value("123")  # String of integer
        assert isinstance(int_value.value, int)
        assert int_value.value == 123
        
        # Invalid integer conversions
        with pytest.raises((TypeError, ValueError)):
            int_value.force_set_value("not a number")
        
        with pytest.raises((TypeError, ValueError)):
            int_value.force_set_value(None)
        
        # Float validation
        float_value = NamedValue[float](name="num")
        
        # Valid float conversions
        float_value.value = 3.14
        assert isinstance(float_value.value, float)
        assert float_value.value == 3.14
        
        float_value.force_set_value("3.14")  # String of float
        assert isinstance(float_value.value, float)
        assert float_value.value == 3.14
        
        # Invalid float conversions
        with pytest.raises((TypeError, ValueError)):
            float_value.force_set_value("not a number")
        
        with pytest.raises((TypeError, ValueError)):
            float_value.force_set_value(None)
        
    def test_inherited_type_validation(self):
        """Test validation for inherited NamedValue[T] classes"""
        class IntegerValue(NamedValue[int]):
            pass
        
        int_value = IntegerValue("num")
        
        # Valid conversions
        int_value.value = 42
        assert int_value.value == 42
        assert isinstance(int_value.value, int)
        
        int_value.force_set_value("123")
        assert int_value.value == 123
        assert isinstance(int_value.value, int)
        
        # Invalid conversions
        with pytest.raises(TypeError) as exc:
            int_value.force_set_value("not a number")
        error_msg = str(exc.value).lower()
        assert "type" in error_msg or "convert" in error_msg

    def test_untyped_validation(self):
        """Test that untyped values preserve their type"""
        value = NamedValue("any")
        
        # Values should keep their original type
        value.value = "123"
        assert value.value == "123"
        assert isinstance(value.value, str)
        
        value.force_set_value(456)
        assert value.value == 456
        assert isinstance(value.value, int)