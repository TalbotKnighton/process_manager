import pytest
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional

from pydantic_workflow.results_manager.model_registry import DEFAULT_NAMESPACE, find_model_in_all_namespaces, find_model_namespace, register_model, get_model_class, _MODEL_REGISTRY


class TestModelRegistry:
    """Test model registry functionality."""
    
    def setup_method(self):
        """Reset the model registry before each test."""
        # This helps ensure tests don't interfere with each other
        _MODEL_REGISTRY.clear()

    def test_register_and_get_model(self):
        """Test registering and retrieving a model class."""
        # Create a model class
        @register_model
        class TestModel(BaseModel):
            name: str
            value: int
        
        # Retrieve the model class
        retrieved = get_model_class("TestModel")
        
        # Verify it's the same class
        assert retrieved is TestModel
        
        # Create an instance to verify the class works
        instance = retrieved(name="test", value=42)
        assert instance.name == "test"
        assert instance.value == 42

    def test_get_nonexistent_model(self):
        """Test getting a nonexistent model."""
        assert get_model_class("NonexistentModel") is None

    def test_register_with_decorator(self):
        """Test using register_model as a decorator."""
        @register_model
        class DecoratedModel(BaseModel):
            field: str
        
        # Verify the decorator returns the class
        assert DecoratedModel.__name__ == "DecoratedModel"
        
        # Check it was registered
        assert get_model_class("DecoratedModel") is DecoratedModel

    def test_register_multiple_models(self):
        """Test registering multiple models."""
        @register_model
        class Model1(BaseModel):
            field1: str
            
        @register_model
        class Model2(BaseModel):
            field2: int
            
        # Verify both are registered
        assert get_model_class("Model1") is Model1
        assert get_model_class("Model2") is Model2
    
    def test_case_sensitivity(self):
        """Test that model names are case sensitive."""
        @register_model
        class CaseSensitiveModel(BaseModel):
            field: str
        
        # Exact case works
        assert get_model_class("CaseSensitiveModel") is CaseSensitiveModel
        
        # Different case doesn't work
        assert get_model_class("casesensitivemodel") is None
        assert get_model_class("CASESENSITIVEMODEL") is None
    
    def test_model_with_methods(self):
        """Test registering and retrieving a model with methods."""
        @register_model
        class ModelWithMethods(BaseModel):
            value: float
            
            def double(self) -> float:
                return self.value * 2
            
            @property
            def squared(self) -> float:
                return self.value ** 2
        
        # Get model class
        retrieved = get_model_class("ModelWithMethods")
        
        # Create instance and test methods
        instance = retrieved(value=3.0)
        assert instance.double() == 6.0
        assert instance.squared == 9.0
    
    def test_model_with_nested_structure(self):
        """Test registering and retrieving models with nested structures."""
        @register_model
        class Address(BaseModel):
            street: str
            city: str
        
        @register_model
        class Person(BaseModel):
            name: str
            addresses: List[Address]
        
        # Get model classes
        person_class = get_model_class("Person")
        address_class = get_model_class("Address")
        
        # Create instances
        address = address_class(street="123 Main St", city="Anytown")
        person = person_class(name="John", addresses=[address])
        
        # Verify nested structure
        assert person.addresses[0].street == "123 Main St"
        assert isinstance(person.addresses[0], Address)
    
    def test_model_with_validation(self):
        """Test registering and retrieving a model with validation."""
        @register_model
        class ValidatedModel(BaseModel):
            id: str
            count: int = Field(gt=0)  # must be greater than 0
        
        # Get model class
        model_class = get_model_class("ValidatedModel")
        
        # Valid instance
        valid = model_class(id="123", count=5)
        assert valid.count == 5
        
        # Invalid instance
        with pytest.raises(ValidationError):
            model_class(id="123", count=0)  # violates gt=0 constraint
    
    def test_model_with_custom_init(self):
        """Test registering and retrieving a model with custom __init__."""
        @register_model
        class ModelWithCustomInit(BaseModel):
            value: int
            doubled: int
            
            def __init__(self, **data):
                # Add doubled value before init
                if 'value' in data and 'doubled' not in data:
                    data['doubled'] = data['value'] * 2
                super().__init__(**data)
        
        # Get model class
        model_class = get_model_class("ModelWithCustomInit")
        
        # Create instance
        instance = model_class(value=5)
        
        # Check custom init behavior
        assert instance.doubled == 10
    
    def test_model_name_collision(self):
        """Test handling of model name collisions."""
        @register_model
        class CollisionTest(BaseModel):
            field1: str
        
        # Try to register another model with the same name
        @register_model
        class CollisionTest(BaseModel):  # Same name but different class
            field2: int
        
        # The most recently registered model should be used
        model_class = get_model_class("CollisionTest")
        
        # Check if the model has field2 (from the second definition)
        # Using model_fields instead of __fields__ (which is deprecated in Pydantic v2)
        assert 'field2' in model_class.model_fields
        assert 'field1' not in model_class.model_fields
    
    def test_model_registration_with_inheritance(self):
        """Test registration of models with inheritance."""
        @register_model
        class ParentModel(BaseModel):
            parent_field: str
        
        @register_model
        class ChildModel(ParentModel):
            child_field: int
        
        # Get model classes
        parent_class = get_model_class("ParentModel")
        child_class = get_model_class("ChildModel")
        
        # Check parent model
        parent = parent_class(parent_field="test")
        assert parent.parent_field == "test"
        
        # Check child model has both fields
        child = child_class(parent_field="test", child_field=42)
        assert child.parent_field == "test"
        assert child.child_field == 42
        
        # Child should be instance of both
        assert isinstance(child, ChildModel)
        assert isinstance(child, ParentModel)

    # Add this test to the TestModelRegistryNamespaces class

    def test_find_model_namespace(self):
        """Test finding the namespace for a model class."""
        # Register models in different namespaces
        @register_model
        class DefaultModel(BaseModel):
            field: str
        
        @register_model(namespace="ns1")
        class CustomModel(BaseModel):
            field: int
        
        @register_model(namespace="ns2")
        class DualModel(BaseModel):
            field: bool
        
        # Also register DualModel in default namespace
        register_model(DualModel)
        
        # Find namespaces
        default_ns = find_model_namespace(DefaultModel)
        custom_ns = find_model_namespace(CustomModel)
        dual_ns = find_model_namespace(DualModel)
        
        # Verify results
        assert default_ns == DEFAULT_NAMESPACE
        assert custom_ns == "ns1"
        # For DualModel, should prioritize non-default namespace
        assert dual_ns == "ns2"

    def test_find_model_in_all_namespaces(self):
        """Test finding a model in all namespaces."""
        # Register same-named models in different namespaces
        @register_model
        class SharedName(BaseModel):
            default_field: str
        
        @register_model(namespace="ns1")
        class SharedName(BaseModel):
            ns1_field: int
        
        @register_model(namespace="ns2")
        class SharedName(BaseModel):
            ns2_field: bool
        
        # Find all models with the name
        matches = find_model_in_all_namespaces("SharedName")
        
        # Verify results
        assert len(matches) == 3
        namespaces = [ns for ns, _ in matches]
        assert DEFAULT_NAMESPACE in namespaces
        assert "ns1" in namespaces
        assert "ns2" in namespaces
        
        # Check the model classes are different
        models = {ns: model for ns, model in matches}
        assert "default_field" in models[DEFAULT_NAMESPACE].model_fields
        assert "ns1_field" in models["ns1"].model_fields
        assert "ns2_field" in models["ns2"].model_fields

    # Add this test to the TestModelRegistryNamespaces class

    def test_find_model_namespace_strict(self):
        """Test finding namespace with strict mode."""
        # Register a model in multiple namespaces
        @register_model(namespace="ns1")
        class MultiNsModel(BaseModel):
            field: str
        
        # Also register in another namespace
        register_model(MultiNsModel, namespace="ns2")
        
        # Should work in non-strict mode
        ns = find_model_namespace(MultiNsModel, strict=False)
        assert ns in ["ns1", "ns2"]  # Will return one of them
        
        # Should raise error in strict mode
        with pytest.raises(ValueError, match="multiple non-default namespaces"):
            find_model_namespace(MultiNsModel, strict=True)