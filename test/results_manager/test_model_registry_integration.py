import pytest
from pathlib import Path
import tempfile
import shutil

from pydantic import BaseModel
from typing import List, Optional

from pydantic_workflow.results_manager import ResultsManager, register_model
from pydantic_workflow.results_manager.model_registry import get_model_class, clear_registry


class TestModelRegistryIntegration:
    """Test model registry integration with ResultsManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Provides a temporary directory for test data."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def results_manager(self, temp_dir):
        """Provides a ResultsManager instance with a temporary directory."""
        return ResultsManager(temp_dir)
    
    # Remove or modify the setup_method that clears the registry:
    def setup_method(self):
        """Setup for each test."""
        # We don't clear the registry so that models from conftest are preserved
        pass
    
    def test_results_manager_with_registered_models(self, results_manager: ResultsManager):
        """Test storing and retrieving registered models."""
        # Register a model
        @register_model
        class TestResult(BaseModel):
            id: str
            value: float
            tags: List[str] = []
        
        # Create and store instance
        result = TestResult(id="test1", value=42.0, tags=["important", "test"])
        results_manager.set("results/test1", result)
        
        # Retrieve without specifying model class
        retrieved = results_manager.get("results/test1")
        
        # Verify it's the correct type and data
        assert isinstance(retrieved, TestResult)
        assert retrieved.id == "test1"
        assert retrieved.value == 42.0
        assert retrieved.tags == ["important", "test"]
    
    def test_round_trip_multiple_models(self, results_manager: ResultsManager):
        """Test storing and retrieving multiple model types."""
        # Register models
        @register_model
        class UserProfile(BaseModel):
            username: str
            bio: Optional[str] = None
        
        @register_model
        class Comment(BaseModel):
            text: str
            author: str
        
        # Create and store instances
        user = UserProfile(username="testuser", bio="Test bio")
        comment = Comment(text="Great post!", author="testuser")
        
        results_manager.set("users/testuser", user)
        results_manager.set("comments/comment1", comment)
        
        # Retrieve without specifying model classes
        retrieved_user = results_manager.get("users/testuser")
        retrieved_comment = results_manager.get("comments/comment1")
        
        # Verify types and data
        assert isinstance(retrieved_user, UserProfile)
        assert retrieved_user.username == "testuser"
        
        assert isinstance(retrieved_comment, Comment)
        assert retrieved_comment.text == "Great post!"
    
    def test_get_model_class_with_dynamic_model_selection(self):
        """Test dynamically selecting models based on a string identifier."""
        # Register multiple models
        @register_model
        class ImageData(BaseModel):
            width: int
            height: int
            format: str
        
        @register_model
        class TextData(BaseModel):
            content: str
            word_count: int
        
        # Function that uses get_model_class for dynamic model selection
        def create_data(data_type: str, **kwargs):
            model_class = get_model_class(data_type)
            if not model_class:
                raise ValueError(f"Unknown data type: {data_type}")
            return model_class(**kwargs)
        
        # Test dynamic creation
        image = create_data("ImageData", width=800, height=600, format="JPEG")
        text = create_data("TextData", content="Hello world", word_count=2)
        
        # Verify instances
        assert isinstance(image, ImageData)
        assert image.width == 800
        
        assert isinstance(text, TextData)
        assert text.content == "Hello world"
        
        # Test with invalid type
        with pytest.raises(ValueError, match="Unknown data type"):
            create_data("VideoData", duration=120)

    def test_model_namespaces_with_results_manager(self, results_manager: ResultsManager):
        """Test using model namespaces with ResultsManager."""
        # Register models in different namespaces
        @register_model(namespace="app1")
        class App1Model(BaseModel):
            name: str
            value: int
        
        @register_model(namespace="app2")
        class App2Model(BaseModel):
            name: str
            active: bool
        
        # Create instances
        app1_data = App1Model(name="Test", value=42)
        app2_data = App2Model(name="Test", active=True)
        
        # Store data
        results_manager.set("app1/data", app1_data)
        results_manager.set("app2/data", app2_data)
        
        # This will work if ResultsManager is modified to try multiple namespaces
        # or if we explicitly provide the model class
        retrieved_app1 = results_manager.get("app1/data", App1Model)
        retrieved_app2 = results_manager.get("app2/data", App2Model)
        
        assert isinstance(retrieved_app1, App1Model)
        assert isinstance(retrieved_app2, App2Model)
    
    # Add a new test to verify namespace storage and retrieva
    def test_namespace_persistence(self, results_manager: ResultsManager):
        """Test that namespace information is stored and retrieved correctly."""
        # Register models in different namespaces
        @register_model(namespace="custom")
        class CustomNamespaceModel(BaseModel):
            field_a: str
            field_b: int
        
        # Create and store an instance with its namespace
        data = CustomNamespaceModel(field_a="test", field_b=42)
        results_manager.set("custom/data", data, namespace="custom")
        
        # Retrieve without specifying model or namespace - should use stored namespace
        retrieved = results_manager.get("custom/data")
        
        # Verify it worked
        assert isinstance(retrieved, CustomNamespaceModel)
        assert retrieved.field_a == "test"
        assert retrieved.field_b == 42
        
        # Also verify with explicit namespace
        retrieved2 = results_manager.get("custom/data", namespace="custom")
        assert isinstance(retrieved2, CustomNamespaceModel)

    # Add this test function to the TestModelRegistryIntegration class

    def test_auto_namespace_detection(self, results_manager: ResultsManager):
        """Test automatic namespace detection when setting data."""
        # Register a model in a custom namespace
        @register_model(namespace="auto_detect")
        class AutoDetectModel(BaseModel):
            name: str
            value: int
        
        # Create an instance
        data = AutoDetectModel(name="test", value=123)
        
        # Set without specifying namespace
        results_manager.set("auto/detect/model", data)
        
        # Get without specifying namespace
        retrieved = results_manager.get("auto/detect/model")
        
        # Verify it worked
        assert isinstance(retrieved, AutoDetectModel)
        assert retrieved.name == "test"
        assert retrieved.value == 123
        
        # Also verify with explicit namespace
        retrieved2 = results_manager.get("auto/detect/model", namespace="auto_detect")
        assert isinstance(retrieved2, AutoDetectModel)

    # Add this test to the TestModelRegistryIntegration class

    def test_set_with_strict_namespace(self, results_manager: ResultsManager):
        """Test setting data with strict namespace checking."""
        # Register a model in multiple namespaces
        @register_model(namespace="ns1")
        class AmbiguousModel(BaseModel):
            name: str
        
        # Also register in another namespace
        register_model(AmbiguousModel, namespace="ns2")
        
        # Create an instance
        data = AmbiguousModel(name="test")
        
        # Should work with explicit namespace
        results_manager.set("ambiguous/data1", data, namespace="ns1")
        
        # Should work with non-strict mode (default)
        results_manager.set("ambiguous/data2", data)
        
        # Should raise error with strict mode
        with pytest.raises(ValueError, match="Cannot automatically determine namespace"):
            results_manager.set("ambiguous/data3", data, strict_namespace=True)