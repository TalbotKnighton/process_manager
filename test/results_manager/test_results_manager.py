import pytest
import json
import os
from pathlib import Path

from pydantic import BaseModel, ValidationError

from results_manager import ResultsManager, SetBehavior, FileBackend
from results_manager.manager import SetBehavior

# Import the test models explicitly
from conftest import PersonModel, TaskModel, NestedModel


class TestResultsManagerBasics:
    """Test basic functionality of ResultsManager."""
    
    def setup_method(self):
        """Reset before each test."""
        # Don't clear registry here as it would remove our test models
        pass

    def test_init_creates_directory(self, temp_dir):
        """Test that directory is created if missing."""
        subdir = temp_dir / "new_subdir"
        assert not subdir.exists()
        
        mgr = ResultsManager(subdir)
        assert subdir.exists()
        assert subdir.is_dir()
        
        # Verify we're using FileBackend
        assert mgr._get_backend_type() == "FileBackend"

    def test_init_no_create(self, temp_dir):
        """Test initialization with create_if_missing=False."""
        subdir = temp_dir / "missing_dir"
        
        with pytest.raises(FileNotFoundError):
            ResultsManager(subdir, create_if_missing=False)

    def test_path_from_id_string(self, results_manager):
        """Test generating path from string ID."""
        # Ensure we have a FileBackend for this test
        assert results_manager._get_backend_type() == "FileBackend"
        
        path = results_manager._get_path_from_id("users/john")
        expected = results_manager.backend.base_dir / "users" / "john.json"
        assert path == expected

    def test_path_from_id_list(self, results_manager):
        """Test generating path from list ID."""
        # Ensure we have a FileBackend for this test
        assert results_manager._get_backend_type() == "FileBackend"
        
        path = results_manager._get_path_from_id(["users", "john"])
        expected = results_manager.backend.base_dir / "users" / "john.json"
        assert path == expected

    def test_path_from_id_empty(self, results_manager):
        """Test error when providing empty ID."""
        # Ensure we have a FileBackend for this test
        assert results_manager._get_backend_type() == "FileBackend"
        
        with pytest.raises(ValueError):
            results_manager._get_path_from_id("")

        with pytest.raises(ValueError):
            results_manager._get_path_from_id([])

    def test_exists(self, results_manager, sample_person):
        """Test checking if result exists."""
        # Initially doesn't exist
        assert not results_manager.exists("users/john")
        
        # After setting, it exists
        results_manager.set("users/john", sample_person)
        assert results_manager.exists("users/john")
        
        # But others don't
        assert not results_manager.exists("users/jane")


class TestSetAndGet:
    """Test setting and retrieving data."""
    
    def setup_method(self):
        """Reset before each test."""
        # Don't clear registry here as it would remove our test models
        pass

    def test_set_get_simple(self, results_manager, sample_person):
        """Test basic set and get operations."""
        # Set data
        results_manager.set("users/john", sample_person)
        
        # Get data (provide the model class explicitly since we're not using get_model_class)
        retrieved = results_manager.get("users/john", PersonModel)
        
        # Verify it's the same
        assert retrieved == sample_person
        assert isinstance(retrieved, PersonModel)

    def test_get_with_model_class(self, results_manager, sample_person):
        """Test get with explicit model class."""
        results_manager.set("users/john", sample_person)
        
        # Get with explicit model class
        retrieved = results_manager.get("users/john", PersonModel)
        
        assert retrieved == sample_person
        assert isinstance(retrieved, PersonModel)

    def test_get_nonexistent(self, results_manager):
        """Test error when getting nonexistent data."""
        with pytest.raises(FileNotFoundError):
            results_manager.get("nonexistent", PersonModel)

    def test_nested_models(self, results_manager, nested_model):
        """Test storing and retrieving nested models."""
        results_manager.set("nested/model1", nested_model)
        
        # Provide the model class explicitly
        retrieved = results_manager.get("nested/model1", NestedModel)
        assert retrieved == nested_model
        assert isinstance(retrieved.items[0], PersonModel)

    def test_set_behavior_raise_if_exists(self, results_manager, sample_person):
        """Test SetBehavior.RAISE_IF_EXISTS."""
        # First set works
        results_manager.set("users/john", sample_person)
        
        # Second set raises error
        with pytest.raises(FileExistsError):
            results_manager.set("users/john", sample_person)

    def test_set_behavior_skip_if_exists(self, results_manager, sample_person, same_data_different_values):
        """Test SetBehavior.SKIP_IF_EXISTS."""
        # Set initial data with explicit model class to avoid registry issues
        results_manager.set("users/john", sample_person)
        
        # Get the data to verify it's stored correctly
        retrieved = results_manager.get("users/john", PersonModel)
        assert retrieved == sample_person
        
        # Setting same data with SKIP_IF_EXISTS should return False
        # Add a timeout in case of deadlock
        result = results_manager.set(
            "users/john", 
            sample_person, 
            behavior=SetBehavior.SKIP_IF_EXISTS
        )
        assert result is False  # This should be skipped
        
        # Setting different data with SKIP_IF_EXISTS should succeed
        result = results_manager.set(
            "users/john", 
            same_data_different_values, 
            behavior=SetBehavior.SKIP_IF_EXISTS
        )
        assert result is True  # This should be written
        
        # Verify data is updated
        retrieved = results_manager.get("users/john", PersonModel)
        assert retrieved == same_data_different_values
        
    def test_set_behavior_raise_if_different(self, results_manager, sample_person, same_data_different_values):
        """Test SetBehavior.RAISE_IF_DIFFERENT."""
        # Set initial data
        results_manager.set("users/john", sample_person)
        
        # Getting data requires model class
        person = results_manager.get("users/john", PersonModel)
        
        # Setting same data works
        results_manager.set(
            "users/john", 
            sample_person, 
            behavior=SetBehavior.RAISE_IF_DIFFERENT
        )
        
        # Setting different data raises error
        with pytest.raises(FileExistsError):
            results_manager.set(
                "users/john", 
                same_data_different_values, 
                behavior=SetBehavior.RAISE_IF_DIFFERENT
            )
    
    def test_missing_model_type(self, results_manager, temp_dir):
        """Test handling file without model_type field."""
        # Create JSON without model_type
        path = temp_dir / "no_type.json"
        with open(path, 'w') as f:
            json.dump({"data": {"name": "John", "age": 30, "email": "john@example.com"}}, f)
        
        with pytest.raises(ValueError, match="missing model type"):
            results_manager.get("no_type", PersonModel)
    
    def test_set_behavior_overwrite(self, results_manager, sample_person, same_data_different_values):
        """Test SetBehavior.OVERWRITE."""
        # Set initial data
        results_manager.set("users/john", sample_person)
        
        # Overwrite with different data
        results_manager.set(
            "users/john", 
            same_data_different_values, 
            behavior=SetBehavior.OVERWRITE
        )
        
        # Verify data is updated
        retrieved = results_manager.get("users/john", PersonModel)
        assert retrieved == same_data_different_values
    
    def test_file_structure(self, results_manager, sample_person, temp_dir):
        """Test the created file structure."""
        results_manager.set("users/john", sample_person)
        
        # Ensure we have a FileBackend
        assert results_manager._get_backend_type() == "FileBackend"
        
        # Check that file exists
        expected_path = temp_dir / "users" / "john.json"
        assert expected_path.exists()
        
        # Check file content
        with open(expected_path, 'r') as f:
            data = json.load(f)
            
        assert data["model_type"] == "PersonModel"
        assert data["data"]["name"] == "John Doe"
        assert data["data"]["age"] == 30


class TestListAndDelete:
    """Test listing and deleting operations."""
    
    def setup_method(self):
        """Reset before each test."""
        # Don't clear registry here
        pass

    def test_list_ids_empty(self, results_manager):
        """Test listing IDs on empty manager."""
        assert results_manager.list_ids() == []

    def test_list_ids(self, results_manager, sample_person, sample_task):
        """Test listing all IDs."""
        # Add some data
        results_manager.set("users/john", sample_person)
        results_manager.set("users/jane", sample_person)
        results_manager.set("tasks/task1", sample_task)
        
        # List all IDs
        ids = results_manager.list_ids()
        assert len(ids) == 3
        assert "users/john" in ids
        assert "users/jane" in ids
        assert "tasks/task1" in ids

    def test_list_ids_with_prefix(self, results_manager, sample_person, sample_task):
        """Test listing IDs with prefix."""
        # Add some data
        results_manager.set("users/john", sample_person)
        results_manager.set("users/jane", sample_person)
        results_manager.set("tasks/task1", sample_task)
        
        # List users
        user_ids = results_manager.list_ids("users")
        assert len(user_ids) == 2
        assert "users/john" in user_ids
        assert "users/jane" in user_ids
        
        # List tasks
        task_ids = results_manager.list_ids("tasks")
        assert len(task_ids) == 1
        assert "tasks/task1" in task_ids

    def test_list_nonexistent_prefix(self, results_manager):
        """Test listing with nonexistent prefix."""
        assert results_manager.list_ids("nonexistent") == []

    def test_delete_existing(self, results_manager, sample_person):
        """Test deleting existing data."""
        results_manager.set("users/john", sample_person)
        assert results_manager.exists("users/john")
        
        # Delete and verify
        assert results_manager.delete("users/john") is True
        assert not results_manager.exists("users/john")

    def test_delete_nonexistent(self, results_manager):
        """Test deleting nonexistent data."""
        assert results_manager.delete("nonexistent") is False
    
    # Example for a test in the TestListAndDelete class
    def test_delete_cleanup_empty_dirs(self, results_manager, sample_person, temp_dir):
        """Test that empty directories are cleaned up after delete."""
        # Create a deep path
        results_manager.set(["deep", "path", "to", "item"], sample_person)
        
        # Verify directory structure exists
        # For directory checks, we need to access the backend's base_dir
        assert (results_manager.backend.base_dir / "deep" / "path" / "to").exists()
        
        # Delete and verify cleanup
        results_manager.delete(["deep", "path", "to", "item"])
        
        # Directories should be removed
        assert not (results_manager.backend.base_dir / "deep" / "path" / "to").exists()
        assert not (results_manager.backend.base_dir / "deep" / "path").exists()
        assert not (results_manager.backend.base_dir / "deep").exists()

    def test_clear(self, results_manager, sample_person, sample_task, temp_dir):
        """Test clearing all data."""
        # Add some data
        results_manager.set("users/john", sample_person)
        results_manager.set("tasks/task1", sample_task)
        
        # Verify data exists
        assert len(results_manager.list_ids()) == 2
        
        # Clear and verify
        results_manager.clear()
        assert len(results_manager.list_ids()) == 0
        
        # Base directory still exists
        assert temp_dir.exists()


class TestErrors:
    """Test error handling."""
    
    def setup_method(self):
        """Reset before each test."""
        # Don't clear registry here
        pass

    def test_get_with_wrong_model(self, results_manager, sample_person, sample_task):
        """Test getting data with wrong model type."""
        results_manager.set("users/john", sample_person)
        
        # Try to get as wrong model type
        with pytest.raises(ValidationError):
            results_manager.get("users/john", TaskModel)

    def test_invalid_file_content(self, results_manager, temp_dir):
        """Test handling of invalid file content."""
        # Create invalid JSON file
        path = temp_dir / "invalid.json"
        with open(path, 'w') as f:
            f.write("{invalid json")
        
        with pytest.raises(json.JSONDecodeError):
            results_manager.get("invalid", PersonModel)

    def test_missing_model_type(self, results_manager, temp_dir):
        """Test handling file without model_type field."""
        # Create JSON without model_type
        path = temp_dir / "no_type.json"
        with open(path, 'w') as f:
            json.dump({"data": {"name": "John"}}, f)
        
        with pytest.raises(ValueError, match="missing model type"):
            results_manager.get("no_type", PersonModel)

    def test_unregistered_model_type(self, results_manager, temp_dir):
        """Test handling unregistered model type."""
        # Create JSON with unregistered type
        path = temp_dir / "unknown_type.json"
        with open(path, 'w') as f:
            json.dump({"model_type": "UnknownModel", "data": {}}, f)
        
        with pytest.raises(ValueError, match="not registered"):
            results_manager.get("unknown_type")