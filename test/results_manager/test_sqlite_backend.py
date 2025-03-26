# test/results_manager/test_sqlite_backend.py
import pytest
import tempfile
import shutil
from pathlib import Path

from pydantic import BaseModel
from typing import Optional, List

from results_manager import ResultsManager, SetBehavior, register_model
from results_manager.backends.base import ResultsBackend

# Import the test models from conftest
from conftest import PersonModel, TaskModel, NestedModel

# Check if we have SQLite support
try:
    from results_manager.backends.sqlite_backend import SqliteBackend
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


@pytest.fixture
def sqlite_temp_dir():
    """Provides a temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sqlite_manager(sqlite_temp_dir):
    """Provides a ResultsManager with SQLiteBackend."""
    if not HAS_SQLITE:
        pytest.skip("SQLite support not available")
    
    db_path = sqlite_temp_dir / "test.db"
    backend = SqliteBackend(db_path)
    manager = ResultsManager(backend=backend)
    yield manager


@pytest.mark.skipif(not HAS_SQLITE, reason="SQLite support not available")
class TestSqliteBackend:
    """Test ResultsManager with SQLiteBackend."""

    def test_backend_type(self, sqlite_manager):
        """Test that we're using the correct backend."""
        assert sqlite_manager._get_backend_type() == "SqliteBackend"
    
    def test_set_get(self, sqlite_manager, sample_person):
        """Test basic set and get operations."""
        # Set data
        sqlite_manager.set("users/john", sample_person)
        
        # Get data
        retrieved = sqlite_manager.get("users/john", PersonModel)
        
        # Verify
        assert isinstance(retrieved, PersonModel)
        assert retrieved.name == "John Doe"
        assert retrieved.age == 30
    
    def test_exists(self, sqlite_manager, sample_person):
        """Test exists method."""
        # Initially doesn't exist
        assert not sqlite_manager.exists("users/john")
        
        # After setting, it exists
        sqlite_manager.set("users/john", sample_person)
        assert sqlite_manager.exists("users/john")
    
    def test_list_ids(self, sqlite_manager, sample_person, sample_task):
        """Test listing IDs."""
        # Set some data
        sqlite_manager.set("users/john", sample_person)
        sqlite_manager.set("tasks/task1", sample_task)
        
        # List all IDs
        ids = sqlite_manager.list_ids()
        assert len(ids) == 2
        assert "users/john" in ids
        assert "tasks/task1" in ids
        
        # List with prefix
        user_ids = sqlite_manager.list_ids("users")
        assert len(user_ids) == 1
        assert "users/john" in user_ids
    
    def test_delete(self, sqlite_manager, sample_person):
        """Test deleting data."""
        # Set data
        sqlite_manager.set("users/john", sample_person)
        assert sqlite_manager.exists("users/john")
        
        # Delete and verify
        result = sqlite_manager.delete("users/john")
        assert result is True
        assert not sqlite_manager.exists("users/john")
        
        # Delete nonexistent
        result = sqlite_manager.delete("nonexistent")
        assert result is False
    
    def test_clear(self, sqlite_manager, sample_person, sample_task):
        """Test clearing all data."""
        # Set some data
        sqlite_manager.set("users/john", sample_person)
        sqlite_manager.set("tasks/task1", sample_task)
        
        # Verify data exists
        assert len(sqlite_manager.list_ids()) == 2
        
        # Clear and verify
        sqlite_manager.clear()
        assert len(sqlite_manager.list_ids()) == 0
    
    def test_set_behavior_skip_if_exists(self, sqlite_manager, sample_person):
        """Test SetBehavior.SKIP_IF_EXISTS."""
        # First set
        sqlite_manager.set("users/john", sample_person)
        
        # Second set with SKIP_IF_EXISTS should return False
        result = sqlite_manager.set(
            "users/john", 
            sample_person, 
            behavior=SetBehavior.SKIP_IF_EXISTS
        )
        assert result is False
    
    def test_set_behavior_raise_if_exists(self, sqlite_manager, sample_person):
        """Test SetBehavior.RAISE_IF_EXISTS."""
        # First set
        sqlite_manager.set("users/john", sample_person)
        
        # Second set with RAISE_IF_EXISTS should raise
        with pytest.raises(FileExistsError):
            sqlite_manager.set(
                "users/john", 
                sample_person, 
                behavior=SetBehavior.RAISE_IF_EXISTS
            )