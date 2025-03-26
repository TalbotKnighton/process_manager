# test/results_manager/test_backend_switching.py
import pytest
import tempfile
import shutil
from pathlib import Path

from pydantic import BaseModel
from typing import Optional, List

from results_manager import ResultsManager, SetBehavior, register_model, FileBackend

# Import the test models from conftest
from conftest import PersonModel, TaskModel, NestedModel

# Check if we have SQLite support
try:
    from results_manager.backends.sqlite_backend import SqliteBackend
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


@pytest.fixture
def base_dir():
    """Provides a temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def file_backend(base_dir):
    """Provides a FileBackend."""
    return FileBackend(base_dir / "file_data")


@pytest.fixture
def sqlite_backend(base_dir):
    """Provides a SqliteBackend."""
    if not HAS_SQLITE:
        pytest.skip("SQLite support not available")
    return SqliteBackend(base_dir / "sqlite_data.db")


@pytest.fixture
def manager(file_backend):
    """Provides a ResultsManager with initial FileBackend."""
    return ResultsManager(backend=file_backend)


@pytest.mark.skipif(not HAS_SQLITE, reason="SQLite support not available")
class TestBackendSwitching:
    """Test switching between different backends."""

    def test_initial_backend(self, manager):
        """Test initial backend type."""
        assert manager._get_backend_type() == "FileBackend"
    
    def test_switch_to_sqlite(self, manager, sqlite_backend, sample_person):
        """Test switching from file to sqlite backend."""
        # Set data with file backend
        manager.set("users/john", sample_person)
        
        # Verify file backend has the data
        assert manager.exists("users/john")
        
        # Switch to SQLite backend
        manager.backend = sqlite_backend
        assert manager._get_backend_type() == "SqliteBackend"
        
        # File data should not be accessible
        assert not manager.exists("users/john")
        
        # Set data with SQLite backend
        manager.set("users/jane", sample_person)
        
        # Verify SQLite backend has the data
        assert manager.exists("users/jane")
    
    def test_data_isolation_between_backends(self, file_backend, sqlite_backend, sample_person):
        """Test that data is isolated between different backends."""
        # Create managers with different backends
        file_manager = ResultsManager(backend=file_backend)
        sqlite_manager = ResultsManager(backend=sqlite_backend)
        
        # Set data in file backend
        file_manager.set("users/john", sample_person)
        
        # Set different data in sqlite backend
        sqlite_manager.set("users/jane", sample_person)
        
        # Verify each backend only has its own data
        assert file_manager.exists("users/john")
        assert not file_manager.exists("users/jane")
        
        assert sqlite_manager.exists("users/jane")
        assert not sqlite_manager.exists("users/john")
    
    def test_copy_between_backends(self, file_backend, sqlite_backend, sample_person):
        """Test copying data between backends."""
        # Create managers with different backends
        file_manager = ResultsManager(backend=file_backend)
        sqlite_manager = ResultsManager(backend=sqlite_backend)
        
        # Set data in file backend
        file_manager.set("users/john", sample_person)
        
        # Get the data
        person = file_manager.get("users/john", PersonModel)
        
        # Copy to sqlite backend
        sqlite_manager.set("users/john", person)
        
        # Verify data is in both backends
        assert file_manager.exists("users/john")
        assert sqlite_manager.exists("users/john")
        
        # Verify data is the same
        file_person = file_manager.get("users/john", PersonModel)
        sqlite_person = sqlite_manager.get("users/john", PersonModel)
        
        assert file_person.name == sqlite_person.name
        assert file_person.age == sqlite_person.age