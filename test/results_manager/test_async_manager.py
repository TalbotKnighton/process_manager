# test/results_manager/test_async_manager.py
import pytest
import pytest_asyncio
import json
import asyncio
from pathlib import Path
import tempfile
import shutil

from pydantic import BaseModel
from typing import Optional, List

from results_manager import (
    register_model, SetBehavior, 
    AsyncResultsManager, AsyncFileBackend
)

# Change this:
from conftest import PersonModel, TaskModel, NestedModel

# # To direct imports:
# @register_model
# class PersonModel(BaseModel):
#     name: str
#     age: int
#     email: Optional[str] = None


# @register_model
# class TaskModel(BaseModel):
#     task_id: str
#     status: str
#     value: float
#     metadata: Optional[dict] = None


# @register_model
# class NestedModel(BaseModel):
#     id: str
#     items: List[PersonModel]


# Check if we have SQLite support
try:
    from results_manager.async_backends.sqlite_backend import AsyncSqliteBackend
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


@pytest_asyncio.fixture
async def async_temp_dir():
    """Provides a temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest_asyncio.fixture
async def async_file_manager(async_temp_dir):
    """Provides an AsyncResultsManager with FileBackend."""
    manager = AsyncResultsManager(async_temp_dir)
    yield manager
    # Cleanup is handled by removing the temp directory


@pytest_asyncio.fixture
async def async_sqlite_manager(async_temp_dir):
    """Provides an AsyncResultsManager with SQLiteBackend."""
    if not HAS_SQLITE:
        pytest.skip("SQLite async support not available")
    
    db_path = async_temp_dir / "test.db"
    backend = AsyncSqliteBackend(db_path)
    manager = AsyncResultsManager(backend=backend)
    yield manager


@pytest.fixture
def sample_person():
    """Returns a sample PersonModel model."""
    return PersonModel(name="John Doe", age=30, email="john@example.com")


@pytest.fixture
def sample_task():
    """Returns a sample TaskModel model."""
    return TaskModel(
        task_id="task123",
        status="completed",
        value=0.95,
        metadata={"source": "unit-test"}
    )

class TestAsyncFileBackend:
    """Test AsyncResultsManager with FileBackend."""
    
    @pytest.mark.asyncio
    async def test_set_get(self, async_file_manager, sample_person):
        """Test basic set and get operations."""
        # Set data
        await async_file_manager.set("users/john", sample_person)
        
        # Get data
        retrieved = await async_file_manager.get("users/john", PersonModel)
        
        # Verify
        assert isinstance(retrieved, PersonModel)
        assert retrieved.name == "John Doe"
        assert retrieved.age == 30
    
    @pytest.mark.asyncio
    async def test_exists(self, async_file_manager, sample_person):
        """Test exists method."""
        # Initially doesn't exist
        exists = await async_file_manager.exists("users/john")
        assert not exists
        
        # After setting, it exists
        await async_file_manager.set("users/john", sample_person)
        exists = await async_file_manager.exists("users/john")
        assert exists
    
    @pytest.mark.asyncio
    async def test_list_ids(self, async_file_manager, sample_person, sample_task):
        """Test listing IDs."""
        # Set some data
        await async_file_manager.set("users/john", sample_person)
        await async_file_manager.set("tasks/task1", sample_task)
        
        # List all IDs
        ids = await async_file_manager.list_ids()
        assert len(ids) == 2
        assert "users/john" in ids
        assert "tasks/task1" in ids
        
        # List with prefix
        user_ids = await async_file_manager.list_ids("users")
        assert len(user_ids) == 1
        assert "users/john" in user_ids
    
    @pytest.mark.asyncio
    async def test_delete(self, async_file_manager, sample_person):
        """Test deleting data."""
        # Set data
        await async_file_manager.set("users/john", sample_person)
        assert await async_file_manager.exists("users/john")
        
        # Delete and verify
        result = await async_file_manager.delete("users/john")
        assert result is True
        assert not await async_file_manager.exists("users/john")
        
        # Delete nonexistent
        result = await async_file_manager.delete("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_clear(self, async_file_manager, sample_person, sample_task):
        """Test clearing all data."""
        # Set some data
        await async_file_manager.set("users/john", sample_person)
        await async_file_manager.set("tasks/task1", sample_task)
        
        # Verify data exists
        assert len(await async_file_manager.list_ids()) == 2
        
        # Clear and verify
        await async_file_manager.clear()
        assert len(await async_file_manager.list_ids()) == 0
    
    @pytest.mark.asyncio
    async def test_set_behavior_skip_if_exists(self, async_file_manager, sample_person):
        """Test SetBehavior.SKIP_IF_EXISTS."""
        # First set
        await async_file_manager.set("users/john", sample_person)
        
        # Second set with SKIP_IF_EXISTS should return False
        result = await async_file_manager.set(
            "users/john", 
            sample_person, 
            behavior=SetBehavior.SKIP_IF_EXISTS
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_set_behavior_raise_if_exists(self, async_file_manager, sample_person):
        """Test SetBehavior.RAISE_IF_EXISTS."""
        # First set
        await async_file_manager.set("users/john", sample_person)
        
        # Second set with RAISE_IF_EXISTS should raise
        with pytest.raises(FileExistsError):
            await async_file_manager.set(
                "users/john", 
                sample_person, 
                behavior=SetBehavior.RAISE_IF_EXISTS
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, async_file_manager, sample_person):
        """Test concurrent operations."""
        # Create a list of coroutines to run concurrently
        tasks = []
        for i in range(10):
            tasks.append(async_file_manager.set(f"concurrent/item{i}", sample_person))
        
        # Run them concurrently
        await asyncio.gather(*tasks)
        
        # Verify all items were created
        ids = await async_file_manager.list_ids("concurrent")
        assert len(ids) == 10


@pytest.mark.skipif(not HAS_SQLITE, reason="SQLite async support not available")
class TestAsyncSqliteBackend:
    """Test AsyncResultsManager with SQLiteBackend."""
    
    @pytest.mark.asyncio
    async def test_set_get(self, async_sqlite_manager, sample_person):
        """Test basic set and get operations."""
        # Set data
        await async_sqlite_manager.set("users/john", sample_person)
        
        # Get data
        retrieved = await async_sqlite_manager.get("users/john", PersonModel)
        
        # Verify
        assert isinstance(retrieved, PersonModel)
        assert retrieved.name == "John Doe"
        assert retrieved.age == 30
    
    @pytest.mark.asyncio
    async def test_exists(self, async_sqlite_manager, sample_person):
        """Test exists method."""
        # Initially doesn't exist
        exists = await async_sqlite_manager.exists("users/john")
        assert not exists
        
        # After setting, it exists
        await async_sqlite_manager.set("users/john", sample_person)
        exists = await async_sqlite_manager.exists("users/john")
        assert exists
    
    @pytest.mark.asyncio
    async def test_list_ids(self, async_sqlite_manager, sample_person, sample_task):
        """Test listing IDs."""
        # Set some data
        await async_sqlite_manager.set("users/john", sample_person)
        await async_sqlite_manager.set("tasks/task1", sample_task)
        
        # List all IDs
        ids = await async_sqlite_manager.list_ids()
        assert len(ids) == 2
        assert "users/john" in ids
        assert "tasks/task1" in ids
        
        # List with prefix
        user_ids = await async_sqlite_manager.list_ids("users")
        assert len(user_ids) == 1
        assert "users/john" in user_ids
    
    @pytest.mark.asyncio
    async def test_delete(self, async_sqlite_manager, sample_person):
        """Test deleting data."""
        # Set data
        await async_sqlite_manager.set("users/john", sample_person)
        assert await async_sqlite_manager.exists("users/john")
        
        # Delete and verify
        result = await async_sqlite_manager.delete("users/john")
        assert result is True
        assert not await async_sqlite_manager.exists("users/john")
        
        # Delete nonexistent
        result = await async_sqlite_manager.delete("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_clear(self, async_sqlite_manager, sample_person, sample_task):
        """Test clearing all data."""
        # Set some data
        await async_sqlite_manager.set("users/john", sample_person)
        await async_sqlite_manager.set("tasks/task1", sample_task)
        
        # Verify data exists
        ids = await async_sqlite_manager.list_ids()
        assert len(ids) == 2
        
        # Clear and verify
        await async_sqlite_manager.clear()
        ids = await async_sqlite_manager.list_ids()
        assert len(ids) == 0