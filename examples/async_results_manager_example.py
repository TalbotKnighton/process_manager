import asyncio
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

from results_manager import (
    register_model, SetBehavior,
    AsyncResultsManager,
    AsyncFileBackend
)

try:
    from results_manager.async_backends.sqlite_backend import AsyncSqliteBackend
    has_sqlite = True
except ImportError:
    has_sqlite = False
    print("aiosqlite not installed, SQLite example will be skipped")

# Define some example models
@register_model
class User(BaseModel):
    id: str
    name: str
    email: Optional[str] = None

@register_model
class Task(BaseModel):
    task_id: str
    description: str
    completed: bool = False
    assigned_to: Optional[str] = None

async def example_file_backend():
    print("\n=== Async File Backend Example ===")
    
    # Create a temporary directory for testing
    base_dir = Path("./async_results_data")
    base_dir.mkdir(exist_ok=True)
    
    # Create the manager with file backend
    manager = AsyncResultsManager(base_dir)
    
    # Store some test data
    user = User(id="user1", name="John Doe", email="john@example.com")
    task = Task(task_id="task1", description="Test async operations", assigned_to="user1")
    
    print("Setting data...")
    await manager.set("users/user1", user)
    await manager.set("tasks/task1", task)
    
    # List all IDs
    print("Listing IDs...")
    ids = await manager.list_ids()
    print(f"Found IDs: {ids}")
    
    # Retrieve data
    print("Retrieving data...")
    retrieved_user = await manager.get("users/user1")
    print(f"Retrieved user: {retrieved_user}")
    
    # Check existence
    print("Checking existence...")
    exists = await manager.exists("tasks/task1")
    print(f"Task exists: {exists}")
    
    # Delete data
    print("Deleting data...")
    await manager.delete("users/user1")
    
    # List IDs after deletion
    print("Listing IDs after deletion...")
    ids = await manager.list_ids()
    print(f"Remaining IDs: {ids}")
    
    # Clear everything
    print("Clearing all data...")
    await manager.clear()
    ids = await manager.list_ids()
    print(f"IDs after clear: {ids}")

async def example_sqlite_backend():
    """Example using the SQLite backend."""
    if not has_sqlite:
        return
    
    print("\n=== Async SQLite Backend Example ===")
    
    # Create a temporary directory for testing
    base_dir = Path("./async_results_data")
    base_dir.mkdir(exist_ok=True)
    db_path = base_dir / "async_results.db"
    
    # Create the SQLite backend
    backend = AsyncSqliteBackend(db_path)
    manager = AsyncResultsManager(backend=backend)
    
    # Store some test data
    user = User(id="user2", name="Jane Smith", email="jane@example.com")
    task = Task(task_id="task2", description="Another async test", assigned_to="user2")
    
    print("Setting data in SQLite...")
    await manager.set("users/user2", user)
    await manager.set("tasks/task2", task)
    
    # List all IDs
    print("Listing IDs from SQLite...")
    ids = await manager.list_ids()
    print(f"Found IDs: {ids}")
    
    # Retrieve data
    print("Retrieving data from SQLite...")
    retrieved_task = await manager.get("tasks/task2")
    print(f"Retrieved task: {retrieved_task}")
    
    # Clear everything
    print("Clearing all SQLite data...")
    await manager.clear()

async def main():
    """Run all async examples."""
    await example_file_backend()
    
    if has_sqlite:
        await example_sqlite_backend()
    
    print("\nAsync examples completed.")

if __name__ == "__main__":
    asyncio.run(main())