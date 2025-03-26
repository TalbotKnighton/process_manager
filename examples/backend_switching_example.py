from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

from results_manager import (
    ResultsManager, register_model, SetBehavior,
    FileBackend
)
from results_manager.backends.sqlite_backend import SqliteBackend

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

def main():
    # Example data
    user1 = User(id="user1", name="John Doe", email="john@example.com")
    user2 = User(id="user2", name="Jane Smith", email="jane@example.com")
    
    task1 = Task(task_id="task1", description="Complete report", assigned_to="user1")
    task2 = Task(task_id="task2", description="Review code", assigned_to="user2")
    
    # Create the base directories
    base_dir = Path("./results_data")
    db_path = Path("./results_data/results.db")
    
    # 1. Using the default FileBackend
    print("Using FileBackend:")
    file_manager = ResultsManager(base_dir)
    
    # Store some data
    file_manager.set("users/user1", user1)
    file_manager.set("users/user2", user2)
    file_manager.set("tasks/task1", task1)
    file_manager.set("tasks/task2", task2)
    
    # List and retrieve
    print(f"File backend IDs: {file_manager.list_ids()}")
    retrieved_user = file_manager.get("users/user1")
    print(f"Retrieved user: {retrieved_user}")
    
    # 2. Using SQLiteBackend
    print("\nUsing SqliteBackend:")
    sqlite_backend = SqliteBackend(db_path)
    sqlite_manager = ResultsManager(backend=sqlite_backend)
    
    # Store the same data
    sqlite_manager.set("users/user1", user1)
    sqlite_manager.set("users/user2", user2)
    sqlite_manager.set("tasks/task1", task1)
    sqlite_manager.set("tasks/task2", task2)
    
    # List and retrieve
    print(f"SQLite backend IDs: {sqlite_manager.list_ids()}")
    retrieved_task = sqlite_manager.get("tasks/task1")
    print(f"Retrieved task: {retrieved_task}")
    
    # 3. Demonstrate switching backends at runtime
    print("\nSwitching backends at runtime:")
    
    # Start with file backend
    manager = ResultsManager(base_dir)
    print(f"Current backend type: {type(manager.backend).__name__}")
    
    # Switch to SQLite backend
    manager.backend = sqlite_backend
    print(f"New backend type: {type(manager.backend).__name__}")
    
    # Data should still be accessible
    print(f"Can still access data: {manager.get('users/user1')}")
    
    # Clean up
    file_manager.clear()
    sqlite_manager.clear()

if __name__ == "__main__":
    main()