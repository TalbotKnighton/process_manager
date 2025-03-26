from pydantic import BaseModel
from typing import List, Optional
from results_manager import ResultsManager, register_model
from results_manager.manager import SetBehavior

# Register some example models
@register_model
class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

@register_model
class TaskResult(BaseModel):
    task_id: str
    status: str
    value: float
    metadata: Optional[dict] = None

def main():
    # Initialize the manager
    results = ResultsManager("./results_data")
    
    # Store some results
    person = Person(name="John Doe", age=30, email="john@example.com")
    
    # Default behavior: raises error if exists
    results.set("users/john", person)
    
    # Try to set again with different behaviors
    try:
        # This will raise an error
        results.set("users/john", person)
    except FileExistsError as e:
        print(f"Expected error: {e}")
    
    # Same data, will skip
    was_set = results.set("users/john", person, behavior=SetBehavior.SKIP_IF_EXISTS)
    print(f"Data was set: {was_set}")  # Will print False
    
    # Different data with RAISE_IF_DIFFERENT
    different_person = Person(name="John Doe", age=31, email="john@example.com")
    try:
        results.set("users/john", different_person, behavior=SetBehavior.RAISE_IF_DIFFERENT)
    except FileExistsError as e:
        print(f"Expected error for different data: {e}")
    
    # Overwrite with different data
    results.set("users/john", different_person, behavior=SetBehavior.OVERWRITE)
    
    # Verify the data was overwritten
    retrieved_person = results.get("users/john")
    print(f"Retrieved person (after overwrite): {retrieved_person}")
    
    # Add another task with different behavior
    task1 = TaskResult(task_id="123", status="complete", value=0.95)
    results.set(["tasks", "processing", "123"], task1)
    
    # List results
    print(f"All results: {results.list_ids()}")
    
    # Clean up
    results.clear()

if __name__ == "__main__":
    main()