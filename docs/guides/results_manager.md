# ResultsManager: Pydantic-Validated Data Storage for Parallel Processing

ResultsManager is a flexible storage system for Pydantic models, designed for keeping track of results from parallel processing tasks. It provides a simple yet powerful interface to store, retrieve, and manage structured data with built-in validation.

## Key Features

- Pydantic Integration: First-class support for storing and retrieving Pydantic models with automatic validation
- Hierarchical Organization: Store data using intuitive path-like IDs (e.g., "projects/project1/results/analysis")
- Multiple Storage Backends: Seamlessly switch between file-based, SQLite, or custom backends
- Namespace Management: Organize models by namespace to prevent naming conflicts
- Concurrency Support: Thread and process safe with proper locking mechanisms
- Async Support: Full async API for use with asyncio-based applications
- Type Safety: Comprehensive type hints and runtime type validation

## When to Use ResultsManager

ResultsManager is ideal for:

- Data Processing Pipelines: Store intermediate and final results from data transformations
- Machine Learning Workflows: Save model artifacts, parameters, and evaluation metrics
- Parallel Task Processing: Manage results from distributed or concurrent processing
- API Result Caching: Store validated results from API calls for reuse
- ETL Processes: Capture extraction, transformation, and loading outputs

## Getting Started
### Installation

``` python
pip install results-manager
```

### Basic Usage

```python
from pydantic import BaseModel
from typing import List, Optional
from results_manager import ResultsManager, register_model

# Define your data models
@register_model
class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

@register_model
class Team(BaseModel):
    name: str
    members: List[Person]

# Create a manager
results = ResultsManager("./data")

# Store some data
person = Person(name="John Doe", age=30, email="john@example.com")
results.set("people/john", person)

team = Team(
    name="Engineering",
    members=[
        Person(name="John Doe", age=30),
        Person(name="Jane Smith", age=28)
    ]
)
results.set("teams/engineering", team)

# Retrieve data later
john = results.get("people/john")
print(f"Retrieved: {john.name}, {john.age}")

# List available results
all_ids = results.list_ids()
print(f"Available results: {all_ids}")

# Find results with a prefix
team_ids = results.list_ids("teams")
print(f"Teams: {team_ids}")

# Check if data exists
if results.exists("people/jane"):
    print("Jane's data exists")
else:
    print("Jane's data not found")

# Delete data when no longer needed
results.delete("people/john")
```

## Storage Backends

ResultsManager offers multiple backends for different use cases:

### File Backend (Default)

The FileBackend stores each result as a separate JSON file in a directory structure that mirrors your ID hierarchy:

``` python
from results_manager import ResultsManager, FileBackend

# Default uses FileBackend
results = ResultsManager("./data")

# Or explicitly specify it
file_backend = FileBackend("./data")
results = ResultsManager(backend=file_backend)
```

Best for:

- Development and testing
- Simple applications
- Small to medium datasets
- Local processing

### SQLite Backend

The SQLiteBackend stores results in a SQLite database for better query performance and atomicity:

``` python
from results_manager import ResultsManager
from results_manager.backends.sqlite_backend import SqliteBackend

sqlite_backend = SqliteBackend("./results.db")
results = ResultsManager(backend=sqlite_backend)
```

Best for:

- Larger datasets
- More frequent updates
- Applications needing transactional consistency
- Situations where you need to query across many results

### Custom Backends

You can implement custom backends by inheriting from ResultsBackend:

``` python
from results_manager import ResultsManager, ResultsBackend

class MyCustomBackend(ResultsBackend):
    # Implement required methods
    # ...

results = ResultsManager(backend=MyCustomBackend())
```

### Switching Backends

One of ResultsManager's most powerful features is the ability to switch backends without changing your application code:

``` python
# Start with file storage during development
results = ResultsManager("./dev_data")

# Later switch to SQLite for production
sqlite_backend = SqliteBackend("./prod.db")
results.backend = sqlite_backend

# Your application code remains unchanged
results.set("key", data)
retrieved = results.get("key")
```

This makes it easy to scale up as your needs grow.

## Async Support

For asyncio-based applications, ResultsManager provides a full async API:

``` python
import asyncio
from results_manager import AsyncResultsManager

async def process_data():
    results = AsyncResultsManager("./data")
    
    # All operations are async
    await results.set("key", data)
    retrieved = await results.get("key")
    
    # Run operations concurrently
    tasks = [
        results.set(f"item/{i}", data) 
        for i in range(10)
    ]
    await asyncio.gather(*tasks)

asyncio.run(process_data())
```

## Namespace Management

ResultsManager uses a model registry with namespace support to avoid naming conflicts:

```python
from results_manager import register_model, get_model_class

# Register in default namespace
@register_model
class User(BaseModel):
    name: str
    
# Register in custom namespace
@register_model(namespace="analytics")
class User(BaseModel):  # Same name, different model
    user_id: str
    visit_count: int

# Get the right model by namespace
user_model = get_model_class("User")  # Default namespace
analytics_user = get_model_class("User", namespace="analytics")
```

## Scaling Your Workflows

ResultsManager is designed to grow with your application needs:

### From Single Process to Distributed Execution

``` python
import concurrent.futures
from results_manager import ResultsManager, SetBehavior

def process_item(item_id):
    # Each process creates its own manager instance
    results = ResultsManager("./results")
    
    # Process the item
    output = compute_result(item_id)
    
    # Store with SKIP_IF_EXISTS to handle cases where another process
    # already completed this item
    results.set(f"items/{item_id}", output, behavior=SetBehavior.SKIP_IF_EXISTS)
    return item_id

# Process items in parallel
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_item, i) for i in range(100)]
    for future in concurrent.futures.as_completed(futures):
        print(f"Completed item {future.result()}")
```

### From Small to Large Datasets

As your data grows, you can easily switch to a more scalable backend:

```python
# During development with small data
results = ResultsManager("./dev_data")

# For production with larger data
from results_manager.backends.sqlite_backend import SqliteBackend
results = ResultsManager(backend=SqliteBackend("./prod.db"))

# Future expansion to other backends
# from results_manager.backends.postgres_backend import PostgresBackend
# results.backend = PostgresBackend(connection_string)
```

## Why ResultsManager?

### Compared to Simple File Storage

- Type Safety: Automatic validation of data structures
- Organization: Hierarchical IDs vs. flat files
- Concurrency: Built-in locking for safe concurrent access
- Flexibility: Multiple backend options

### Compared to Databases

- Simplified Interface: No SQL or ORM knowledge required
- Schema Flexibility: Models can evolve without migrations
- Type Validation: Automatic through Pydantic
- Python-Native: Works directly with Python objects

### Compared to Key-Value Stores

- Rich Data Models: Full Pydantic model support vs. simple values
- Hierarchical Structure: Natural organization vs. flat namespaces
- Type Safety: Strongly typed vs. schema-less

## Real-World Use Cases

### Machine Learning Experiment Tracking

``` python
from results_manager import ResultsManager, register_model

@register_model
class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    training_time: float
    parameters: Dict[str, Any]

results = ResultsManager("./experiments")

# Track experiment results
metrics = ModelMetrics(
    model_name="RandomForest",
    accuracy=0.92,
    precision=0.89,
    recall=0.94,
    training_time=45.2,
    parameters={"n_estimators": 100, "max_depth": 10}
)
results.set("models/random_forest/run_1", metrics)

# Later, analyze all experiments
for result_id in results.list_ids("models/random_forest"):
    metrics = results.get(result_id, ModelMetrics)
    print(f"{result_id}: Accuracy={metrics.accuracy}, Time={metrics.training_time}s")
```

### Data Processing Pipeline

``` python
from results_manager import ResultsManager, register_model

@register_model
class RawData(BaseModel):
    # Raw data schema
    ...

@register_model
class ProcessedData(BaseModel):
    # Processed data schema
    ...

@register_model
class AnalysisResult(BaseModel):
    # Analysis results schema
    ...

results = ResultsManager("./pipeline_data")

# Stage 1: Extract data
raw_data = extract_data()
results.set("pipeline/extraction", raw_data)

# Stage 2: Process data
raw = results.get("pipeline/extraction", RawData)
processed = process_data(raw)
results.set("pipeline/processing", processed)

# Stage 3: Analyze data
processed = results.get("pipeline/processing", ProcessedData)
analysis = analyze_data(processed)
results.set("pipeline/analysis", analysis)

# Get final results any time later
final_results = results.get("pipeline/analysis", AnalysisResult)
```

## Conclusion

ResultsManager provides a robust solution for managing structured data in Python applications. Its combination of type safety, flexible storage options, and intuitive interface makes it an ideal choice for data processing, machine learning workflows, and parallel task management.

Whether you're working on a small personal project or a large-scale data processing pipeline, ResultsManager adapts to your needs and grows with your application.


<!-- 
### Handling Existing Data
ResultsManager provides different behaviors for handling existing data:

``` python
from results_manager import ResultsManager, SetBehavior, register_model

results = ResultsManager("./data")

# Default behavior - raise error if data exists
try:
    results.set("key1", data)  # If already exists, raises FileExistsError
except FileExistsError:
    print("Data already exists")

# Skip if data already exists (returns False if skipped)
was_set = results.set("key2", data, behavior=SetBehavior.SKIP_IF_EXISTS)
if not was_set:
    print("Data was not changed")

# Raise error only if different data exists
results.set("key3", data, behavior=SetBehavior.RAISE_IF_DIFFERENT)

# Always overwrite existing data
results.set("key4", data, behavior=SetBehavior.OVERWRITE)
``` -->