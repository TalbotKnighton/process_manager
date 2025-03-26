# Flow System Design

## Core Goals
[previous content remains the same until "Current Implementation" section]

## Current Implementation and Code References

### Existing Codebase References
From `src/flow/core/types.py`:
```python
class FlowType(Enum):
    INLINE = "inline"
    THREAD = "thread"
    PROCESS = "process"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"

class FlowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StorageType(Enum):
    MEMORY = "memory"
    DISK = "disk"
    SQLITE = "sqlite"
```

### Key Implementation References Needed
Please copy and save these code snippets as they are essential references for continuing development:

1. Core Flow Class Structure:
```python
class Flow:
    def __init__(
        self,
        processor: Processable,
        config: FlowConfig,
        process_id: Optional[str] = None
    ):
        self.processor = processor
        self.config = config
        self.process_id = process_id or str(uuid.uuid4())
        self.context = FlowContext.get_instance()
        self._parent_flow = None
        self._dependencies: Dict[str, DependencyType] = {}
        self._dependent_flows: Set[str] = set()

    def register_to(
        self,
        parent_flow: 'Flow',
        required_deps: Optional[List[str]] = None,
        optional_deps: Optional[List[str]] = None
    ) -> None:
        """Register this flow as a child of another flow with specified dependencies."""
        # [Implementation details from our discussion]
        pass
```

2. Dependency Data Interface:
```python
class DependencyData(BaseModel):
    """Explicit interface for dependency data access"""
    def __init__(self, results_manager: ResultsManager, process_id: str):
        self._results_manager = results_manager
        self._process_id = process_id

    async def get(self, dep_name: str) -> Any:
        return await self._results_manager.get_dependency_data(
            self._process_id, 
            dep_name
        )

    async def get_all(self) -> Dict[str, Any]:
        return await self._results_manager.get_dependencies_data(
            self._process_id
        )
```

3. Flow Configuration:
```python
class FlowConfig(BaseModel):
    name: str
    description: Optional[str] = None
    timeout: Optional[float] = None
    retries: int = 0
    flow_type: FlowType = FlowType.PROCESS
    
    class Config:
        frozen = True
```

4. Basic Result Structure:
```python
class FlowResult(BaseModel):
    process_id: str
    status: FlowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
```

## Next Steps and Implementation Plan

### 1. ProcessPoolManager Implementation
Need to implement:
- Task submission and execution
- Resource management
- Timeout handling
- Pool cleanup

### 2. ResultsManager Implementation
Need to implement:
- Result storage interface
- Memory/disk storage handling
- Dependency data access
- Serialization/deserialization

### 3. FlowContext Implementation
Need to implement:
- Flow registration
- Dependency validation
- Cycle detection
- Service coordination

### 4. Error Handling System
Need to implement:
- Error propagation
- Optional dependency bypass
- Retry logic
- Error reporting

## Prompt to Resume Development
"We are developing a flow management system with the core Flow class implementation as shown in the code references above. The system uses the existing types from src/flow/core/types.py. Please continue implementing the supporting managers (ProcessPoolManager, ResultsManager, FlowContext) and their integration with the Flow class.

The core Flow class uses composition to wrap user-defined processors and handles flow execution, dependency management, and error handling. Dependencies are defined during flow registration rather than configuration.

Please implement [specific manager] with integration into the existing Flow class structure."

When resuming development, specify which manager to implement first, and reference the code structures provided above.

## Implementation Status

### Completed Components

1. Core Flow System (`core/flow.py`):
   - Unified Flow class that wraps user processors
   - Dependency management through registration
   - Configurable execution types (inline/thread/process)
   - Error handling and retries
   - Support for optional and required dependencies
   - Async execution with cancellation support

2. Flow Context (`core/context.py`):
   - Singleton manager for system coordination
   - Dependency graph management
   - Cycle detection
   - Flow registration and tracking
   - Failure propagation handling

3. Results Manager (`core/results.py`):
   - Flexible storage backends (memory/disk/sqlite)
   - Result serialization/deserialization
   - Dependency data access
   - Cleanup and garbage collection
   - Async operation support

4. Process Pool Manager (`execution/pool.py`):
   - Thread and process pool management
   - Task submission and cancellation
   - Timeout handling
   - Resource cleanup
   - Execution context management

5. Supporting Types (`core/types.py`):
   - FlowType (INLINE/THREAD/PROCESS/PARALLEL/SEQUENTIAL)
   - FlowStatus (PENDING/RUNNING/COMPLETED/FAILED/CANCELLED)
   - StorageType (MEMORY/DISK/SQLITE)

### Current Usage Pattern

```python
# Define processor with Pydantic models
class MyProcessor(BaseModel):
    class Input(BaseModel):
        data: List[float]
        
    class Output(BaseModel):
        result: List[float]
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = self.Input(**input_data)
        # Processing logic
        return self.Output(result=processed_data).model_dump()

# Create and configure flow
flow = Flow(
    processor=MyProcessor(),
    config=FlowConfig(
        name="my_flow",
        flow_type=FlowType.PROCESS
    )
)

# Register dependencies
flow.register_to(
    parent_flow,
    required_deps=[dep1.process_id],
    optional_deps=[dep2.process_id]
)

# Execute
result = await flow.execute({"data": [1, 2, 3]})
```

### Next Steps

1. Testing Framework
   - Unit tests for each component
   - Integration tests for flow execution
   - Performance testing
   - Error handling scenarios

2. Additional Features
   - Flow monitoring and metrics
   - Progress tracking
   - Flow visualization
   - Dynamic dependency resolution
   - Flow templating
   - Flow versioning
   - Data streaming between flows
   - Resource limiting
   - Priority queuing

3. Storage Enhancements
   - Redis backend support
   - S3/cloud storage support
   - Compressed storage
   - Result caching
   - Partial result loading

4. Execution Enhancements
   - Distributed execution
   - GPU support
   - Flow scheduling
   - Flow throttling
   - Automatic retries with backoff
   - Resource-aware execution

5. Documentation
   - API documentation
   - Usage examples
   - Best practices
   - Performance guidelines
   - Error handling guide
   - Storage configuration guide

6. Developer Tools
   - CLI for flow management
   - Flow debugging tools
   - Performance profiling
   - Flow validation tools
   - Configuration validation

### Immediate Priorities
1. Implement comprehensive testing suite
2. Add flow monitoring and metrics
3. Enhance error handling with more scenarios
4. Add progress tracking
5. Create detailed documentation

Would you like to focus on any of these next steps?