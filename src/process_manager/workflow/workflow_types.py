# src/process_manager/workflow/types.py
"""Common type definitions for workflow system."""
from __future__ import annotations

# Standard imports
from enum import Enum
from typing import Dict, List, Optional, Any, TYPE_CHECKING, ForwardRef
from datetime import datetime

# External imports
from pydantic import BaseModel, Field, ConfigDict

# Local imports
if TYPE_CHECKING:
    from process_manager.workflow.process import BaseProcess

# No imports from other workflow modules here

class ProcessState(Enum):
    """States a process can be in during execution."""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"

class ProcessType(Enum):
    """Types of process execution strategies."""
    ASYNC = "async"
    THREAD = "thread"
    PROCESS = "process"

class ProcessMetadata(BaseModel):
    """Runtime metadata for a process.
    
    Tracks the current state, progress, and execution history of a process.
    
    Attributes:
        process_id: Unique identifier for the process
        state: Current execution state
        retries: Number of retry attempts made
        retry_count: Alias for retries
        progress: Completion percentage (0-100)
        start_time: When execution started
        end_time: When execution finished
        last_error: Most recent error message
        result: Latest execution result
    """
    process_id: str
    state: ProcessState = ProcessState.WAITING
    retries: int = 0
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_error: Optional[str] = None
    result: Optional[ProcessResult] = None  # Will be populated after execution
    
    @property
    def retry_count(self) -> int:
        """Alias for retries count."""
        return self.retries

class ProcessResult(BaseModel):
    """Result of a process execution.
    
    Contains all information about a process execution attempt including
    timing, success/failure status, and any output data or errors.
    
    Attributes:
        success: Whether execution completed successfully
        data: Output data from the process
        execution_time: Time taken in seconds
        start_time: When execution started
        end_time: When execution finished
        error: Error message if execution failed
        error_type: Type of error that occurred
        error_message: Formatted error message
    """
    success: bool
    data: Any = None
    execution_time: float
    start_time: datetime
    end_time: datetime
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    @property
    def error_message(self) -> Optional[str]:
        """Formatted error message combining type and description."""
        if self.error:
            if self.error_type:
                return f"{self.error_type}: {self.error}"
            return self.error
        return None
    
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Convert result to dictionary format.
        
        Overrides pydantic's dict() to handle datetime serialization.
        """
        base_dict = super().model_dump(*args, **kwargs)
        # Convert datetime objects to ISO format strings
        base_dict['start_time'] = self.start_time.isoformat()
        base_dict['end_time'] = self.end_time.isoformat()
        return base_dict

class ProcessConfig(BaseModel):
    """Configuration settings for a process.
    
    Defines how a process should be executed and handled.
    
    Attributes:
        process_type: Type of execution (async, thread, process)
        process_id: Unique identifier for the process
        timeout: Maximum execution time in seconds
        retry_strategy: Configuration for retry behavior
        fail_fast: Whether to stop workflow on failure
    """
    process_type: ProcessType
    process_id: str
    timeout: Optional[float] = None
    retry_strategy: Optional[RetryStrategy] = None
    fail_fast: bool = False  # Added fail_fast option

class ProcessType(Enum):
    """Types of process execution strategies."""
    ASYNC = "async"
    THREAD = "thread"
    PROCESS = "process"

class RetryStrategy(BaseModel):
    """Configuration for process retry behavior."""
    max_retries: int = 3
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0

class WorkflowNode(BaseModel):
    """Represents a node in the workflow graph."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    process: BaseProcess
    dependencies: List[str] = Field(default_factory=list)
    required: bool = Field(default=True)

    def validate_dependencies(self, available_nodes: List[str]) -> List[str]:
        """Validate that all dependencies exist in the workflow."""
        return [dep for dep in self.dependencies if dep not in available_nodes]

    def can_execute(self, completed_nodes: List[str]) -> bool:
        """Check if this node is ready to execute."""
        return all(dep in completed_nodes for dep in self.dependencies)

# At the end of the file, after WorkflowNode is fully defined
from process_manager.workflow.process import BaseProcess
WorkflowNode.model_rebuild()
