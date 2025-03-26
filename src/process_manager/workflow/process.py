"""Base process implementation."""
from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from datetime import datetime
import asyncio
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from typing import Optional, Any, NamedTuple
import traceback
from pydantic import BaseModel

from process_manager.workflow.core import WorkflowPoolManager
from process_manager.workflow.id_generator import ProcessIdGenerator
from process_manager.workflow.workflow_types import (
    ProcessConfig,
    ProcessMetadata,
    ProcessResult,
    ProcessState,
    ProcessType,
    RetryStrategy,
    WorkflowNode,
)

if TYPE_CHECKING:
    from process_manager.workflow.core import Workflow


logger = logging.getLogger(name=__name__)

class ProcessId(NamedTuple):
    """Structured process identifier."""
    base_name: str
    index: Optional[int] = None
    
    def __str__(self) -> str:
        """Convert to string format used in workflow."""
        if self.index is not None:
            return f"{self.base_name}_{self.index}"
        return self.base_name

# Add a global instance of the generator
_default_id_generator = ProcessIdGenerator()

class ProcessConfig(BaseModel):
    """Configuration for all processes."""
    process_id: Optional[str] = None  # Make process_id optional
    process_type: ProcessType = ProcessType.ASYNC
    retry_strategy: Optional[RetryStrategy] = None
    timeout: Optional[float] = None
    validation_rules: Optional[List[Callable]] = None
    fail_fast: bool = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-generate process_id if not provided
        if self.process_id is None:
            self.process_id = _default_id_generator.next_id()


class BaseProcess(ABC):
    """Abstract base class for all process implementations.
    
    The BaseProcess class provides a framework for implementing custom processes
    that can be executed in different contexts (async, threaded, or multiprocess).
    It handles common functionality like execution tracking, error handling, and
    workflow integration.
    
    Key Features:
        - Flexible execution strategies (async, thread, process)
        - Built-in timeout handling
        - Automatic execution time tracking
        - Process state management
        - Error handling and reporting
    
    Attributes:
        config (ProcessConfig): Configuration settings for the process
        metadata (ProcessMetadata): Runtime metadata and state tracking
        _workflow (Optional[Workflow]): Reference to parent workflow
    
    Example:
        ```python
        class MyProcess(BaseProcess):
            def __init__(self):
                super().__init__(ProcessConfig(
                    process_type=ProcessType.THREAD,
                    process_id="my_process"
                ))
            
            async def execute(self, input_data: dict) -> dict:
                # Implement process logic here
                return processed_data
        ```
    """
    
    def __init__(self, config: ProcessConfig):
        """Initialize the process with configuration settings.
        
        Args:
            config (ProcessConfig): Process configuration including:
                - process_type: Execution strategy (ASYNC, THREAD, PROCESS)
                - process_id: Unique identifier for the process
                - timeout: Optional timeout duration in seconds
        """
        self.config = config
        self.metadata = ProcessMetadata(
            process_id=config.process_id,
            state=ProcessState.WAITING
        )
        self._workflow: Optional[Workflow] = None
        
    @classmethod
    def get_process_result(cls, input_data: Dict[str, Any], process_id: ProcessId) -> Any:
        """
        Helper method to consistently extract data from process results.
        
        Args:
            input_data: Dictionary containing process results
            process_id: ProcessId of the process whose result we want
            
        Returns:
            The data from the specified process result
            
        Raises:
            ValueError: If the process result is not found or invalid
        """
        key = str(process_id)
        if key not in input_data:
            raise ValueError(f"Required process {key} not found in input data")
        
        result = input_data[key]
        if isinstance(result, ProcessResult):
            if not result.success:
                raise ValueError(f"Process {key} did not complete successfully")
            return result.data
        else:
            raise ValueError(f"Expected ProcessResult for {key}, got {type(result)}")
    
    def process(self, input_data: Any) -> Any:
        """
        Main processing logic to be implemented by subclasses.
        
        This is the primary method that users should override. It contains
        just the core processing logic without worrying about execution details.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The processed result
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def set_workflow(self, workflow: Workflow) -> None:
        """Set reference to parent workflow for resource access.
        
        This method is called by the workflow when the process is added.
        The workflow reference provides access to shared resources like
        thread and process pools.
        
        Args:
            workflow (Workflow): Parent workflow instance
        """
        self._workflow = workflow
    
    async def execute(self, input_data: Any) -> ProcessResult:
        """
        Execute the process using the appropriate execution strategy.
        
        Uses the parent workflow's pool manager if the process is attached to a workflow,
        otherwise creates a standalone pool manager for independent execution.
        
        Args:
            input_data: Input data for the process
            
        Returns:
            ProcessResult containing the execution result and metadata
        """
        try:
            pool_manager = self._workflow.pool_manager
            process_id = self._workflow.process_id  # Already unique
        except AttributeError:
            # Process is not attached to a workflow, create standalone pool manager
            pool_manager = WorkflowPoolManager.get_instance()
            process_id = f"workflow_{id(self)}"  # Match workflow's ID format
            print(f"Warning: Process {self.config.process_id} is not attached to a workflow. Using standalone pool manager.")
        
        try:
            pools = pool_manager.get_or_create_pools(process_id)
            match self.config.process_type:
                case ProcessType.ASYNC:
                    return await self._sync_execute(input_data)
                    
                case ProcessType.THREAD | ProcessType.PROCESS:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        pools[self.config.process_type],
                        self._sync_execute,
                        input_data
                    )
                    
                case _:
                    raise ValueError(f"Unknown process type: {self.config.process_type}")
        finally:
            pass
            
    async def _run_multiprocess(self, input_data: Any) -> Any:
        """Execute the process in a process pool.
        
        Runs the process's _sync_execute method in a separate process using
        the workflow's process pool. This method is used when the process
        type is ProcessType.PROCESS.
        
        Args:
            input_data (Any): Input data for the process
            
        Returns:
            Any: Process output data
            
        Raises:
            RuntimeError: If process is not attached to a workflow
        """
        if not self._workflow:
            raise RuntimeError("Process not attached to a workflow")
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._workflow.process_pool,
            self._sync_execute,
            input_data
        )
    
    async def _run_threaded(self, input_data: Any) -> Any:
        """Execute the process in a thread pool.
        
        Runs the process's _sync_execute method in a separate thread using
        the workflow's thread pool. This method is used when the process
        type is ProcessType.THREAD.
        
        Args:
            input_data (Any): Input data for the process
            
        Returns:
            Any: Process output data
            
        Raises:
            RuntimeError: If process is not attached to a workflow
        """
        if not self._workflow:
            raise RuntimeError("Process not attached to a workflow")
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._workflow.thread_pool,
            self._sync_execute,
            input_data
        )
    
    async def run(self, input_data: Any) -> ProcessResult:
        """Run the process with the configured execution strategy.
        
        This method handles:
        1. Process state management
        2. Execution time tracking
        3. Error handling
        4. Result packaging
        
        The execution strategy is determined by the process_type setting
        in the configuration (ASYNC, THREAD, or PROCESS).
        
        Args:
            input_data (Any): Input data for the process
            
        Returns:
            ProcessResult: Object containing:
                - success: Whether execution completed successfully
                - data: Process output data
                - execution_time: Time taken in seconds
                - start_time: Execution start timestamp
                - end_time: Execution end timestamp
                - error: Error information if execution failed
                
        Raises:
            ProcessError: If execution fails for any reason
        """
        self.metadata.state = ProcessState.RUNNING
        start_time = datetime.now()
        
        try:
            # Choose execution strategy based on process type
            if self.config.process_type == ProcessType.ASYNC:
                result = await self._run_async(input_data)
            elif self.config.process_type == ProcessType.THREAD:
                result = await self._run_threaded(input_data)
            else:  # ProcessType.PROCESS
                result = await self._run_multiprocess(input_data)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            process_result = ProcessResult(
                success=True,
                data=result,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time
            )
            
            self.metadata.state = ProcessState.COMPLETED
            return process_result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            process_result = ProcessResult(
                success=False,
                data=None,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                error=str(e),
                error_type=type(e).__name__
            )
            
            self.metadata.state = ProcessState.FAILED
            return process_result
    
    async def _run_async(self, input_data: Any) -> Any:
        """Execute the process as an async coroutine.
        
        Runs the process's execute method directly as a coroutine,
        optionally with timeout handling. This method is used when
        the process type is ProcessType.ASYNC.
        
        Args:
            input_data (Any): Input data for the process
            
        Returns:
            Any: Process output data
            
        Raises:
            asyncio.TimeoutError: If execution exceeds configured timeout
        """
        if self.config.timeout:
            return await asyncio.wait_for(
                self.execute(input_data),
                timeout=self.config.timeout
            )
        return await self.execute(input_data)
    
    def _sync_execute(self, input_data: Any) -> Any:
        """Synchronous version of execute for thread/process pools.
        
        This method must be implemented by processes that use THREAD
        or PROCESS execution types. It should contain the synchronous
        version of the process logic.
        
        Args:
            input_data (Any): Input data for the process
            
        Returns:
            Any: Process output data
            
        Raises:
            NotImplementedError: Must be implemented by subclasses using
                               THREAD or PROCESS types
        """
        start_time = datetime.now()
        try:
            # Execute the user's processing logic
            result = self.process(input_data)
            
            end_time = datetime.now()
            return ProcessResult(
                success=True,
                data=result,
                execution_time=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            # logger.error(traceback.format_exc())
            end_time = datetime.now()
            return ProcessResult(
                success=False,
                data=None,
                execution_time=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time,
                error='\n\n\n'+str(traceback.format_exc()),
                error_type=type(e).__name__
            )
        
    def register_to(
        self,
        workflow: Workflow,
        dependencies: Optional[List[str|ProcessId]] = None,
        required: bool = True
    ) -> BaseProcess:
        """
        Register this process as a WorkflowNode to the given workflow.
        
        Args:
            workflow (Workflow): The workflow to register this process to
            dependencies (Optional[List[str|ProcessId]]): Optional list of process IDs that this process depends on
            required (bool): Whether this process is required for workflow completion
            
        Returns:
            (Self): Reference for method chaining
        """
        workflow.add_node(
            WorkflowNode(
                process=self,
                dependencies=dependencies or [],
                required=required,
            )
        )
        return self