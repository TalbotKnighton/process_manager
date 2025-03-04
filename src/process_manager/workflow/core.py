"""Core workflow implementation."""
from __future__ import annotations

# Standard library imports
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import datetime
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import threading
import multiprocessing as mp
from contextlib import contextmanager

# External imports
from pydantic import BaseModel, ConfigDict, Field
import json

# Local imports
from process_manager.workflow.process import BaseProcess
from process_manager.workflow.workflow_types import (
    ProcessConfig,
    ProcessResult,
    ProcessState,
    ProcessType,
    WorkflowNode,
)

class WorkflowNode(BaseModel):
    """Represents a node in the workflow graph.
    
    A WorkflowNode encapsulates a process and its dependency information within
    a workflow. It tracks both the process itself and any other processes that
    must complete before this node can execute.
    
    Attributes:
        process (BaseProcess): The process to execute at this node
        dependencies (List[str]): Process IDs that must complete before this node
        required (bool): Whether this node must complete for workflow success
    
    Example:
        ```python
        node = WorkflowNode(
            process=MyProcess(),
            dependencies=["process1", "process2"],
            required=True
        )
        ```
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow BaseProcess type
        protected_namespaces=()  # Allow attributes starting with model_
    )
    
    process: BaseProcess
    dependencies: List[str] = Field(
        default_factory=list,
        description="Process IDs that must complete before this node can execute"
    )
    required: bool = Field(
        default=True,
        description="Whether this node must complete for workflow success"
    )
    
    def validate_dependencies(self, available_nodes: List[str]) -> List[str]:
        """Validate that all dependencies exist in the workflow.
        
        Args:
            available_nodes (List[str]): List of all process IDs in the workflow
            
        Returns:
            List[str]: List of any missing dependencies
            
        Example:
            ```python
            missing = node.validate_dependencies(["process1", "process2"])
            if missing:
                print(f"Missing dependencies: {missing}")
            ```
        """
        return [
            dep for dep in self.dependencies 
            if dep not in available_nodes
        ]
    
    def can_execute(self, completed_nodes: List[str]) -> bool:
        """Check if this node is ready to execute.
        
        A node can execute when all its dependencies have completed.
        
        Args:
            completed_nodes (List[str]): Process IDs that have completed
            
        Returns:
            bool: True if all dependencies are satisfied
            
        Example:
            ```python
            if node.can_execute(["process1", "process2"]):
                await node.process.run(input_data)
            ```
        """
        return all(
            dep in completed_nodes 
            for dep in self.dependencies
        )

class WorkflowPoolManager:
    """Manages thread and process pools across workflows."""
    _instance = None
    _lock = threading.Lock()
    _pools = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_or_create_pools(self, process_id: str, max_threads: Optional[int] = None, max_processes: Optional[int] = None) -> Dict[str, Any]:
        """Get or create pools for a specific process."""
        if process_id not in self._pools:
            self._pools[process_id] = {
                'thread_pool': ThreadPoolExecutor(max_workers=max_threads),
                'process_pool': ProcessPoolExecutor(max_workers=max_processes)
            }
        return self._pools[process_id]

    def cleanup_pools(self, process_id: str) -> None:
        """Cleanup pools for a specific process."""
        if process_id in self._pools:
            pools = self._pools[process_id]
            pools['thread_pool'].shutdown()
            pools['process_pool'].shutdown()
            del self._pools[process_id]

class Workflow:
    """Manages the execution of a workflow graph."""
    
    def __init__(self, 
                 max_processes: Optional[int] = None,
                 max_threads: Optional[int] = None,
                 process_id: Optional[str] = None):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.results: Dict[str, ProcessResult] = {}
        self.max_processes = max_processes
        self.max_threads = max_threads
        self.process_id = process_id or f"workflow_{id(self)}"
        self.pool_manager = WorkflowPoolManager.get_instance()

    @contextmanager
    def get_pools(self):
        """Get pools for this workflow."""
        try:
            pools = self.pool_manager.get_or_create_pools(
                self.process_id,
                self.max_threads,
                self.max_processes
            )
            yield pools['thread_pool'], pools['process_pool']
        finally:
            pass  # Don't cleanup here, pools are reused

    def _run_sync_process(self, process: BaseProcess, input_data: Any) -> ProcessResult:
        """Execute a synchronous process without lambda."""
        return process._sync_execute(input_data)

    async def execute(self, initial_data: Dict[str, Any] = None) -> Dict[str, ProcessResult]:
        """Execute the workflow."""
        try:
            with self.get_pools() as (thread_pool, process_pool):
                # ... (keep existing code until the results processing part)
                initial_data = initial_data or {}
                node_results: Dict[str, ProcessResult] = {}
                completed_nodes = set()
                failed_nodes = set()

                all_nodes = set(self.nodes.keys())
                
                while len(completed_nodes) < len(self.nodes):
                    ready_nodes = [
                        node_id for node_id in all_nodes - completed_nodes
                        if (
                            all(dep in completed_nodes for dep in self.nodes[node_id].dependencies) and
                            all(
                                dep not in failed_nodes or 
                                not self.nodes[dep].required 
                                for dep in self.nodes[node_id].dependencies
                            )
                        )
                    ]

                    if not ready_nodes:
                        remaining_nodes = all_nodes - completed_nodes
                        if remaining_nodes:
                            unmet_dependencies = {
                                node: [
                                    dep for dep in self.nodes[node].dependencies
                                    if dep not in completed_nodes
                                ]
                                for node in remaining_nodes
                            }
                            raise Exception(
                                f"Workflow deadlock detected. Unmet dependencies: {unmet_dependencies}"
                            )
                        break

                    async_nodes = []
                    thread_nodes = []
                    process_nodes = []
                    
                    for node_id in ready_nodes:
                        process_type = self.nodes[node_id].process.config.process_type
                        match process_type:
                            case ProcessType.ASYNC:
                                async_nodes.append(node_id)
                            case ProcessType.THREAD:
                                thread_nodes.append(node_id)
                            case ProcessType.PROCESS:
                                process_nodes.append(node_id)
                            case _:
                                raise ValueError(f"Invalid ProcessType: {process_type}")

                    tasks = []
                    
                    for node_id in ready_nodes:
                        node = self.nodes[node_id]
                        node_input = {}
                        
                        for dep in node.dependencies:
                            if dep in node_results and node_results[dep].success:
                                node_input[dep] = node_results[dep].data
                        
                        if not node.dependencies and node_id in initial_data:
                            node_input = initial_data[node_id]

                        if node_id in async_nodes:
                            tasks.append(node.process.run(node_input))
                        
                        elif node_id in thread_nodes:
                            loop = asyncio.get_running_loop()
                            tasks.append(
                                loop.run_in_executor(
                                    thread_pool,
                                    self._run_sync_process,
                                    node.process,
                                    node_input
                                )
                            )
                        
                        else:  # process_nodes
                            loop = asyncio.get_running_loop()
                            tasks.append(
                                loop.run_in_executor(
                                    process_pool,
                                    self._run_sync_process,
                                    node.process,
                                    node_input
                                )
                            )

                    # Execute all ready nodes in parallel
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results and update workflow state
                    for node_id, result in zip(ready_nodes, results):
                        node = self.nodes[node_id]
                        completed_nodes.add(node_id)

                        if isinstance(result, Exception):
                            node.process.metadata.state = ProcessState.FAILED
                            node.process.metadata.last_error = str(result)
                            failed_nodes.add(node_id)

                            if node.required:
                                if node.process.config.fail_fast:
                                    raise Exception(
                                        f"Critical node {node_id} failed: {str(result)}"
                                    )
                            continue

                        # Wrap the result in ProcessResult if it isn't already
                        if not isinstance(result, ProcessResult):
                            result = ProcessResult(
                                success=True,
                                data=result,
                                execution_time=0,  # We don't have timing info here
                                start_time=datetime.now(),  # Approximate
                                end_time=datetime.now()
                            )

                        node_results[node_id] = result
                        if not result.success:
                            failed_nodes.add(node_id)
                            if node.required and node.process.config.fail_fast:
                                raise Exception(
                                    f"Critical node {node_id} failed: {result.error}"
                                )

                self.results = node_results
                return node_results
    # async def execute(self, initial_data: Dict[str, Any] = None) -> Dict[str, ProcessResult]:
    #     """Execute the workflow."""
    #     try:
    #         with self.get_pools() as (thread_pool, process_pool):
    #             initial_data = initial_data or {}
    #             node_results: Dict[str, ProcessResult] = {}
    #             completed_nodes = set()
    #             failed_nodes = set()

    #             all_nodes = set(self.nodes.keys())
                
    #             while len(completed_nodes) < len(self.nodes):
    #                 ready_nodes = [
    #                     node_id for node_id in all_nodes - completed_nodes
    #                     if (
    #                         all(dep in completed_nodes for dep in self.nodes[node_id].dependencies) and
    #                         all(
    #                             dep not in failed_nodes or 
    #                             not self.nodes[dep].required 
    #                             for dep in self.nodes[node_id].dependencies
    #                         )
    #                     )
    #                 ]

    #                 if not ready_nodes:
    #                     remaining_nodes = all_nodes - completed_nodes
    #                     if remaining_nodes:
    #                         unmet_dependencies = {
    #                             node: [
    #                                 dep for dep in self.nodes[node].dependencies
    #                                 if dep not in completed_nodes
    #                             ]
    #                             for node in remaining_nodes
    #                         }
    #                         raise Exception(
    #                             f"Workflow deadlock detected. Unmet dependencies: {unmet_dependencies}"
    #                         )
    #                     break

    #                 async_nodes = []
    #                 thread_nodes = []
    #                 process_nodes = []
                    
    #                 for node_id in ready_nodes:
    #                     process_type = self.nodes[node_id].process.config.process_type
    #                     match process_type:
    #                         case ProcessType.ASYNC:
    #                             async_nodes.append(node_id)
    #                         case ProcessType.THREAD:
    #                             thread_nodes.append(node_id)
    #                         case ProcessType.PROCESS:
    #                             process_nodes.append(node_id)
    #                         case _:
    #                             raise ValueError(f"Invalid ProcessType: {process_type}")

    #                 tasks = []
                    
    #                 for node_id in ready_nodes:
    #                     node = self.nodes[node_id]
    #                     node_input = {}
                        
    #                     for dep in node.dependencies:
    #                         if dep in node_results and node_results[dep].success:
    #                             node_input[dep] = node_results[dep].data
                        
    #                     if not node.dependencies and node_id in initial_data:
    #                         node_input = initial_data[node_id]

    #                     if node_id in async_nodes:
    #                         tasks.append(node.process.run(node_input))
                        
    #                     elif node_id in thread_nodes:
    #                         loop = asyncio.get_running_loop()
    #                         tasks.append(
    #                             loop.run_in_executor(
    #                                 thread_pool,
    #                                 self._run_sync_process,
    #                                 node.process,
    #                                 node_input
    #                             )
    #                         )
                        
    #                     else:  # process_nodes
    #                         loop = asyncio.get_running_loop()
    #                         tasks.append(
    #                             loop.run_in_executor(
    #                                 process_pool,
    #                                 self._run_sync_process,
    #                                 node.process,
    #                                 node_input
    #                             )
    #                         )

    #                 results = await asyncio.gather(*tasks, return_exceptions=True)

    #                 for node_id, result in zip(ready_nodes, results):
    #                     node = self.nodes[node_id]
    #                     completed_nodes.add(node_id)

    #                     if isinstance(result, Exception):
    #                         node.process.metadata.state = ProcessState.FAILED
    #                         node.process.metadata.last_error = str(result)
    #                         failed_nodes.add(node_id)

    #                         if node.required:
    #                             if node.process.config.fail_fast:
    #                                 raise Exception(
    #                                     f"Critical node {node_id} failed: {str(result)}"
    #                                 )
    #                         continue

    #                     node_results[node_id] = result
    #                     if not result.success:
    #                         failed_nodes.add(node_id)
    #                         if node.required and node.process.config.fail_fast:
    #                             raise Exception(
    #                                 f"Critical node {node_id} failed: {result.error}"
    #                             )

    #             self.results = node_results
    #             return node_results

        except Exception as e:
            for node_id in all_nodes - completed_nodes:
                self.nodes[node_id].process.metadata.state = ProcessState.SKIPPED
            raise

        finally:
            for node_id, result in node_results.items():
                self.nodes[node_id].process.metadata.result = result

    def shutdown(self):
        """Cleanup resources for this workflow."""
        self.pool_manager.cleanup_pools(self.process_id)

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the workflow."""
        self.nodes[node.process.config.process_id] = node

    @property
    def execution_graph(self) -> Dict[str, List[str]]:
        """Build a graph of process dependencies."""
        return {node_id: node.dependencies for node_id, node in self.nodes.items()}

    def to_json(self) -> str:
        """Serialize the workflow state to JSON."""
        workflow_state = {
            "nodes": {
                node_id: {
                    "process_id": node.process.config.process_id,
                    "dependencies": node.dependencies,
                    "required": node.required,
                    "state": node.process.metadata.state.value,
                    "retry_count": node.process.metadata.retry_count,
                    "last_error": node.process.metadata.last_error
                }
                for node_id, node in self.nodes.items()
            },
            "results": {
                node_id: result.dict() for node_id, result in self.results.items()
            }
        }
        return json.dumps(workflow_state, default=str)

def create_workflow(
        max_processes: Optional[int] = None,
        max_threads: Optional[int] = None,
        process_id: Optional[str] = None) -> Workflow:
    """Create a new workflow instance with specified pool sizes.
    
    Args:
        max_processes: Maximum number of processes in the process pool
        max_threads: Maximum number of threads in the thread pool
        process_id: Unique identifier for this workflow instance
        
    Returns:
        Workflow: A new workflow instance
    """
    return Workflow(
        max_processes=max_processes,
        max_threads=max_threads,
        process_id=process_id
    )
# class Workflow:
#     """Manages the execution of a workflow graph."""
    
#     def __init__(self, 
#                  max_processes: Optional[int] = None,
#                  max_threads: Optional[int] = None):
#         self.nodes: Dict[str, WorkflowNode] = {}
#         self.results: Dict[str, ProcessResult] = {}
#         self.thread_pool = ThreadPoolExecutor(max_workers=max_threads)
#         self.process_pool = ProcessPoolExecutor(max_workers=max_processes)

#     def add_node(self, node: WorkflowNode) -> None:
#         """Add a node to the workflow and set up its process."""
#         node.process.set_workflow(self)  # Give process access to shared pools
#         self.nodes[node.process.config.process_id] = node

#     def shutdown(self, wait: bool = True):
#         """Properly shutdown the workflow's resources."""
#         self.thread_pool.shutdown(wait=wait)
#         self.process_pool.shutdown(wait=wait)

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.shutdown()

#     def _run_sync_process(self, process: BaseProcess, input_data: Any) -> ProcessResult:
#         """Execute a synchronous process without lambda."""
#         return process._sync_execute(input_data)

#     async def execute(self, initial_data: Dict[str, Any] = None) -> Dict[str, ProcessResult]:
#         """Execute the workflow."""
#         initial_data = initial_data or {}
#         node_results: Dict[str, ProcessResult] = {}
#         completed_nodes = set()
#         failed_nodes = set()

#         all_nodes = set(self.nodes.keys())
        
#         try:
#             while len(completed_nodes) < len(self.nodes):
#                 # Find nodes that are ready to execute (all dependencies completed)
#                 ready_nodes = [
#                     node_id for node_id in all_nodes - completed_nodes
#                     if (
#                         all(dep in completed_nodes for dep in self.nodes[node_id].dependencies) and
#                         all(
#                             dep not in failed_nodes or 
#                             not self.nodes[dep].required 
#                             for dep in self.nodes[node_id].dependencies
#                         )
#                     )
#                 ]

#                 if not ready_nodes:
#                     remaining_nodes = all_nodes - completed_nodes
#                     if remaining_nodes:
#                         unmet_dependencies = {
#                             node: [
#                                 dep for dep in self.nodes[node].dependencies
#                                 if dep not in completed_nodes
#                             ]
#                             for node in remaining_nodes
#                         }
#                         raise Exception(
#                             f"Workflow deadlock detected. Unmet dependencies: {unmet_dependencies}"
#                         )
#                     break

#                 # Group ready nodes by execution type
#                 async_nodes = []
#                 thread_nodes = []
#                 process_nodes = []
                
#                 for node_id in ready_nodes:
#                     process_type = self.nodes[node_id].process.config.process_type
#                     match process_type:
#                         case ProcessType.ASYNC:
#                             async_nodes.append(node_id)
#                         case ProcessType.THREAD:
#                             thread_nodes.append(node_id)
#                         case ProcessType.PROCESS:
#                             process_nodes.append(node_id)
#                         case _:
#                             raise ValueError(f"Invalid ProcessType: {process_type}")

#                 tasks = []
                
#                 # Prepare input data and tasks for each node
#                 for node_id in ready_nodes:
#                     node = self.nodes[node_id]
#                     node_input = {}
                    
#                     # Gather input data from dependencies
#                     for dep in node.dependencies:
#                         if dep in node_results and node_results[dep].success:
#                             node_input[dep] = node_results[dep].data
                    
#                     # Add any relevant initial data
#                     if not node.dependencies and node_id in initial_data:
#                         node_input = initial_data[node_id]

#                     # Create appropriate task based on process type
#                     if node_id in async_nodes:
#                         tasks.append(node.process.run(node_input))
                    
#                     elif node_id in thread_nodes:
#                         loop = asyncio.get_running_loop()
#                         tasks.append(
#                             loop.run_in_executor(
#                                 self.thread_pool,
#                                 self._run_sync_process,
#                                 node.process,
#                                 node_input
#                             )
#                         )
                    
#                     else:  # process_nodes
#                         loop = asyncio.get_running_loop()
#                         tasks.append(
#                             loop.run_in_executor(
#                                 self.process_pool,
#                                 self._run_sync_process,
#                                 node.process,
#                                 node_input
#                             )
#                         )

#                 # Execute all ready nodes in parallel
#                 results = await asyncio.gather(*tasks, return_exceptions=True)

#                 # Process results and update workflow state
#                 for node_id, result in zip(ready_nodes, results):
#                     node = self.nodes[node_id]
#                     completed_nodes.add(node_id)

#                     if isinstance(result, Exception):
#                         node.process.metadata.state = ProcessState.FAILED
#                         node.process.metadata.last_error = str(result)
#                         failed_nodes.add(node_id)

#                         if node.required:
#                             if node.process.config.fail_fast:
#                                 raise Exception(
#                                     f"Critical node {node_id} failed: {str(result)}"
#                                 )
#                         continue

#                     # Store successful result
#                     node_results[node_id] = result
#                     if not result.success:
#                         failed_nodes.add(node_id)
#                         if node.required and node.process.config.fail_fast:
#                             raise Exception(
#                                 f"Critical node {node_id} failed: {result.error_message}"
#                             )

#             self.results = node_results
#             return node_results

#         except Exception as e:
#             # Update status of all uncompleted nodes to SKIPPED
#             for node_id in all_nodes - completed_nodes:
#                 self.nodes[node_id].process.metadata.state = ProcessState.SKIPPED
#             raise

#         finally:
#             # Ensure we capture final state of all nodes
#             for node_id, result in node_results.items():
#                 self.nodes[node_id].process.metadata.result = result
    
#     def add_node(self, node: WorkflowNode) -> None:
#         """Add a node to the workflow."""
#         self.nodes[node.process.config.process_id] = node

#     @property
#     def execution_graph(self) -> Dict[str, List[str]]:
#         """Build a graph of process dependencies."""
#         graph = {}
#         for node_id, node in self.nodes.items():
#             graph[node_id] = node.dependencies
#         return graph

#     def to_json(self) -> str:
#         """Serialize the workflow state to JSON."""
#         workflow_state = {
#             "nodes": {
#                 node_id: {
#                     "process_id": node.process.config.process_id,
#                     "dependencies": node.dependencies,
#                     "required": node.required,
#                     "state": node.process.metadata.state.value,
#                     "retry_count": node.process.metadata.retry_count,
#                     "last_error": node.process.metadata.last_error
#                 }
#                 for node_id, node in self.nodes.items()
#             },
#             "results": {
#                 node_id: result.dict() for node_id, result in self.results.items()
#             }
#         }
#         return json.dumps(workflow_state, default=str)

# # Create a convenience function for workflow creation
# def create_workflow(
#         max_processes: Optional[int] = None,
#         max_threads: Optional[int] = None) -> Workflow:
#     """Create a new workflow instance with specified pool sizes."""
#     return Workflow(max_processes=max_processes, max_threads=max_threads)