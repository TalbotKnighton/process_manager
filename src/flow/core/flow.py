# src/flow/core/registry.py
from __future__ import annotations
from datetime import datetime
import traceback
from typing import Any, Dict, Set, List, Optional, Callable
from collections import defaultdict
import asyncio
import logging
import uuid

from pydantic import BaseModel
from flow.core.results import FlowResult
from flow.core.types import FlowStatus, FlowType
from flow.core.errors import FlowError

logger = logging.getLogger(__name__)

class FlowConfig(BaseModel):
    """Configuration for a flow."""
    name: str
    description: Optional[str] = None
    timeout: Optional[float] = None
    retries: int = 0
    flow_type: FlowType = FlowType.PROCESS

class FlowTree:
    def __init__(self, max_workers: int = 4):
        self._flows: Dict[str, Flow] = {}
        self._prerequisites: Dict[str, Set[str]] = defaultdict(set)
        self._dependent_flows: Dict[str, Set[str]] = defaultdict(set)
        self._max_workers = max_workers
        self._running_flows: Set[str] = set()
        self._completed_flows: Set[str] = set()
        self._results: Dict[str, FlowResult] = {}  # Store results
        self._lock = asyncio.Lock()
        logger.debug(f"{type(self).__name__} initialized")

    async def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of flow execution."""
        async with self._lock:
            total_flows = len(self._flows)
            completed = len(self._completed_flows)
            failed = sum(
                1 for r in self._results.values()
                if r.status == FlowStatus.FAILED
            )
            
            return {
                "total_flows": total_flows,
                "completed_flows": completed,
                "failed_flows": failed,
                "success_rate": completed / total_flows if total_flows > 0 else 0,
                "flow_names": [flow.config.name for flow in self._flows.values()]
            }
    
    def register_flow(self, flow: Flow) -> None:
        """Register a flow with the registry."""
        assert flow.id not in self._flows.keys()
        self._flows[flow.id] = flow
        logger.debug(f"Registered flow: {flow.config.name} ({flow.id})")

    def add_prerequisite(self, flow_id: str, prerequisite_id: str) -> None:
        """Add a prerequisite relationship between flows."""
        if flow_id not in self._flows or prerequisite_id not in self._flows:
            raise FlowError("Both flows must be registered first")
        
        self._prerequisites[flow_id].add(prerequisite_id)
        self._dependent_flows[prerequisite_id].add(flow_id)
        logger.debug(f"Added prerequisite: {prerequisite_id} -> {flow_id}")

    def get_root_flows(self) -> Set[str]:
        """Get all flows with no prerequisites."""
        return {
            flow_id for flow_id in self._flows.keys()
            if not self._prerequisites[flow_id]
        }

    async def get_ready_flows(self) -> Set[str]:
        """Get flows whose prerequisites are all completed."""
        async with self._lock:
            ready_flows = set()
            for flow_id, prerequisites in self._prerequisites.items():
                if (
                    flow_id not in self._running_flows and
                    flow_id not in self._completed_flows and
                    all(p in self._completed_flows for p in prerequisites)
                ):
                    ready_flows.add(flow_id)
            return ready_flows

    async def _execute_flow(self, flow: 'Flow', input_data: Dict[str, Any]) -> None:
        """Execute a single flow and store its result."""
        try:
            logger.debug(f"Executing flow: {flow.config.name}")
            result = await flow.execute(input_data)
            
            async with self._lock:
                self._running_flows.remove(flow.id)
                self._completed_flows.add(flow.id)
                self._results[flow.id] = result  # Store the result
                
            logger.debug(f"Completed flow: {flow.config.name} with result: {result}")
            
        except Exception as e:
            logger.error(f"Flow execution failed: {flow.config.name} - {e}")
            async with self._lock:
                self._running_flows.remove(flow.id)
            raise

    # Add result retrieval methods
    async def get_result(self, flow_id: str) -> Optional[FlowResult]:
        """Get result for a specific flow."""
        async with self._lock:
            return self._results.get(flow_id)

    async def get_results_by_name(self, flow_name: str) -> List[FlowResult]:
        """Get results for all flows with a given name."""
        async with self._lock:
            return [
                result for flow_id, result in self._results.items()
                if self._flows[flow_id].config.name == flow_name
            ]

    async def get_all_results(self) -> Dict[str, FlowResult]:
        """Get all flow results."""
        async with self._lock:
            return self._results.copy()

    def get_flow_by_name(self, name: str) -> Optional[Flow]:
        """Get a flow by its name."""
        for flow in self._flows.values():
            if flow.config.name == name:
                return flow
        return None

    def get_flow_names(self) -> List[str]:
        """Get names of all registered flows."""
        return list(set(flow.config.name for flow in self._flows.values()))

    async def execute_all(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, FlowResult]:
        """Execute all flows in the correct order based on prerequisites."""
        input_data = input_data or {}
        
        async with self._lock:
            self._running_flows.clear()
            self._completed_flows.clear()
            self._results.clear()  # Clear previous results

        while True:
            # Get flows that are ready to execute
            ready_flows = await self.get_ready_flows()
            if not ready_flows:
                # If no flows are ready and none are running, we're done
                if not self._running_flows:
                    break
                # Otherwise wait a bit and check again
                await asyncio.sleep(0.1)
                continue

            # Execute ready flows up to max_workers
            available_workers = self._max_workers - len(self._running_flows)
            if available_workers > 0:
                flows_to_execute = list(ready_flows)[:available_workers]
                execution_tasks = []
                
                for flow_id in flows_to_execute:
                    flow = self._flows[flow_id]
                    async with self._lock:
                        self._running_flows.add(flow_id)
                    
                    execution_tasks.append(self._execute_flow(flow, input_data))
                
                # Execute flows in parallel
                await asyncio.gather(*execution_tasks)

        # Return all results
        return await self.get_all_results()

    async def _execute_flow(self, flow: 'Flow', input_data: Dict[str, Any]) -> None:
        """Execute a single flow and store its result."""
        try:
            logger.debug(f"Executing flow: {flow.config.name}")
            result = await flow.execute(input_data)
            
            async with self._lock:
                self._running_flows.remove(flow.id)
                self._completed_flows.add(flow.id)
                self._results[flow.id] = result
                logger.debug(f"Stored result for {flow.config.name}: {result}")
            
        except Exception as e:
            logger.error(f"Flow execution failed: {flow.config.name} - {e}")
            async with self._lock:
                self._running_flows.remove(flow.id)
            raise

# class FlowRegistry:
#     """Central registry for all flows and their prerequisites."""
    
#     def __init__(self, max_workers: int = 4):
#         self._flows: Dict[str, Flow] = {}
#         self._prerequisites: Dict[str, Set[str]] = defaultdict(set)  # flow_id -> set of prerequisite flow_ids
#         self._dependent_flows: Dict[str, Set[str]] = defaultdict(set)  # flow_id -> set of dependent flow_ids
#         self._max_workers = max_workers
#         self._running_flows: Set[str] = set()
#         self._completed_flows: Set[str] = set()
#         self._lock = asyncio.Lock()
#         logger.debug("FlowRegistry initialized")

#     def register_flow(self, flow: 'Flow') -> None:
#         """Register a flow with the registry."""
#         self._flows[flow.process_id] = flow
#         logger.debug(f"Registered flow: {flow.config.name} ({flow.process_id})")

#     def add_prerequisite(self, flow_id: str, prerequisite_id: str) -> None:
#         """Add a prerequisite relationship between flows."""
#         if flow_id not in self._flows or prerequisite_id not in self._flows:
#             raise FlowError("Both flows must be registered first")
        
#         self._prerequisites[flow_id].add(prerequisite_id)
#         self._dependent_flows[prerequisite_id].add(flow_id)
#         logger.debug(f"Added prerequisite: {prerequisite_id} -> {flow_id}")

#     def get_root_flows(self) -> Set[str]:
#         """Get all flows with no prerequisites."""
#         return {
#             flow_id for flow_id in self._flows.keys()
#             if not self._prerequisites[flow_id]
#         }

#     async def get_ready_flows(self) -> Set[str]:
#         """Get flows whose prerequisites are all completed."""
#         async with self._lock:
#             ready_flows = set()
#             for flow_id, prerequisites in self._prerequisites.items():
#                 if (
#                     flow_id not in self._running_flows and
#                     flow_id not in self._completed_flows and
#                     all(p in self._completed_flows for p in prerequisites)
#                 ):
#                     ready_flows.add(flow_id)
#             return ready_flows

#     async def execute_all(self, input_data: Optional[Dict[str, Any]] = None) -> None:
#         """Execute all flows in the correct order based on prerequisites."""
#         input_data = input_data or {}
        
#         async with self._lock:
#             self._running_flows.clear()
#             self._completed_flows.clear()

#         while True:
#             # Get flows that are ready to execute
#             ready_flows = await self.get_ready_flows()
#             if not ready_flows:
#                 # If no flows are ready and none are running, we're done
#                 if not self._running_flows:
#                     break
#                 # Otherwise wait a bit and check again
#                 await asyncio.sleep(0.1)
#                 continue

#             # Execute ready flows up to max_workers
#             available_workers = self._max_workers - len(self._running_flows)
#             if available_workers > 0:
#                 flows_to_execute = list(ready_flows)[:available_workers]
#                 execution_tasks = []
                
#                 for flow_id in flows_to_execute:
#                     flow = self._flows[flow_id]
#                     async with self._lock:
#                         self._running_flows.add(flow_id)
                    
#                     execution_tasks.append(self._execute_flow(flow, input_data))
                
#                 # Execute flows in parallel
#                 await asyncio.gather(*execution_tasks)

# #     async def _execute_flow(self, flow: 'Flow', input_data: Dict[str, Any]) -> None:
# #         """Execute a single flow and update registry state."""
# #         try:
# #             logger.debug(f"Executing flow: {flow.config.name}")
# #             result = await flow.execute(input_data)
            
# #             async with self._lock:
# #                 self._running_flows.remove(flow.process_id)
# #                 self._completed_flows.add(flow.process_id)
                
# #             logger.debug(f"Completed flow: {flow.config.name}")
            
# #         except Exception as e:
# #             logger.error(f"Flow execution failed: {flow.config.name} - {e}")
# #             async with self._lock:
# #                 self._running_flows.remove(flow.process_id)
# #             raise
class Flow:
    """Core flow implementation."""
    
    @property
    def id(self):
        return self.config.name
    
    def __init__(
        self,
        name: str,
        callable: Callable[[Any], Any],
        flow_tree: Optional[FlowTree] = None,
        required_prerequisites: Optional[List[Flow]] = None,
        optional_prerequisites: Optional[List[Flow]] = None,
    ):
        self.callable = callable
        self.config = FlowConfig(name=name)
        self.status = FlowStatus.PENDING
        logger.debug(f"Initialized flow: {self.config.name} ({self.config.name})")
        self._flow_tree = None

        
        if flow_tree is not None:
            self.register_to_flow_tree(flow_tree)

        if required_prerequisites is not None:
            assert flow_tree is not None
            for prereq in required_prerequisites:
                flow_tree.add_prerequisite(flow_id=self.id, prerequisite_id=prereq.id)
        
        if optional_prerequisites is not None:
            assert flow_tree is not None
            for prereq in optional_prerequisites:
                flow_tree.add_prerequisite(flow_id=self.id, prerequisite_id=prereq.id)
    
    def register_to_flow_tree(self, flow_tree: FlowTree):
        self._flow_tree = flow_tree
        flow_tree.register_flow(self)
        return self
    
    async def execute(self, input_data: Dict[str, Any]) -> FlowResult:
        """Execute the flow."""
        logger.debug(f"Executing flow {self.config.name} with input: {input_data}")
        start_time = datetime.now()
        self.status = FlowStatus.RUNNING
        
        try:
            output = self.callable(input_data)
            logger.debug(f"Flow {self.config.name} produced output: {output}")
            
            result = FlowResult.create_completed(
                process_id=self.config.name,
                output=output,
                start_time=start_time,
                metadata={"flow_name": self.config.name}
            )
            self.status = FlowStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Flow {self.config.name} execution failed: {e}")
            result = FlowResult.create_failed(
                process_id=self.config.name,
                error=str(e),
                start_time=start_time,
                traceback=traceback.format_exc(),
                metadata={"flow_name": self.config.name}
            )
            self.status = FlowStatus.FAILED
            raise
            
        return result
    

    def add_prerequisite(self, prerequisite_flow: Flow) -> None:
        """Add a prerequisite flow."""
        self._flow_tree.add_prerequisite(self.id, prerequisite_flow.id)

    # async def execute(self, input_data: Dict[str, Any]) -> FlowResult:
    #     """Execute the flow (simplified as prerequisites are handled by registry)."""
    #     start_time = datetime.now()
    #     self.status = FlowStatus.RUNNING
        
    #     try:
    #         output = self.processor(input_data)
            
    #         result = FlowResult.create_completed(
    #             process_id=self.process_id,
    #             output=output,
    #             start_time=start_time,
    #             metadata={"flow_name": self.config.name}
    #         )
    #         self.status = FlowStatus.COMPLETED
            
    #     except Exception as e:
    #         result = FlowResult.create_failed(
    #             process_id=self.process_id,
    #             error=str(e),
    #             start_time=start_time,
    #             traceback=traceback.format_exc(),
    #             metadata={"flow_name": self.config.name}
    #         )
    #         self.status = FlowStatus.FAILED
    #         raise
            
    #     return result