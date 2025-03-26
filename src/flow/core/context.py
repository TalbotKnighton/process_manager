"""Flow context management and coordination."""
from __future__ import annotations
import networkx as nx
from typing import Dict, Set, Optional, Any, TYPE_CHECKING
import asyncio
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

if TYPE_CHECKING:
    from flow.core.flow import FlowResult, Flow
from flow.core.results import ResultsManager
from flow.core.types import FlowStatus, FlowType
from flow.core.errors import FlowError

logger = logging.getLogger(__name__)

class FlowContext:
    """Central manager for flow coordination and service access."""
    _instance = None
    
    def __init__(self):
        if FlowContext._instance is not None:
            raise RuntimeError("FlowContext is a singleton - use get_instance()")
        
        self._flow_graph = nx.DiGraph()
        self._flows: Dict[str, Flow] = {}
        self._status_locks: Dict[str, asyncio.Lock] = {}
        self._execution_locks: Dict[str, asyncio.Lock] = {}
        
        # Initialize managers
        self.results_manager = ResultsManager(context=self)
        
        from flow.execution.pool import ProcessPoolManager
        self.pool_manager = ProcessPoolManager()
        
        logger.info("FlowContext initialized")

    @classmethod
    def get_instance(cls) -> FlowContext:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_flow(self, flow: Flow) -> None:
        """Register a flow with the context."""
        if flow.process_id in self._flows:
            raise FlowError(f"Flow with id {flow.process_id} already registered")
        
        self._flows[flow.process_id] = flow
        self._flow_graph.add_node(flow.process_id)
        self._status_locks[flow.process_id] = asyncio.Lock()
        self._execution_locks[flow.process_id] = asyncio.Lock()
        
        logger.debug(f"Registered flow: {flow.process_id}")

    def get_flow(self, process_id: str) -> Optional[Flow]:
        """Get a flow by its process ID."""
        return self._flows.get(process_id)

    def register_dependency(self, parent_id: str, child_id: str) -> None:
        """Register a dependency relationship between flows."""
        if parent_id not in self._flows or child_id not in self._flows:
            raise FlowError(f"Cannot register dependency - flows not found")
        
        self._flow_graph.add_edge(parent_id, child_id)
        
        # Validate no cycles were created
        if not nx.is_directed_acyclic_graph(self._flow_graph):
            self._flow_graph.remove_edge(parent_id, child_id)
            raise FlowError("Adding dependency would create a cycle")
        
        logger.debug(f"Registered dependency: {parent_id} -> {child_id}")

    def has_cycle(self, start_id: str) -> bool:
        """Check if adding a flow would create a cycle."""
        try:
            nx.find_cycle(self._flow_graph, source=start_id)
            return True
        except nx.NetworkXNoCycle:
            return False

    async def wait_for_flows(self, flow_ids: Set[str], timeout: Optional[float] = None) -> None:
        """Wait for multiple flows to complete."""
        if not flow_ids:
            return

        async def wait_for_flow(flow_id: str) -> None:
            flow = self._flows.get(flow_id)
            if not flow:
                raise FlowError(f"Flow {flow_id} not found")
            
            async with self._status_locks[flow_id]:
                while flow.status not in (FlowStatus.COMPLETED, FlowStatus.FAILED):
                    await asyncio.sleep(0.1)

        # Wait for all flows with timeout
        wait_tasks = [wait_for_flow(fid) for fid in flow_ids]
        try:
            await asyncio.wait_for(asyncio.gather(*wait_tasks), timeout=timeout)
        except asyncio.TimeoutError:
            raise FlowError(f"Timeout waiting for flows: {flow_ids}")

    async def handle_flow_failure(self, process_id: str) -> None:
        """Handle flow failure and notify dependent flows."""
        flow = self._flows.get(process_id)
        if not flow:
            return

        async with self._status_locks[process_id]:
            # Get all dependent flows
            dependent_flows = set(self._flow_graph.successors(process_id))
            
            for dep_id in dependent_flows:
                dep_flow = self._flows.get(dep_id)
                if not dep_flow:
                    continue
                
                # If this was an optional dependency, mark it as skipped
                if process_id in dep_flow.get_dependencies(DependencyType.OPTIONAL):
                    logger.warning(f"Optional dependency {process_id} failed for {dep_id}")
                    continue
                    
                # For required dependencies, fail the dependent flow
                logger.error(f"Required dependency {process_id} failed - failing {dep_id}")
                await self.fail_flow(dep_id, f"Required dependency {process_id} failed")

    async def fail_flow(self, process_id: str, reason: str) -> None:
        """Mark a flow as failed with the given reason."""
        flow = self._flows.get(process_id)
        if not flow:
            return
        from flow.core.flow import FlowResult
        async with self._status_locks[process_id]:
            flow.status = FlowStatus.FAILED
            await self.results_manager.save_result(
                process_id,
                FlowResult(
                    process_id=process_id,
                    status=FlowStatus.FAILED,
                    error=reason,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
            )

    # def cleanup(self) -> None:
    #     """Cleanup all managers and resources."""
    #     self.pool_manager.shutdown()
    #     self.results_manager.cleanup()
    #     self._flows.clear()
    #     self._flow_graph.clear()
    #     self._status_locks.clear()
    #     self._execution_locks.clear()
    #     logger.info("FlowContext cleaned up")

    def cleanup(self) -> None:
        """Cleanup all managers and resources."""
        self.pool_manager.shutdown()
        self.results_manager.cleanup()
        self._flows.clear()
        self._flow_graph.clear()
        self._status_locks.clear()
        self._execution_locks.clear()
        logger.info("FlowContext cleaned up")