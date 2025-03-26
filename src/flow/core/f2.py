"""Fixed Flow implementation that properly abstracts execution details."""
# src/flow/core/flow.py
from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
import uuid
import logging
import traceback
from pydantic import BaseModel

from flow.core.types import FlowType, FlowStatus, DependencyType, LoggingLevel
from flow.core.errors import FlowError
from flow.monitoring.service import MonitoringService
from process_manager.workflow.process import ProcessResult

logger = logging.getLogger(__name__)

class FlowResult(BaseModel):
    """Result of a flow execution."""
    process_id: str
    status: FlowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = {}

class FlowConfig(BaseModel):
    """Configuration for a flow."""
    name: str
    description: Optional[str] = None
    timeout: Optional[float] = None
    retries: int = 0
    flow_type: FlowType = FlowType.PROCESS

class Flow:
    """Core flow implementation that wraps user-defined processors."""
    
    # def __init__(
    #     self,
    #     processor: Any,
    #     config: FlowConfig,
    #     process_id: Optional[str] = None
    # ):
    #     from flow.core.context import FlowContext
        
    #     self.processor = processor
    #     self.config = config
    #     self.process_id = process_id or str(uuid.uuid4())
    #     self.context = FlowContext.get_instance()
        
    #     # Flow state
    #     self.status = FlowStatus.PENDING
    #     self._parent_flow = None
    #     self._dependencies: Dict[str, DependencyType] = {}
    #     self._dependent_flows: Set[str] = set()
        
    #     # Initialize logger
    #     self.logger = logging.getLogger(f"flow.{self.config.name}")
        
    #     # Register with context
    #     self.context.register_flow(self)
        
    #     logger.debug(f"Initialized flow: {self.config.name} ({self.process_id})")

    def register_to(
        self,
        parent_flow: 'Flow',
        required_deps: Optional[List[str]] = None,
        optional_deps: Optional[List[str]] = None
    ) -> None:
        """Register this flow as a child of another flow."""
        self._parent_flow = parent_flow
        
        # Register dependencies
        if required_deps:
            for dep_id in required_deps:
                dep_flow = self.context.get_flow(dep_id)
                if not dep_flow:
                    raise FlowError(f"Required dependency {dep_id} not found")
                self._dependencies[dep_id] = DependencyType.REQUIRED
                dep_flow._dependent_flows.add(self.process_id)
                self.context.register_dependency(dep_id, self.process_id)

        if optional_deps:
            for dep_id in optional_deps:
                dep_flow = self.context.get_flow(dep_id)
                if not dep_flow:
                    raise FlowError(f"Optional dependency {dep_id} not found")
                self._dependencies[dep_id] = DependencyType.OPTIONAL
                dep_flow._dependent_flows.add(self.process_id)
                self.context.register_dependency(dep_id, self.process_id)
    '''
    async def execute(
        self,
        input_data: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Execute the flow and its dependencies.
        
        This is the public interface that users should call.
        """
        monitoring_service = MonitoringService.get_instance()
        
        async with monitoring_service.monitor_flow(self):
            try:
                # Record start
                await monitoring_service.record_flow_event(
                    self,
                    "execution_started",
                    f"Started execution of flow {self.config.name}",
                    LoggingLevel.INFO
                )
                
                # Execute flow
                result = await self._execute_with_dependencies(input_data or {})
                
                # Record completion
                await monitoring_service.record_flow_event(
                    self,
                    "execution_completed",
                    f"Completed execution of flow {self.config.name}",
                    LoggingLevel.INFO,
                    {"status": result.status}
                )
                
                return result
                
            except Exception as e:
                await monitoring_service.record_flow_event(
                    self,
                    "execution_failed",
                    f"Flow {self.config.name} failed: {str(e)}",
                    LoggingLevel.ERROR
                )
                raise

    async def _execute_with_dependencies(
        self,
        input_data: Dict[str, Any]
    ) -> FlowResult:
        """Internal method to handle dependency execution and data passing."""
        # Initialize result
        result = FlowResult(
            process_id=self.process_id,
            status=FlowStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Execute required dependencies
            deps_data = {}
            for dep_id, dep_type in self._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if not dep_flow:
                    if dep_type == DependencyType.REQUIRED:
                        raise FlowError(f"Required dependency {dep_id} not found")
                    continue
                
                try:
                    dep_result = await dep_flow.execute(input_data)
                    if dep_result.status == FlowStatus.COMPLETED:
                        deps_data.update(dep_result.output or {})
                    elif dep_type == DependencyType.REQUIRED:
                        raise FlowError(f"Required dependency {dep_id} failed")
                except Exception as e:
                    if dep_type == DependencyType.REQUIRED:
                        raise
                    logger.warning(f"Optional dependency {dep_id} failed: {e}")

            # Merge dependency data with input data
            execution_data = {**deps_data, **input_data}

            # Execute the processor
            self.status = FlowStatus.RUNNING
            try:
                if hasattr(self.processor, 'process'):
                    output = await self.context.pool_manager.submit_task(
                        self.process_id,
                        self.config.flow_type,
                        self.processor.process,
                        execution_data,
                        timeout=self.config.timeout
                    )
                else:
                    output = await self.context.pool_manager.submit_task(
                        self.process_id,
                        self.config.flow_type,
                        self.processor,
                        execution_data,
                        timeout=self.config.timeout
                    )
                
                result.output = output
                result.status = FlowStatus.COMPLETED
                self.status = FlowStatus.COMPLETED
                
            except Exception as e:
                logger.error(f"Processor execution failed: {e}")
                result.status = FlowStatus.FAILED
                result.error = str(e)
                raise

        except Exception as e:
            result.status = FlowStatus.FAILED
            result.error = str(e)
            self.status = FlowStatus.FAILED
            raise
            
        finally:
            result.end_time = datetime.now()
            await self.context.results_manager.save_result(self.process_id, result)
            
        return result
    '''
    
    def __init__(
        self,
        processor: Any,
        config: FlowConfig,
        process_id: Optional[str] = None
    ):
        from flow.core.context import FlowContext
        
        self.processor = processor
        self.config = config
        self.process_id = process_id or str(uuid.uuid4())
        self.context = FlowContext.get_instance()
        
        # Flow state
        self.status = FlowStatus.PENDING
        self._parent_flow = None
        self._dependencies: Dict[str, DependencyType] = {}
        self._dependent_flows: Set[str] = set()
        
        # Register with context
        self.context.register_flow(self)
        logger.debug(f"Initialized flow: {self.config.name} ({self.process_id})")
    async def execute(
        self,
        input_data: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Execute the flow and its dependencies."""
        logger.debug(f"Starting execution of flow {self.config.name} ({self.process_id})")
        input_data = input_data or {}
        
        # Initialize result first
        result = FlowResult(
            process_id=self.process_id,
            status=FlowStatus.RUNNING,
            start_time=datetime.now(),
            metadata={}
        )
        
        try:
            # Set status before dependency execution
            self.status = FlowStatus.RUNNING
            logger.debug(f"Flow {self.config.name} status set to RUNNING")
            
            # Execute dependencies
            logger.debug(f"Executing dependencies for {self.config.name}")
            deps_data = await self._execute_dependencies(input_data)
            logger.debug(f"Dependencies completed for {self.config.name}")
            
            # Merge input data
            execution_data = {**deps_data, **input_data}
            logger.debug(f"Prepared execution data for {self.config.name}")
            
            # Execute processor
            logger.debug(f"Executing processor for {self.config.name}")
            try:
                # Direct execution without process pool for debugging
                if hasattr(self.processor, 'process'):
                    output = self.processor.process(execution_data)
                else:
                    output = self.processor(execution_data)
                
                # Ensure output is a dictionary
                if not isinstance(output, dict):
                    output = {"result": output}
                
                logger.debug(f"Processor execution completed for {self.config.name}")
                
                # Update result
                result.status = FlowStatus.COMPLETED
                result.output = output
                self.status = FlowStatus.COMPLETED
                
            except Exception as e:
                logger.error(f"Processor execution failed for {self.config.name}: {e}")
                result.status = FlowStatus.FAILED
                result.error = str(e)
                result.traceback = traceback.format_exc()
                self.status = FlowStatus.FAILED
                raise
            
        except Exception as e:
            logger.error(f"Flow execution failed for {self.config.name}: {e}")
            result.status = FlowStatus.FAILED
            result.error = str(e)
            result.traceback = traceback.format_exc()
            self.status = FlowStatus.FAILED
            raise
            
        finally:
            # Always save result and update timing
            result.end_time = datetime.now()
            logger.debug(f"Saving result for {self.config.name}")
            await self.context.results_manager.save_result(
                self.process_id,
                result
            )
            logger.debug(f"Result saved for {self.config.name}")
        
        logger.debug(f"Flow {self.config.name} execution completed with status {result.status}")
        return result

    async def _execute_dependencies(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute dependencies and collect their outputs."""
        deps_data = {}
        
        # Execute required dependencies first
        required_deps = self.get_dependencies(DependencyType.REQUIRED)
        if required_deps:
            logger.debug(f"Executing required dependencies for {self.config.name}: {required_deps}")
            for dep_id in required_deps:
                dep_flow = self.context.get_flow(dep_id)
                if not dep_flow:
                    raise FlowError(f"Required dependency {dep_id} not found")
                
                logger.debug(f"Executing required dependency {dep_flow.config.name}")
                dep_result = await dep_flow.execute(input_data)
                
                if dep_result.status != FlowStatus.COMPLETED:
                    raise FlowError(
                        f"Required dependency {dep_id} failed: {dep_result.error}"
                    )
                deps_data.update(dep_result.output or {})
                logger.debug(f"Required dependency {dep_flow.config.name} completed")

        # Execute optional dependencies
        optional_deps = self.get_dependencies(DependencyType.OPTIONAL)
        if optional_deps:
            logger.debug(f"Executing optional dependencies for {self.config.name}: {optional_deps}")
            for dep_id in optional_deps:
                try:
                    dep_flow = self.context.get_flow(dep_id)
                    if dep_flow:
                        logger.debug(f"Executing optional dependency {dep_flow.config.name}")
                        dep_result = await dep_flow.execute(input_data)
                        if dep_result.status == FlowStatus.COMPLETED:
                            deps_data.update(dep_result.output or {})
                            logger.debug(f"Optional dependency {dep_flow.config.name} completed")
                except Exception as e:
                    logger.warning(f"Optional dependency {dep_id} failed: {e}")

        return deps_data
    '''
    async def execute(
        self,
        input_data: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Execute the flow and its dependencies."""
        input_data = input_data or {}
        monitoring_service = MonitoringService.get_instance()
        
        async with monitoring_service.monitor_flow(self):
            try:
                await monitoring_service.record_flow_event(
                    self,
                    "execution_started",
                    f"Started execution of flow {self.config.name}",
                    LoggingLevel.INFO
                )
                
                # Initialize result
                result = FlowResult(
                    process_id=self.process_id,
                    status=FlowStatus.RUNNING,
                    start_time=datetime.now(),
                    metadata={}
                )
                
                try:
                    # Execute dependencies first
                    deps_data = await self._execute_dependencies(input_data)
                    
                    # Merge dependency data with input data
                    execution_data = {**deps_data, **input_data}
                    
                    # Execute the processor
                    self.status = FlowStatus.RUNNING
                    output = await self._execute_processor(execution_data)
                    
                    # Update result
                    result.status = FlowStatus.COMPLETED
                    result.output = output
                    self.status = FlowStatus.COMPLETED
                    
                except Exception as e:
                    result.status = FlowStatus.FAILED
                    result.error = str(e)
                    result.traceback = traceback.format_exc()
                    self.status = FlowStatus.FAILED
                    raise
                
                finally:
                    result.end_time = datetime.now()
                    await self.context.results_manager.save_result(
                        self.process_id,
                        result
                    )
                
                await monitoring_service.record_flow_event(
                    self,
                    "execution_completed",
                    f"Completed execution of flow {self.config.name}",
                    LoggingLevel.INFO,
                    {"status": result.status}
                )
                
                return result
                
            except Exception as e:
                await monitoring_service.record_flow_event(
                    self,
                    "execution_failed",
                    f"Flow {self.config.name} failed: {str(e)}",
                    LoggingLevel.ERROR
                )
                raise

    async def _execute_dependencies(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute dependencies and collect their outputs."""
        deps_data = {}
        
        # Execute required dependencies
        for dep_id in self.get_dependencies(DependencyType.REQUIRED):
            dep_flow = self.context.get_flow(dep_id)
            if not dep_flow:
                raise FlowError(f"Required dependency {dep_id} not found")
            
            dep_result = await dep_flow.execute(input_data)
            if dep_result.status != FlowStatus.COMPLETED:
                raise FlowError(
                    f"Required dependency {dep_id} failed: {dep_result.error}"
                )
            deps_data.update(dep_result.output or {})

        # Execute optional dependencies
        for dep_id in self.get_dependencies(DependencyType.OPTIONAL):
            try:
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    dep_result = await dep_flow.execute(input_data)
                    if dep_result.status == FlowStatus.COMPLETED:
                        deps_data.update(dep_result.output or {})
            except Exception as e:
                logger.warning(f"Optional dependency {dep_id} failed: {e}")

        return deps_data
    '''
    async def _execute_processor(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the processor and return its output."""
        try:
            # Execute processor based on flow type
            if self.config.flow_type in (FlowType.PROCESS, FlowType.THREAD):
                output = await self.context.pool_manager.submit_task(
                    self.process_id,
                    self.config.flow_type,
                    self.processor.process,
                    input_data,
                    timeout=self.config.timeout
                )
            else:
                # Execute inline
                output = self.processor.process(input_data)

            # Ensure output is a dictionary
            if not isinstance(output, dict):
                output = {"result": output}
                
            return output
                
        except Exception as e:
            logger.error(f"Processor execution failed: {e}", exc_info=True)
            raise

    async def cancel(self) -> None:
        """Cancel the flow execution."""
        if self.status == FlowStatus.RUNNING:
            await self.context.pool_manager.cancel_task(self.process_id)
            self.status = FlowStatus.CANCELLED
# """Core Flow implementation for process management."""
# from __future__ import annotations
# import asyncio
# from datetime import datetime
# from typing import Dict, Any, Optional, List, Set
# import uuid
# import logging
# import traceback
# from enum import Enum
# from pydantic import BaseModel, Field

# from flow.core.logging import FlowLogger
# from flow.core.types import FlowType, FlowStatus, StorageType, VisFormat, DependencyType
# from flow.core.errors import FlowError, FlowTimeoutError
# from flow.core.context import FlowContext
# from flow.core.results import ResultsManager
# from flow.monitoring.service import MonitoringService
# from flow.core.types import LoggingLevel
# from flow.monitoring.types import FlowEvent, FlowStatistics, HealthStatus
# from flow.visualization.graph import FlowVisualizer

# logger = logging.getLogger(__name__)

# class FlowConfig(BaseModel):
#     """Configuration for a flow."""
#     name: str
#     description: Optional[str] = None
#     timeout: Optional[float] = None
#     retries: int = 0
#     flow_type: FlowType = FlowType.PROCESS
    
#     class Config:
#         frozen = True

# class FlowResult(BaseModel):
#     """Result of a flow execution."""
#     process_id: str
#     status: FlowStatus  # This ensures we use the enum
#     start_time: datetime
#     end_time: Optional[datetime] = None
#     output: Optional[Dict[str, Any]] = None
#     error: Optional[str] = None
#     traceback: Optional[str] = None

#     class Config:
#         use_enum_values = False  # This ensures we store the actual enum

# class DependencyData:
#     """Interface for accessing dependency data."""
#     def __init__(self, results_manager: ResultsManager, process_id: str):
#         self._results_manager = results_manager
#         self._process_id = process_id

#     async def get(self, dep_id: str) -> Any:
#         """Get data from a specific dependency."""
#         return await self._results_manager.get_dependency_data(
#             self._process_id, 
#             dep_id
#         )

#     async def get_all(self) -> Dict[str, Any]:
#         """Get all dependency data."""
#         return await self._results_manager.get_dependencies_data(
#             self._process_id
#         )

# class Flow:
#     """Core flow implementation that wraps user-defined processors."""
    
#     def __init__(
#         self,
#         processor: Any,  # User-defined processor
#         config: FlowConfig,
#         process_id: Optional[str] = None
#     ):
#         self.processor = processor
#         self.config = config
#         self.process_id = process_id or str(uuid.uuid4())
#         self.context = FlowContext.get_instance()
        
#         # Flow state
#         self.status = FlowStatus.PENDING
#         self._parent_flow = None
#         self._dependencies: Dict[str, DependencyType] = {}
#         self._dependent_flows: Set[str] = set()
        
        
#         # Initialize logger with flow context
#         self.logger = FlowLogger("flow.core")
        
#         # Register with context
#         self.context.register_flow(self)
        
#         self.logger.info(
#             "Initialized flow",
#             extra={
#                 'flow_context': {
#                     'config': self.config.model_dump(),
#                     'processor_type': type(self.processor).__name__
#                 }
#             }
#         )
#         self.status: FlowStatus = FlowStatus.PENDING  # Explicitly type and use enum
        
#         logger.debug(f"Initialized flow: {self.config.name} ({self.process_id})")
#     '''
#     def register_to(
#         self,
#         parent_flow: 'Flow',
#         required_deps: Optional[List[str]] = None,
#         optional_deps: Optional[List[str]] = None
#     ) -> None:
#         """Register this flow as a child of another flow with specified dependencies."""
#         self._parent_flow = parent_flow
        
#         # Register dependencies
#         if required_deps:
#             for dep_id in required_deps:
#                 dep_flow = self.context.get_flow(dep_id)
#                 if not dep_flow:
#                     raise FlowError(f"Required dependency flow {dep_id} not found")
#                 self._dependencies[dep_id] = DependencyType.REQUIRED
#                 dep_flow._dependent_flows.add(self.process_id)

#         if optional_deps:
#             for dep_id in optional_deps:
#                 dep_flow = self.context.get_flow(dep_id)
#                 if not dep_flow:
#                     raise FlowError(f"Optional dependency flow {dep_id} not found")
#                 self._dependencies[dep_id] = DependencyType.OPTIONAL
#                 dep_flow._dependent_flows.add(self.process_id)

#         # Validate for cycles
#         if self.context.has_cycle(self.process_id):
#             raise FlowError("Registration would create a circular dependency")

#         logger.info(f"Registered flow {self.process_id} to {parent_flow.process_id}")
#     '''
#     def register_to(
#         self,
#         parent_flow: Flow,
#         required_deps: Optional[List[str]] = None,
#         optional_deps: Optional[List[str]] = None
#     ) -> None:
#         """Register this flow as a child of another flow with specified dependencies.
        
#         Args:
#             parent_flow: The parent flow
#             required_deps: List of required dependency process IDs
#             optional_deps: List of optional dependency process IDs
#         """
#         self._parent_flow = parent_flow
        
#         # Register dependencies in both directions
#         if required_deps:
#             for dep_id in required_deps:
#                 dep_flow = self.context.get_flow(dep_id)
#                 if not dep_flow:
#                     raise FlowError(f"Required dependency flow {dep_id} not found")
#                 self._dependencies[dep_id] = DependencyType.REQUIRED
#                 dep_flow._dependent_flows.add(self.process_id)
                
#                 # Register in context graph
#                 self.context.register_dependency(dep_id, self.process_id)

#         if optional_deps:
#             for dep_id in optional_deps:
#                 dep_flow = self.context.get_flow(dep_id)
#                 if not dep_flow:
#                     raise FlowError(f"Optional dependency flow {dep_id} not found")
#                 self._dependencies[dep_id] = DependencyType.OPTIONAL
#                 dep_flow._dependent_flows.add(self.process_id)
                
#                 # Register in context graph
#                 self.context.register_dependency(dep_id, self.process_id)

#         # Validate for cycles
#         if self.context.has_cycle(self.process_id):
#             # Cleanup registrations before raising
#             for dep_id in self._dependencies:
#                 dep_flow = self.context.get_flow(dep_id)
#                 if dep_flow:
#                     dep_flow._dependent_flows.remove(self.process_id)
#                 self.context._flow_graph.remove_edge(dep_id, self.process_id)
#             raise FlowError("Registration would create a circular dependency")

#         logger.info(f"Registered flow {self.process_id} to {parent_flow.process_id}")
    def get_dependencies(self, dep_type: Optional[DependencyType] = None) -> Set[str]:
        """Get set of dependency process IDs, optionally filtered by type."""
        if dep_type is None:
            return set(self._dependencies.keys())
        return {
            dep_id for dep_id, dtype in self._dependencies.items() 
            if dtype == dep_type
        }

#     # async def execute(self, input_data: Optional[Dict[str, Any]] = None) -> FlowResult:
#     #     """Execute the flow with dependency handling."""
#     #     result = FlowResult(
#     #         process_id=self.process_id,
#     #         status=FlowStatus.RUNNING,
#     #         start_time=datetime.now()
#     #     )
        
#     #     try:
#     #         # Set running status
#     #         self.status = FlowStatus.RUNNING
            
#     #         # Create dependency interface
#     #         deps = DependencyData(self.context.results_manager, self.process_id)
            
#     #         # Wait for required dependencies
#     #         required_deps = self.get_dependencies(DependencyType.REQUIRED)
#     #         if required_deps:
#     #             await self.context.wait_for_flows(required_deps, self.config.timeout)
            
#     #         # Execute with retries
#     #         for attempt in range(self.config.retries + 1):
#     #             try:
#     #                 # Merge dependency data with input data
#     #                 execution_data = {
#     #                     **(await deps.get_all()),
#     #                     **(input_data or {})
#     #                 }
                    
#     #                 # Submit to appropriate executor
#     #                 if hasattr(self.processor, 'process'):
#     #                     output = await self.context.pool_manager.submit_task(
#     #                         self.process_id,
#     #                         self.config.flow_type,
#     #                         self.processor.process,
#     #                         execution_data,
#     #                         timeout=self.config.timeout
#     #                     )
#     #                 else:
#     #                     # Assume processor is callable
#     #                     output = await self.context.pool_manager.submit_task(
#     #                         self.process_id,
#     #                         self.config.flow_type,
#     #                         self.processor,
#     #                         execution_data,
#     #                         timeout=self.config.timeout
#     #                     )
                    
#     #                 result.output = output
#     #                 result.status = FlowStatus.COMPLETED
#     #                 self.status = FlowStatus.COMPLETED
#     #                 break
                    
#     #             except Exception as e:
#     #                 if attempt == self.config.retries:
#     #                     raise
#     #                 logger.warning(
#     #                     f"Attempt {attempt + 1}/{self.config.retries + 1} failed for {self.process_id}: {str(e)}"
#     #                 )
#     #                 await asyncio.sleep(1)  # Basic retry delay
                    
#     #     except Exception as e:
#     #         result.status = FlowStatus.FAILED
#     #         result.error = str(e)
#     #         result.traceback = traceback.format_exc()
#     #         self.status = FlowStatus.FAILED
            
#     #         logger.error(f"Flow {self.process_id} failed: {str(e)}")
            
#     #         # Handle failure propagation
#     #         await self.context.handle_flow_failure(self.process_id)
            
#     #         # Raise if this is a required dependency for others
#     #         if self._dependent_flows and not all(
#     #             self._dependencies.get(dep_id) == DependencyType.OPTIONAL
#     #             for dep_id in self._dependent_flows
#     #         ):
#     #             raise
                
#     #     finally:
#     #         result.end_time = datetime.now()
#     #         await self.context.results_manager.save_result(self.process_id, result)
            
#     #     return result

#     async def cancel(self) -> None:
#         """Cancel the flow execution."""
#         if self.status == FlowStatus.RUNNING:
#             await self.context.pool_manager.cancel_task(self.process_id)
#             self.status = FlowStatus.CANCELLED
            
#             # Save cancelled status
#             result = FlowResult(
#                 process_id=self.process_id,
#                 status=FlowStatus.CANCELLED,
#                 start_time=datetime.now(),
#                 end_time=datetime.now(),
#                 error="Flow cancelled"
#             )
#             await self.context.results_manager.save_result(self.process_id, result)

#     def __str__(self) -> str:
#         return f"Flow(name='{self.config.name}', id={self.process_id}, status={self.status})"

#     def __repr__(self) -> str:
#         return self.__str__()
#     '''
#     async def execute(
#         self,
#         input_data: Optional[Dict[str, Any]] = None,
#         fail_quickly: bool = False
#     ) -> FlowResult:
#         """Execute the flow and its dependencies.
        
#         Args:
#             input_data: Input data for this flow
#             fail_quickly: If True, fails immediately on any error
            
#         Returns:
#             FlowResult containing execution status and output
#         """
#         result = FlowResult(
#             process_id=self.process_id,
#             status=FlowStatus.RUNNING,
#             start_time=datetime.now()
#         )
        
#         try:
#             # 1. Execute required dependencies first
#             deps_data = {}
#             required_deps = self.get_dependencies(DependencyType.REQUIRED)
            
#             if required_deps:
#                 logger.info(f"Executing required dependencies for {self.config.name}")
#                 for dep_id in required_deps:
#                     dep_flow = self.context.get_flow(dep_id)
#                     if not dep_flow:
#                         raise FlowError(f"Required dependency {dep_id} not found")
                    
#                     # Execute dependency if not already executed
#                     if dep_flow.status == FlowStatus.PENDING:
#                         dep_result = await dep_flow.execute(input_data, fail_quickly)
#                         if dep_result.status != FlowStatus.COMPLETED:
#                             raise FlowError(
#                                 f"Required dependency {dep_id} failed with status {dep_result.status}"
#                             )
                    
#                     # Get dependency output
#                     dep_result = await self.context.results_manager.get_result(dep_id)
#                     if not dep_result or not dep_result.output:
#                         raise FlowError(f"No output from required dependency {dep_id}")
#                     deps_data[dep_id] = dep_result.output

#             # 2. Execute optional dependencies
#             optional_deps = self.get_dependencies(DependencyType.OPTIONAL)
#             if optional_deps:
#                 logger.info(f"Executing optional dependencies for {self.config.name}")
#                 for dep_id in optional_deps:
#                     try:
#                         dep_flow = self.context.get_flow(dep_id)
#                         if not dep_flow:
#                             logger.warning(f"Optional dependency {dep_id} not found")
#                             continue
                            
#                         if dep_flow.status == FlowStatus.PENDING:
#                             dep_result = await dep_flow.execute(input_data, fail_quickly)
#                             if dep_result.status == FlowStatus.COMPLETED:
#                                 dep_result = await self.context.results_manager.get_result(dep_id)
#                                 if dep_result and dep_result.output:
#                                     deps_data[dep_id] = dep_result.output
#                     except Exception as e:
#                         logger.warning(f"Optional dependency {dep_id} failed: {e}")

#             # 3. Prepare execution data
#             execution_data = {
#                 **deps_data,  # Dependency outputs
#                 **(input_data or {})  # Direct inputs (override deps)
#             }

#             # 4. Execute this flow's processor
#             logger.info(f"Executing flow {self.config.name}")
#             self.status = FlowStatus.RUNNING
            
#             try:
#                 if hasattr(self.processor, 'process'):
#                     output = await self.context.pool_manager.submit_task(
#                         self.process_id,
#                         self.config.flow_type,
#                         self.processor.process,
#                         execution_data,
#                         timeout=self.config.timeout
#                     )
#                 else:
#                     output = await self.context.pool_manager.submit_task(
#                         self.process_id,
#                         self.config.flow_type,
#                         self.processor,
#                         execution_data,
#                         timeout=self.config.timeout
#                     )
                
#                 result.output = output
#                 result.status = FlowStatus.COMPLETED
#                 self.status = FlowStatus.COMPLETED
                
#             except Exception as e:
#                 if fail_quickly:
#                     raise
#                 logger.error(f"Flow {self.config.name} execution failed: {e}")
#                 result.status = FlowStatus.FAILED
#                 result.error = str(e)
#                 self.status = FlowStatus.FAILED
                
#                 # Only raise if this is a required dependency for others
#                 if self._dependent_flows and any(
#                     flow._dependencies.get(self.process_id) == DependencyType.REQUIRED
#                     for flow in [self.context.get_flow(fid) for fid in self._dependent_flows]
#                 ):
#                     raise

#         except Exception as e:
#             result.status = FlowStatus.FAILED
#             result.error = str(e)
#             self.status = FlowStatus.FAILED
#             raise
            
#         finally:
#             result.end_time = datetime.now()
#             await self.context.results_manager.save_result(self.process_id, result)
            
#         return result
#     '''


#     async def execute(
#         self,
#         input_data: Optional[Dict[str, Any]] = None,
#         fail_quickly: bool = False
#     ) -> FlowResult:
#         """Execute the flow with monitoring.
        
#         This is the public interface that handles monitoring and high-level flow control.
#         The actual execution logic is delegated to _execute_flow.
#         """
#         monitoring_service = MonitoringService.get_instance()
        
#         async with monitoring_service.monitor_flow(self):
#             try:
#                 await monitoring_service.record_flow_event(
#                     self,
#                     "execution_started",
#                     f"Started execution of flow {self.config.name}",
#                     LoggingLevel.INFO,
#                     {"fail_quickly": fail_quickly}
#                 )
                
#                 result = await self._execute_flow(input_data, fail_quickly)
                
#                 await monitoring_service.record_flow_event(
#                     self,
#                     "execution_completed",
#                     f"Completed execution of flow {self.config.name}",
#                     LoggingLevel.INFO,
#                     {
#                         "output_size": len(str(result.output)) if result.output else 0,
#                         "status": result.status.value
#                     }
#                 )
                
#                 return result
                
#             except Exception as e:
#                 await monitoring_service.record_flow_event(
#                     self,
#                     "execution_failed",
#                     f"Flow {self.config.name} failed: {str(e)}",
#                     LoggingLevel.ERROR,
#                     {
#                         "error": str(e),
#                         "traceback": traceback.format_exc(),
#                         "fail_quickly": fail_quickly
#                     }
#                 )
#                 raise

#     async def _execute_flow(
#         self,
#         input_data: Optional[Dict[str, Any]] = None,
#         fail_quickly: bool = False
#     ) -> FlowResult:
#         """Internal flow execution method that handles the actual execution logic."""
#         with self.logger.flow_context(flow_name=self.config.name, process_id=self.process_id):
#             result = FlowResult(
#                 process_id=self.process_id,
#                 status=FlowStatus.RUNNING,
#                 start_time=datetime.now()
#             )
            
#             try:
#                 # 1. Execute dependencies first
#                 deps_data = await self._execute_dependencies(input_data, fail_quickly)

#                 # 2. Prepare execution data
#                 execution_data = {
#                     **deps_data,  # Dependency outputs
#                     **(input_data or {})  # Direct inputs (override deps)
#                 }

#                 # 3. Execute this flow's processor
#                 self.status = FlowStatus.RUNNING
#                 self.logger.info("Executing processor", extra={
#                     'flow_context': {
#                         'input_keys': list(execution_data.keys())
#                     }
#                 })
                
#                 output = await self._execute_processor(execution_data)
                
#                 result.output = output
#                 result.status = FlowStatus.COMPLETED
#                 self.status = FlowStatus.COMPLETED
                
#                 self.logger.info("Execution completed successfully", extra={
#                     'flow_context': {
#                         'output_size': len(str(output)) if output else 0
#                     }
#                 })
                
#             except Exception as e:
#                 self.logger.error("Flow execution failed", extra={
#                     'flow_context': {
#                         'error': str(e),
#                         'traceback': traceback.format_exc()
#                     }
#                 })
#                 result.status = FlowStatus.FAILED
#                 result.error = str(e)
#                 result.traceback = traceback.format_exc()
#                 self.status = FlowStatus.FAILED
                
#                 if fail_quickly or self._has_required_dependents():
#                     raise
                    
#             finally:
#                 result.end_time = datetime.now()
#                 await self.context.results_manager.save_result(self.process_id, result)
                
#             return result

#     async def _execute_dependencies(
#         self,
#         input_data: Optional[Dict[str, Any]],
#         fail_quickly: bool
#     ) -> Dict[str, Any]:
#         """Execute dependencies and collect their outputs."""
#         deps_data = {}
        
#         # Execute required dependencies
#         for dep_id in self.get_dependencies(DependencyType.REQUIRED):
#             dep_flow = self.context.get_flow(dep_id)
#             if not dep_flow:
#                 raise FlowError(f"Required dependency {dep_id} not found")
                
#             dep_result = await dep_flow.execute(input_data, fail_quickly)
#             if dep_result.status != FlowStatus.COMPLETED:
#                 raise FlowError(f"Required dependency {dep_id} failed")
#             deps_data.update(dep_result.output or {})

#         # Execute optional dependencies
#         for dep_id in self.get_dependencies(DependencyType.OPTIONAL):
#             try:
#                 dep_flow = self.context.get_flow(dep_id)
#                 if dep_flow:
#                     dep_result = await dep_flow.execute(input_data, fail_quickly)
#                     if dep_result.status == FlowStatus.COMPLETED:
#                         deps_data.update(dep_result.output or {})
#             except Exception as e:
#                 self.logger.warning(f"Optional dependency {dep_id} failed: {e}")

#         return deps_data

#     async def _execute_processor(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute the flow's processor with the given data."""
#         try:
#             if hasattr(self.processor, 'process'):
#                 return await self.context.pool_manager.submit_task(
#                     self.process_id,
#                     self.config.flow_type,
#                     self.processor.process,
#                     execution_data,
#                     timeout=self.config.timeout
#                 )
#             else:
#                 return await self.context.pool_manager.submit_task(
#                     self.process_id,
#                     self.config.flow_type,
#                     self.processor,
#                     execution_data,
#                     timeout=self.config.timeout
#                 )
#         except Exception as e:
#             self.logger.error("Processor execution failed", extra={
#                 'flow_context': {
#                     'error': str(e),
#                     'input_keys': list(execution_data.keys())
#                 }
#             })
#             raise

#     def _has_required_dependents(self) -> bool:
#         """Check if this flow is a required dependency for any other flows."""
#         return any(
#             flow._dependencies.get(self.process_id) == DependencyType.REQUIRED
#             for flow in [self.context.get_flow(fid) for fid in self._dependent_flows]
#         )
    
#     async def cleanup_flows(self) -> None:
#         """Clean up all flows and their resources."""
#         try:
#             # Cancel any running flows
#             for flow in self._flows.values():
#                 if flow.status == FlowStatus.RUNNING:
#                     try:
#                         await flow.cancel()
#                     except Exception as e:
#                         logger.error(f"Error cancelling flow {flow.config.name}: {e}")

#             # Clean up pools
#             self.pool_manager.shutdown(wait=True)
            
#             # Clean up results
#             self.results_manager.cleanup()
            
#             # Clear flow tracking
#             self._flows.clear()
#             self._flow_graph.clear()
#             self._status_locks.clear()
#             self._execution_locks.clear()
            
#         except Exception as e:
#             logger.error(f"Error during flow cleanup: {e}")
#         finally:
#             logger.info("Flow cleanup completed")
#     '''
#     async def execute(
#         self,
#         input_data: Optional[Dict[str, Any]] = None,
#         fail_quickly: bool = False
#     ) -> FlowResult:
#         """Execute the flow with monitoring.
        
#         Args:
#             input_data: Input data for the flow
#             fail_quickly: If True, fails immediately on any error without retries
#         """
#         monitoring_service = MonitoringService.get_instance()
        
#         async with monitoring_service.monitor_flow(self):
#             try:
#                 await monitoring_service.record_flow_event(
#                     self,
#                     "execution_started",
#                     f"Started execution of flow {self.config.name}",
#                     LoggingLevel.INFO,
#                     {"fail_quickly": fail_quickly}
#                 )
                
#                 result = await self._execute_flow(input_data, fail_quickly)
                
#                 await monitoring_service.record_flow_event(
#                     self,
#                     "execution_completed",
#                     f"Completed execution of flow {self.config.name}",
#                     LoggingLevel.INFO,
#                     {
#                         "output_size": len(str(result.output)),
#                         "status": result.status.value
#                     }
#                 )
                
#                 return result
                
#             except Exception as e:
#                 await monitoring_service.record_flow_event(
#                     self,
#                     "execution_failed",
#                     f"Flow {self.config.name} failed: {str(e)}",
#                     LoggingLevel.ERROR,
#                     {
#                         "error": str(e),
#                         "traceback": traceback.format_exc(),
#                         "fail_quickly": fail_quickly
#                     }
#                 )
#                 raise
        
#     async def _execute_flow(self, input_data: Optional[Dict[str, Any]] = None, fail_quickly: bool = False) -> FlowResult:
#         """Internal flow execution method.
        
#         Args:
#             input_data: Input data for the flow
#             fail_quickly: If True, fails immediately on any error without retries
#         """
#         with self.logger.flow_context(flow_name=self.config.name, process_id=self.process_id):
#             result = FlowResult(
#                 process_id=self.process_id,
#                 status=FlowStatus.RUNNING,
#                 start_time=datetime.now()
#             )
            
#             try:
#                 self.status = FlowStatus.RUNNING
#                 self.logger.info("Starting flow execution", extra={
#                     'flow_context': {
#                         'input_data_keys': list(input_data.keys()) if input_data else None,
#                         'fail_quickly': fail_quickly
#                     }
#                 })
#                 deps = DependencyData(self.context.results_manager, self.process_id)
                
#                 # Wait for required dependencies
#                 required_deps = self.get_dependencies(DependencyType.REQUIRED)
#                 if required_deps:
#                     self.logger.info("Waiting for required dependencies", extra={
#                         'flow_context': {'dependencies': list(required_deps)}
#                     })
#                     try:
#                         await self.context.wait_for_flows(required_deps, self.config.timeout)
#                     except Exception as e:
#                         self.logger.error("Required dependency failed", extra={
#                             'flow_context': {
#                                 'error': str(e),
#                                 'dependencies': list(required_deps)
#                             }
#                         })
#                         # If a required dependency failed, we must fail
#                         raise FlowError(f"Required dependency failed: {str(e)}")

#                 # Check optional dependencies - continue if they fail
#                 optional_deps = self.get_dependencies(DependencyType.OPTIONAL)
#                 if optional_deps:
#                     try:
#                         await self.context.wait_for_flows(optional_deps, self.config.timeout)
#                     except Exception as e:
#                         logger.warning(f"Optional dependency failed for {self.process_id}: {str(e)}")
#                         # Continue execution - optional dependency failure is not fatal

#                 # Execute with retries unless fail_quickly is True
#                 max_attempts = 1 if fail_quickly else (self.config.retries + 1)
#                 for attempt in range(max_attempts):
#                     try:
#                         self.logger.info(f"Execution attempt {attempt + 1}/{self.config.retries + 1}")
                        
#                         # Get all available dependency data - skip failed optional deps
#                         deps_data = {}
#                         for dep_id in self._dependencies:
#                             try:
#                                 deps_data[dep_id] = await deps.get(dep_id)
#                             except Exception as e:
#                                 if self._dependencies[dep_id] == DependencyType.REQUIRED:
#                                     raise
#                                 logger.warning(f"Skipping failed optional dependency {dep_id}: {str(e)}")

#                         # Merge with input data
#                         execution_data = {
#                             **deps_data,
#                             **(input_data or {})
#                         }
                        
#                         # Submit to appropriate executor
#                         if hasattr(self.processor, 'process'):
#                             output = await self.context.pool_manager.submit_task(
#                                 self.process_id,
#                                 self.config.flow_type,
#                                 self.processor.process,
#                                 execution_data,
#                                 timeout=self.config.timeout
#                             )
#                         else:
#                             output = await self.context.pool_manager.submit_task(
#                                 self.process_id,
#                                 self.config.flow_type,
#                                 self.processor,
#                                 execution_data,
#                                 timeout=self.config.timeout
#                             )
                        
#                         result.output = output
#                         result.status = FlowStatus.COMPLETED
#                         self.status = FlowStatus.COMPLETED
#                         self.logger.info("Execution completed successfully", extra={
#                             'flow_context': {
#                                 'attempt': attempt + 1,
#                                 'output_size': len(str(result.output))
#                             }
#                         })
#                         break
                        
#                     except Exception as e:
#                         if fail_quickly or attempt == max_attempts - 1:
#                             raise
#                         self.logger.warning(f"Attempt failed, retrying", extra={
#                             'flow_context': {
#                                 'attempt': attempt + 1,
#                                 'error': str(e),
#                                 'retry_delay': 1
#                             }
#                         })
#                         await asyncio.sleep(1)  # Basic retry delay
                        
#             except Exception as e:
#                 self.logger.error("Flow execution failed", extra={
#                     'flow_context': {
#                         'error': str(e),
#                         'traceback': traceback.format_exc()
#                     }
#                 })
#                 result.status = FlowStatus.FAILED
#                 result.error = str(e)
#                 result.traceback = traceback.format_exc()
#                 self.status = FlowStatus.FAILED
                
#                 logger.error(f"Flow {self.process_id} failed: {str(e)}")
                
#                 # Handle failure propagation
#                 await self.context.handle_flow_failure(self.process_id)
                
#                 # Only raise if this is a required dependency for others
#                 if self._dependent_flows and any(
#                     flow._dependencies.get(self.process_id) == DependencyType.REQUIRED
#                     for flow in [self.context.get_flow(fid) for fid in self._dependent_flows]
#                 ):
#                     raise
                    
#             finally:
#                 result.end_time = datetime.now()
#                 await self.context.results_manager.save_result(self.process_id, result)
#                 execution_time = (result.end_time - result.start_time).total_seconds()
#                 self.logger.info("Flow execution finished", extra={
#                     'flow_context': {
#                         'status': result.status.value,
#                         'execution_time': execution_time
#                     }
#                 })
#             return result
# '''
#     async def get_health(self) -> HealthStatus:
#         """Get flow health status."""
#         return await MonitoringService.get_instance().get_flow_health(self)

#     async def get_statistics(self) -> FlowStatistics:
#         """Get flow execution statistics."""
#         return await MonitoringService.get_instance().get_flow_statistics(self)

#     async def get_recent_events(
#         self,
#         limit: int = 100,
#         level: Optional[LoggingLevel] = None
#     ) -> List[FlowEvent]:
#         """Get recent events for this flow."""
#         return await MonitoringService.get_instance().get_recent_events(
#             self, limit, level
#         )
    
#     def get_visualizer(self) -> FlowVisualizer:
#         """Get a visualizer for this flow."""
#         return FlowVisualizer(self)
    
#     def visualize(
#         self,
#         format: VisFormat = VisFormat.MERMAID,
#         output_path: Optional[str] = None
#     ) -> Any:
#         """Visualize the flow graph.
        
#         Args:
#             format: Visualization format (VisFormat enum)
#             output_path: Optional path to save the visualization
            
#         Returns:
#             The visualization result, format depends on the chosen format:
#             - MERMAID: str (mermaid markdown)
#             - GRAPHVIZ: str (path to saved file) or dot string
#             - PLOTLY: str (path to saved file) or Figure object
            
#         Raises:
#             ValueError: If format is not supported
#         """
#         visualizer = self.get_visualizer()
        
#         match format:
#             case VisFormat.MERMAID:
#                 return visualizer.to_mermaid()
#             case VisFormat.GRAPHVIZ:
#                 return visualizer.to_graphviz(output_path)
#             case VisFormat.PLOTLY:
#                 return visualizer.to_plotly(output_path)
#             case _:
#                 raise ValueError(f"Unsupported visualization format: {format}")