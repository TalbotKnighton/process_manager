"""Monitoring service for flow execution."""
from __future__ import annotations
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from flow.core.types import DependencyType, LoggingLevel
from flow.monitoring.metrics import FlowMonitor
from flow.monitoring.types import (
    FlowMetrics, FlowEvent, ResourceMetrics,
    HealthStatus, FlowStatistics
)
from flow.core.types import FlowStatus
if TYPE_CHECKING:
    from flow.core.flow import Flow

logger = logging.getLogger(__name__)

# class MonitoringService:
#     """Service for monitoring and managing flow metrics."""
#     _instance = None

#     def __init__(self):
#         if MonitoringService._instance is not None:
#             raise RuntimeError("MonitoringService is a singleton - use get_instance()")
        
#         self.monitor = FlowMonitor()
#         self._background_task: Optional[asyncio.Task] = None
#         self._stopping = False
        
#         # Initialize background monitoring
#         self._start_background_monitoring()

#     @classmethod
#     def get_instance(cls) -> MonitoringService:
#         """Get or create the singleton instance."""
#         if cls._instance is None:
#             cls._instance = cls()
#         return cls._instance

#     async def _monitor_loop(self) -> None:
#         """Background monitoring loop."""
#         while not self._stopping:
#             try:
#                 # Collect system metrics
#                 metrics = await self.monitor.get_resource_metrics()
                
#                 # Check for resource warnings
#                 if metrics.cpu_percent > 80:
#                     await self.monitor.record_event(
#                         "system",
#                         "high_cpu_usage",
#                         f"CPU usage is high: {metrics.cpu_percent}%",
#                         LoggingLevel.WARNING
#                     )
                
#                 if metrics.memory_percent > 80:
#                     await self.monitor.record_event(
#                         "system",
#                         "high_memory_usage",
#                         f"Memory usage is high: {metrics.memory_percent}%",
#                         LoggingLevel.WARNING
#                     )
                
#                 await asyncio.sleep(60)  # Monitor every minute
                
#             except Exception as e:
#                 logger.error(f"Error in monitoring loop: {e}")
#                 await asyncio.sleep(5)  # Back off on error

#     @asynccontextmanager
#     async def monitor_flow(self, flow: Flow) -> None:
#         """Context manager for monitoring a flow execution."""
#         try:
#             # Start monitoring
#             await self.monitor.start_monitoring(flow.process_id)
#             yield
#         finally:
#             # Stop monitoring and collect final metrics
#             await self.monitor.stop_monitoring(flow.process_id)
class MonitoringService:
    """Service for monitoring and managing flow metrics."""
    _instance = None

    def __init__(self):
        if MonitoringService._instance is not None:
            raise RuntimeError("MonitoringService is a singleton - use get_instance()")
        
        self.monitor = FlowMonitor()
        self._background_task: Optional[asyncio.Task] = None
        self._stopping = False
        
        # Initialize background monitoring
        self._init_background_monitoring()

    def _init_background_monitoring(self) -> None:
        """Initialize background monitoring."""
        loop = asyncio.get_event_loop()
        self._background_task = loop.create_task(self._monitor_loop())

    @classmethod
    def get_instance(cls) -> 'MonitoringService':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @asynccontextmanager
    async def monitor_flow(self, flow: Flow) -> AsyncGenerator[None, None]:
        """Context manager for monitoring a flow execution."""
        try:
            # Start monitoring
            await self.monitor.start_monitoring(flow.process_id)
            yield
        finally:
            # Stop monitoring and collect final metrics
            await self.monitor.stop_monitoring(flow.process_id)
    
    async def record_flow_event(
        self,
        flow: Flow,
        event_type: str,
        description: str,
        level: int,  # Using logging levels
        details: Dict[str, Any] = None
    ) -> None:
        """Record a flow-related event.
        
        Args:
            flow: Flow instance
            event_type: Type of event
            description: Event description
            level: Logging level (e.g., LoggingLevel.INFO, LoggingLevel.ERROR)
            details: Additional event details
        """
        await self.monitor.record_event(
            flow.process_id,
            event_type,
            description,
            level,
            details
        )

        # Also log using Python's logging system
        logger.log(level, f"Flow {flow.process_id} - {description}")

    async def get_recent_events(
        self,
        flow: Optional['Flow'] = None,
        limit: int = 100,
        min_level: int = LoggingLevel.INFO  # Filter by minimum logging level
    ) -> List[FlowEvent]:
        """Get recent events, optionally filtered."""
        events = self.monitor.events
        
        if flow:
            events = [e for e in events if e.process_id == flow.process_id]
        
        # Filter by minimum logging level
        events = [e for e in events if e.level >= min_level]
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

    async def get_flow_health(self, flow: 'Flow') -> HealthStatus:
        """Get health status for a flow."""
        # Get current statistics
        stats = await self.get_flow_statistics(flow)
        
        # Check basic health metrics
        is_healthy = True
        issues = []
        
        if stats.error_rate > 0.1:  # More than 10% error rate
            is_healthy = False
            issues.append(f"High error rate: {stats.error_rate:.2%}")
            
        if stats.avg_execution_time > 300:  # More than 5 minutes average
            is_healthy = False
            issues.append(f"High average execution time: {stats.avg_execution_time:.1f}s")
            
        # Check dependency health
        deps_health: Dict[str, HealthStatus] = {}
        for dep_id in flow.get_dependencies():
            dep_flow = flow.context.get_flow(dep_id)
            if dep_flow:
                deps_health[dep_id] = await self.get_flow_health(dep_flow)
                
        # If any required dependency is unhealthy, mark as unhealthy
        unhealthy_deps = [
            dep_id for dep_id, health in deps_health.items()
            if not health.healthy and dep_id in flow.get_dependencies(DependencyType.REQUIRED)
        ]
        if unhealthy_deps:
            is_healthy = False
            issues.append(f"Unhealthy required dependencies: {', '.join(unhealthy_deps)}")

        message = "Flow is healthy" if is_healthy else f"Flow has issues: {', '.join(issues)}"
        
        return HealthStatus(
            healthy=is_healthy,
            message=message,
            last_check=datetime.now(),
            details={
                "error_rate": stats.error_rate,
                "avg_execution_time": stats.avg_execution_time,
                "total_executions": stats.total_executions,
                "last_execution": stats.last_execution_time
            },
            dependencies_health=deps_health
        )

    async def get_flow_statistics(self, flow: Flow) -> FlowStatistics:
        """Get execution statistics for a flow."""
        # Get metrics from the collector
        collector = self.monitor.collectors.get(flow.process_id)
        if not collector:
            return FlowStatistics()  # Return empty stats if no collector
            
        # Get all metrics
        metrics = collector.metrics
        
        # Calculate statistics
        executions = [m for m in metrics.get('executions', [])]
        total = len(executions)
        if total == 0:
            return FlowStatistics()
            
        successful = sum(1 for m in executions if m.status == FlowStatus.COMPLETED)
        failed = sum(1 for m in executions if m.status == FlowStatus.FAILED)
        
        execution_times = [m.execution_time for m in executions]
        total_time = sum(execution_times)
        
        return FlowStatistics(
            total_executions=total,
            successful_executions=successful,
            failed_executions=failed,
            total_execution_time=total_time,
            avg_execution_time=total_time / total if total > 0 else 0.0,
            min_execution_time=min(execution_times) if execution_times else None,
            max_execution_time=max(execution_times) if execution_times else None,
            last_execution_time=max(m.timestamp for m in executions) if executions else None,
            error_rate=failed / total if total > 0 else 0.0,
            avg_retry_count=sum(m.retry_count for m in executions) / total if total > 0 else 0.0
        )

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stopping:
            try:
                # Collect system metrics
                metrics = await self.monitor.get_resource_metrics()
                
                # Check for resource warnings
                if metrics.cpu_percent > 80:
                    await self.monitor.record_event(
                        "system",
                        "high_cpu_usage",
                        f"CPU usage is high: {metrics.cpu_percent}%",
                        LoggingLevel.WARNING
                    )
                
                if metrics.memory_percent > 80:
                    await self.monitor.record_event(
                        "system",
                        "high_memory_usage",
                        f"Memory usage is high: {metrics.memory_percent}%",
                        LoggingLevel.WARNING
                    )
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Back off on error
