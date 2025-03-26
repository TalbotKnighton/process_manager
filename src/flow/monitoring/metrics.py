"""Metric collection and management for flows."""
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import psutil
import logging
from collections import defaultdict
import statistics

from flow.monitoring.types import (
    MetricType, MetricValue, FlowMetrics,
    FlowEvent, ResourceMetrics, HealthStatus, FlowStatistics
)
from flow.core.types import FlowStatus

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages metrics for a flow."""
    
    def __init__(self, process_id: str):
        self.process_id = process_id
        self.start_time: Optional[datetime] = None
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self._process = psutil.Process()
        self._prev_cpu_time = 0
        self._prev_time = time.time()

    def start_collection(self) -> None:
        """Start collecting metrics."""
        self.start_time = datetime.now()
        self._prev_cpu_time = self._process.cpu_times().user + self._process.cpu_times().system
        self._prev_time = time.time()

    def collect_metrics(self) -> FlowMetrics:
        """Collect current metrics."""
        current_time = time.time()
        cpu_times = self._process.cpu_times()
        current_cpu_time = cpu_times.user + cpu_times.system
        
        # Calculate CPU usage
        cpu_usage = (current_cpu_time - self._prev_cpu_time) / (current_time - self._prev_time) * 100
        
        # Update previous values
        self._prev_cpu_time = current_cpu_time
        self._prev_time = current_time
        
        return FlowMetrics(
            process_id=self.process_id,
            execution_time=time.time() - self.start_time.timestamp(),
            memory_usage=self._process.memory_percent(),
            cpu_usage=cpu_usage,
            start_time=self.start_time,
            end_time=datetime.now()
        )

    def add_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None) -> None:
        """Add a metric measurement."""
        self.metrics[name].append(
            MetricValue(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {}
            )
        )

class FlowMonitor:
    """Monitors and tracks flow execution and health."""
    
    def __init__(self):
        self.collectors: Dict[str, MetricsCollector] = {}
        self.events: List[FlowEvent] = []
        self.statistics: Dict[str, FlowStatistics] = defaultdict(FlowStatistics)
        self._lock = asyncio.Lock()

    async def start_monitoring(self, process_id: str) -> None:
        """Start monitoring a flow."""
        async with self._lock:
            collector = MetricsCollector(process_id)
            collector.start_collection()
            self.collectors[process_id] = collector
            
        await self.record_event(
            process_id,
            "monitoring_started",
            "Started monitoring flow",
            LoggingLevel.INFO
        )

    async def stop_monitoring(self, process_id: str) -> None:
        """Stop monitoring a flow and collect final metrics."""
        async with self._lock:
            if process_id in self.collectors:
                metrics = self.collectors[process_id].collect_metrics()
                await self.update_statistics(process_id, metrics)
                del self.collectors[process_id]

    async def record_event(
        self,
        process_id: str,
        event_type: str,
        description: str,
        level: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Record a flow event."""
        event = FlowEvent(
            timestamp=datetime.now(),
            process_id=process_id,
            event_type=event_type,
            description=description,
            level=level,
            details=details or {}
        )
        
        async with self._lock:
            self.events.append(event)
            
        if level in (LoggingLevel.ERROR, LoggingLevel.CRITICAL):
            logger.error(f"Flow {process_id} - {description}")
        elif level == LoggingLevel.WARNING:
            logger.warning(f"Flow {process_id} - {description}")
        else:
            logger.info(f"Flow {process_id} - {description}")

    async def update_statistics(self, process_id: str, metrics: FlowMetrics) -> None:
        """Update flow statistics with new metrics."""
        async with self._lock:
            stats = self.statistics[process_id]
            stats.total_executions += 1
            stats.total_execution_time += metrics.execution_time
            stats.avg_execution_time = stats.total_execution_time / stats.total_executions
            
            if stats.min_execution_time is None or metrics.execution_time < stats.min_execution_time:
                stats.min_execution_time = metrics.execution_time
            
            if stats.max_execution_time is None or metrics.execution_time > stats.max_execution_time:
                stats.max_execution_time = metrics.execution_time
            
            stats.last_execution_time = datetime.now()

    async def get_health_status(self, process_id: str) -> HealthStatus:
        """Get current health status of a flow."""
        stats = self.statistics.get(process_id)
        if not stats:
            return HealthStatus(
                healthy=True,
                message="No execution history",
                last_check=datetime.now()
            )
        
        # Calculate health based on error rate and performance
        error_rate = stats.failed_executions / stats.total_executions if stats.total_executions > 0 else 0
        avg_exec_time = stats.avg_execution_time
        
        healthy = error_rate < 0.1 and (stats.max_execution_time or 0) < 300  # example thresholds
        
        return HealthStatus(
            healthy=healthy,
            message="Flow operating normally" if healthy else "Flow showing issues",
            last_check=datetime.now(),
            details={
                "error_rate": error_rate,
                "avg_execution_time": avg_exec_time,
                "total_executions": stats.total_executions
            }
        )

    async def get_resource_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        return ResourceMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent,
            network_bytes_sent=psutil.net_io_counters().bytes_sent,
            network_bytes_received=psutil.net_io_counters().bytes_recv,
            thread_count=psutil.Process().num_threads(),
            process_count=len(psutil.Process().children())
        )