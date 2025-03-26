"""Type definitions for flow monitoring system."""
from __future__ import annotations
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from flow.core.types import FlowStatus, LoggingLevel 

class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"      # Incrementing values (e.g., number of executions)
    GAUGE = "gauge"         # Point-in-time values (e.g., memory usage)
    HISTOGRAM = "histogram" # Distribution of values (e.g., execution times)
    STATE = "state"        # State transitions (e.g., flow status changes)

class MetricValue(BaseModel):
    """Single metric measurement."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)

class HistogramBucket(BaseModel):
    """Histogram bucket for distribution metrics."""
    le: float  # less than or equal to
    count: int

class HistogramValue(BaseModel):
    """Histogram metric with distribution information."""
    count: int
    sum: float
    buckets: List[HistogramBucket]

class FlowMetrics(BaseModel):
    """Collection of metrics for a flow."""
    process_id: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    start_time: datetime
    end_time: Optional[datetime] = None
    retry_count: int = 0
    error_count: int = 0
    dependency_wait_time: float = 0.0
    queue_time: float = 0.0
    processed_data_size: Optional[int] = None
    
    class Config:
        frozen = True

class FlowEvent(BaseModel):
    """Event record for flow monitoring."""
    timestamp: datetime
    process_id: str
    event_type: str
    description: str
    level: LoggingLevel
    details: Dict[str, Any] = Field(default_factory=dict)

class ResourceMetrics(BaseModel):
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_received: int
    thread_count: int
    process_count: int
    
    class Config:
        frozen = True

class HealthStatus(BaseModel):
    """Health check status for a flow or component."""
    healthy: bool
    message: str
    last_check: datetime
    details: Dict[str, Any] = Field(default_factory=dict)
    dependencies_health: Dict[str, HealthStatus] = Field(default_factory=dict)

class FlowStatistics(BaseModel):
    """Statistical information about flow execution."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    min_execution_time: Optional[float] = None
    max_execution_time: Optional[float] = None
    last_execution_time: Optional[datetime] = None
    error_rate: float = 0.0
    avg_retry_count: float = 0.0
    status_counts: Dict[FlowStatus, int] = Field(
        default_factory=lambda: {status: 0 for status in FlowStatus}
    )

    def update_status(self, status: FlowStatus) -> None:
        self.status_counts[status] += 1
# """Type definitions for flow monitoring system."""
# from enum import Enum
# from typing import Dict, Any, Optional, List
# from datetime import datetime
# from pydantic import BaseModel, Field

# from flow.core.types import FlowStatus

# class MetricType(Enum):
#     """Types of metrics that can be collected."""
#     COUNTER = "counter"      # Incrementing values (e.g., number of executions)
#     GAUGE = "gauge"         # Point-in-time values (e.g., memory usage)
#     HISTOGRAM = "histogram" # Distribution of values (e.g., execution times)
#     STATE = "state"        # State transitions (e.g., flow status changes)

# class LoggingLevel(Enum):
#     """Alert severity levels."""
#     INFO = "info"
#     WARNING = "warning"
#     ERROR = "error"
#     CRITICAL = "critical"

# class MetricValue(BaseModel):
#     """Single metric measurement."""
#     timestamp: datetime
#     value: float
#     labels: Dict[str, str] = Field(default_factory=dict)

# class HistogramBucket(BaseModel):
#     """Histogram bucket for distribution metrics."""
#     le: float  # less than or equal to
#     count: int

# class HistogramValue(BaseModel):
#     """Histogram metric with distribution information."""
#     count: int
#     sum: float
#     buckets: List[HistogramBucket]

# class FlowMetrics(BaseModel):
#     """Collection of metrics for a flow."""
#     process_id: str
#     execution_time: float
#     memory_usage: float
#     cpu_usage: float
#     start_time: datetime
#     end_time: Optional[datetime] = None
#     retry_count: int = 0
#     error_count: int = 0
#     dependency_wait_time: float = 0.0
#     queue_time: float = 0.0
#     processed_data_size: Optional[int] = None
    
#     class Config:
#         frozen = True

# class FlowEvent(BaseModel):
#     """Event record for flow monitoring."""
#     timestamp: datetime
#     process_id: str
#     event_type: str
#     description: str
#     level: LoggingLevel
#     details: Dict[str, Any] = Field(default_factory=dict)

# class ResourceMetrics(BaseModel):
#     """System resource metrics."""
#     cpu_percent: float
#     memory_percent: float
#     disk_usage_percent: float
#     network_bytes_sent: int
#     network_bytes_received: int
#     thread_count: int
#     process_count: int
    
#     class Config:
#         frozen = True

# class HealthStatus(BaseModel):
#     """Health check status for a flow or component."""
#     healthy: bool
#     message: str
#     last_check: datetime
#     details: Dict[str, Any] = Field(default_factory=dict)
#     dependencies_health: Dict[str, 'HealthStatus'] = Field(default_factory=dict)

# class FlowStatistics(BaseModel):
#     """Statistical information about flow execution."""
#     total_executions: int = 0
#     successful_executions: int = 0
#     failed_executions: int = 0
#     total_execution_time: float = 0.0
#     avg_execution_time: float = 0.0
#     min_execution_time: Optional[float] = None
#     max_execution_time: Optional[float] = None
#     last_execution_time: Optional[datetime] = None
#     error_rate: float = 0.0
#     avg_retry_count: float = 0.0
#     status_counts: Dict[FlowStatus, int] = Field(
#         default_factory=lambda: {status: 0 for status in FlowStatus}
#     )

#     def update_status(self, status: FlowStatus) -> None:
#         self.status_counts[status] += 1