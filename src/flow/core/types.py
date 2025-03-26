"""Core type definitions for the flow package."""
from enum import Enum, auto
from typing import Optional, Any
import logging

class LoggingLevel(Enum):
    ERROR = logging.ERROR                                                                                                             
    WARNING = logging.WARNING
    INFO = logging.INFO
    CRITICAL = logging.CRITICAL
    DEBUG = logging.DEBUG

    def __str__(self):
        return str(self.value)
    def __int__(self):
        return int(self.value)
    def __repr__(self):
        return repr(self.value)

class FlowType(Enum):
    """Type of flow execution."""
    INLINE = "inline"
    THREAD = "thread"
    PROCESS = "process"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"

class FlowStatus(Enum):
    """Status of flow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StorageType(Enum):
    """Type of result storage."""
    MEMORY = "memory"
    DISK = "disk"
    SQLITE = "sqlite"

class VisFormat(Enum):
    """Visualization format options."""
    MERMAID = auto()
    GRAPHVIZ = auto()
    PLOTLY = auto()

class DependencyType(Enum):
    """Type of dependency relationship."""
    REQUIRED = "required"
    OPTIONAL = "optional"
