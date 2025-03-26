"""Logging configuration for the flow system."""
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import threading
from functools import partial
import traceback
import contextlib

from flow.core.types import LoggingLevel

class FlowFormatter(logging.Formatter):
    """Custom formatter for flow system logs.
    
    Format example:
    2024-01-24 15:30:45.123 [INFO] [FlowName:abc123] Started execution - {"context": "additional data"}
    2024-01-24 15:30:46.234 [ERROR] [FlowName:abc123] Execution failed - {"error": "details", "traceback": "..."}
    """
    
    def __init__(self, include_process_thread: bool = False):
        super().__init__()
        self.include_process_thread = include_process_thread

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        # Extract flow information if available
        flow_info = f"[{getattr(record, 'flow_name', 'System')}:{getattr(record, 'process_id', 'N/A')}]"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Format level
        level = record.levelname.ljust(8)
        
        # Process and thread info if requested
        proc_thread = ""
        if self.include_process_thread:
            proc_thread = f"[P:{record.process}|T:{record.thread}] "
        
        # Format the message
        msg = record.getMessage()
        
        # Format any extra contextual data
        extra = ""
        if hasattr(record, 'flow_context'):
            try:
                extra = f" - {json.dumps(record.flow_context, default=str)}"
            except Exception:
                extra = f" - {str(record.flow_context)}"
        
        # Combine all parts
        log_message = f"{timestamp} [{level}] {proc_thread}{flow_info} {msg}{extra}"
        
        # Add exception information if present
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            log_message = f"{log_message}\nException:\n{exc_text}"
            
        return log_message

class FlowLogger:
    """Enhanced logger for flow system with context management."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
        self._context_lock = threading.Lock()
    
    @contextlib.contextmanager
    def flow_context(self, **kwargs):
        """Context manager for adding flow-specific context to logs."""
        with self._context_lock:
            old_context = self.context.copy()
            self.context.update(kwargs)
            try:
                yield
            finally:
                self.context = old_context

    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal logging method that adds flow context."""
        extra = kwargs.pop('extra', {})
        extra['flow_context'] = {**self.context, **extra.get('flow_context', {})}
        self.logger.log(level, msg, *args, extra=extra, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log(LoggingLevel.INFO.value, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log(LoggingLevel.ERROR.value, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(LoggingLevel.WARNING.value, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(LoggingLevel.DEBUG.value, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log(LoggingLevel.CRITICAL.value, msg, *args, **kwargs)

def setup_logging(
    log_file: Optional[str] = None,
    level: int = LoggingLevel.INFO,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    include_process_thread: bool = False
) -> None:
    """Set up logging configuration for the flow system.
    
    Args:
        log_file: Optional path to log file. If None, logs to console only
        level: Minimum logging level
        max_bytes: Maximum size of each log file
        backup_count: Number of backup log files to keep
        include_process_thread: Whether to include process and thread IDs in logs
    """
    root_logger = logging.getLogger('flow')
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = FlowFormatter(include_process_thread=include_process_thread)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

# Usage example:
"""
# Set up logging
setup_logging(
    log_file="logs/flow_system.log",
    level=LoggingLevel.INFO,
    include_process_thread=True
)

# Create a logger for a component
logger = FlowLogger("flow.core")

# Use in a flow
with logger.flow_context(flow_name="DataProcessor", process_id="abc123"):
    logger.info("Starting data processing", extra={
        'flow_context': {
            'input_size': 1000,
            'batch_size': 100
        }
    })
    
    try:
        # Process data
        logger.info("Processing complete", extra={
            'flow_context': {
                'records_processed': 1000,
                'time_taken': 5.2
            }
        })
    except Exception as e:
        logger.error(
            "Processing failed",
            extra={
                'flow_context': {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
            }
        )
"""