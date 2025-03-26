"""Custom exceptions for the flow package."""

class FlowError(Exception):
    """Base class for flow errors."""
    pass

class FlowExecutionError(FlowError):
    """Error during flow execution."""
    def __init__(self, message: str, traceback: str):
        super().__init__(message)
        self.traceback = traceback

class FlowTimeoutError(FlowError):
    """Flow execution timed out."""
    pass

class FlowRetryError(FlowError):
    """Flow retry attempts exhausted."""
    def __init__(self, message: str, original_error: Exception):
        super().__init__(message)
        self.original_error = original_error

class MissingDependencyError(FlowError):
    """Required dependency not found."""
    pass