# src/flow/core/results.py
from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from flow.core.types import FlowStatus

if TYPE_CHECKING:
    from flow.core.context import FlowContext

logger = logging.getLogger(__name__)

class FlowResult(BaseModel):
    """Immutable result of a completed flow execution."""
    process_id: str
    status: FlowStatus
    start_time: datetime
    end_time: datetime
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True  # Make the model immutable

    @classmethod
    def create_completed(
        cls, 
        process_id: str,
        output: Dict[str, Any],
        start_time: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Create a result for a completed flow."""
        return cls(
            process_id=process_id,
            status=FlowStatus.COMPLETED,
            start_time=start_time,
            end_time=datetime.now(),
            output=output,
            metadata=metadata or {}
        )

    @classmethod
    def create_failed(
        cls,
        process_id: str,
        error: str,
        start_time: datetime,
        traceback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'FlowResult':
        """Create a result for a failed flow."""
        return cls(
            process_id=process_id,
            status=FlowStatus.FAILED,
            start_time=start_time,
            end_time=datetime.now(),
            error=error,
            traceback=traceback,
            metadata=metadata or {}
        )

class ResultsManager:
    """Manages immutable flow results."""
    
    def __init__(self, context: 'FlowContext'):
        self.context = context
        self._results: Dict[str, FlowResult] = {}
        self._lock = asyncio.Lock()
        logger.debug("ResultsManager initialized")

    async def save_result(self, process_id: str, result: FlowResult) -> None:
        """Save an immutable flow result."""
        async with self._lock:
            self._results[process_id] = result
            logger.debug(f"Saved result for process {process_id}: {result}")

    async def get_result(self, process_id: str) -> Optional[FlowResult]:
        """Get an immutable flow result."""
        async with self._lock:
            result = self._results.get(process_id)
            logger.debug(f"Retrieved result for process {process_id}: {result}")
            return result

    def cleanup(self) -> None:
        """Clear all stored results."""
        self._results.clear()
        logger.debug("ResultsManager cleaned up")
    async def get_dependency_outputs(self, process_id: str, dep_ids: set[str]) -> Dict[str, Any]:
        """Get combined outputs from multiple dependencies."""
        async with self._lock:
            outputs = {}
            for dep_id in dep_ids:
                result = self._results.get(dep_id)
                if result and result.output:
                    # Store outputs with process ID as prefix to avoid collisions
                    prefixed_outputs = {
                        f"{dep_id}.{k}": v 
                        for k, v in result.output.items()
                    }
                    outputs.update(prefixed_outputs)
            logger.debug(f"Retrieved dependency outputs for {process_id}: {outputs}")
            return outputs

    async def get_dependency_output(self, process_id: str, dep_id: str) -> Optional[Dict[str, Any]]:
        """Get output from a specific dependency."""
        async with self._lock:
            result = self._results.get(dep_id)
            if result and result.output:
                logger.debug(f"Retrieved dependency {dep_id} output for {process_id}: {result.output}")
                return result.output
            return None
