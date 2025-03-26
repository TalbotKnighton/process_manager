"""Process pool management for flow execution."""
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Dict, Any, Optional, Callable, Set
import multiprocessing as mp
import threading
import logging
import time
from functools import partial

from flow.core.types import FlowType, FlowStatus
from flow.core.errors import FlowError, FlowTimeoutError

logger = logging.getLogger(__name__)

class ProcessPoolManager:
    """Manages process and thread pools for flow execution."""

    def __init__(
        self,
        max_threads: Optional[int] = None,
        max_processes: Optional[int] = None
    ):
        self.max_threads = max_threads or mp.cpu_count()
        self.max_processes = max_processes or mp.cpu_count()
        
        # Initialize pools
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_threads)
        self._process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        
        # Track active tasks
        self._futures: Dict[str, Future] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._loop = asyncio.get_event_loop()
        
    async def submit_task(
        self,
        process_id: str,
        flow_type: FlowType,
        func: Callable,
        input_data: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """Submit a task for execution."""
        if process_id in self._futures:
            raise FlowError(f"Task {process_id} already running")
            
        self._locks[process_id] = asyncio.Lock()
        
        try:
            # Create the executor based on flow type
            executor = self._thread_pool if flow_type == FlowType.THREAD else self._process_pool
            
            # Submit the task
            future = executor.submit(func, input_data)
            self._futures[process_id] = future
            
            # Wait for completion
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=timeout
                    )
                else:
                    result = await asyncio.wrap_future(future)
                
                return result
                
            except asyncio.TimeoutError:
                future.cancel()
                raise FlowTimeoutError(f"Task {process_id} timed out after {timeout} seconds")
                
        finally:
            self._cleanup_task(process_id)
    # async def submit_task(
    #     self,
    #     process_id: str,
    #     flow_type: FlowType,
    #     func: Callable,
    #     *args,
    #     timeout: Optional[float] = None,
    #     **kwargs
    # ) -> Any:
    #     """Submit a task for execution.
        
    #     Args:
    #         process_id: Unique identifier for the task
    #         flow_type: Type of execution (THREAD/PROCESS)
    #         func: Function to execute
    #         *args: Positional arguments for the function
    #         timeout: Optional timeout in seconds
    #         **kwargs: Keyword arguments for the function
            
    #     Returns:
    #         Task result
    #     """
    #     if process_id in self._futures:
    #         raise FlowError(f"Task {process_id} already running")
            
    #     self._locks[process_id] = asyncio.Lock()
        
    #     try:
    #         if flow_type == FlowType.THREAD:
    #             future = self._thread_pool.submit(func, *args, **kwargs)
    #         elif flow_type == FlowType.PROCESS:
    #             future = self._process_pool.submit(func, *args, **kwargs)
    #         else:
    #             raise FlowError(f"Unsupported flow type: {flow_type}")
                
    #         self._futures[process_id] = future
            
    #         # Wait for completion with timeout
    #         try:
    #             result = await asyncio.wait_for(
    #                 self._loop.run_in_executor(None, future.result),
    #                 timeout=timeout
    #             )
    #             return result
                
    #         except asyncio.TimeoutError:
    #             future.cancel()
    #             raise FlowTimeoutError(f"Task {process_id} timed out after {timeout} seconds")
                
    #     finally:
    #         self._cleanup_task(process_id)

    async def cancel_task(self, process_id: str) -> None:
        """Cancel a running task."""
        future = self._futures.get(process_id)
        if future and not future.done():
            future.cancel()
            self._cleanup_task(process_id)

    def _cleanup_task(self, process_id: str) -> None:
        """Clean up task resources."""
        self._futures.pop(process_id, None)
        self._locks.pop(process_id, None)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the pool manager."""
        # Cancel all running tasks
        for process_id in list(self._futures.keys()):
            asyncio.create_task(self.cancel_task(process_id))
            
        # Shutdown pools
        self._thread_pool.shutdown(wait=wait)
        self._process_pool.shutdown(wait=wait)
        
        # Clear tracking
        self._futures.clear()
        self._locks.clear()

    async def wait_for_task(self, process_id: str, timeout: Optional[float] = None) -> None:
        """Wait for a specific task to complete."""
        future = self._futures.get(process_id)
        if not future:
            return
            
        try:
            await asyncio.wait_for(
                self._loop.run_in_executor(None, future.result),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise FlowTimeoutError(f"Timeout waiting for task {process_id}")

    @property
    def active_tasks(self) -> Set[str]:
        """Get set of currently active task IDs."""
        return {
            pid for pid, future in self._futures.items()
            if not future.done()
        }

class _ProcessContext:
    """Context manager for process pool tasks."""
    def __init__(self, pool: ProcessPoolManager, process_id: str):
        self.pool = pool
        self.process_id = process_id
        self.lock = pool._locks.get(process_id)

    async def __aenter__(self):
        if self.lock:
            await self.lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.lock:
            self.lock.release()
        if exc_type:
            await self.pool.cancel_task(self.process_id)