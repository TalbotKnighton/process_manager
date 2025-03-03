from threading import Lock
from typing import Optional
from datetime import datetime
import re

class ProcessIdGenerator:
    """Thread-safe process ID generator."""
    
    def __init__(self, prefix: str = "proc"):
        self._prefix = prefix
        self._counter = 0
        self._lock = Lock()
        self._reserved_ids = set()
    
    def next_id(self, prefix: Optional[str] = None) -> str:
        """
        Generate and reserve the next available process ID.
        
        Args:
            prefix: Optional override for the default prefix
        
        Returns:
            A unique process ID string in format: prefix_number (e.g., proc_1)
        """
        with self._lock:
            self._counter += 1
            while True:
                process_id = f"{prefix or self._prefix}_{self._counter}"
                if process_id not in self._reserved_ids:
                    self._reserved_ids.add(process_id)
                    return process_id
                self._counter += 1
    
    def reserve_id(self, process_id: str) -> bool:
        """
        Try to reserve a specific process ID.
        
        Args:
            process_id: The process ID to reserve
            
        Returns:
            True if the ID was reserved, False if already taken
        """
        with self._lock:
            if process_id in self._reserved_ids:
                return False
            self._reserved_ids.add(process_id)
            
            # Update counter if necessary to maintain sequence
            match = re.match(f"{self._prefix}_(\d+)", process_id)
            if match:
                num = int(match.group(1))
                self._counter = max(self._counter, num)
            
            return True
    
    def release_id(self, process_id: str) -> None:
        """
        Release a reserved process ID back to the pool.
        
        Args:
            process_id: The process ID to release
        """
        with self._lock:
            self._reserved_ids.discard(process_id)
    
    def is_reserved(self, process_id: str) -> bool:
        """
        Check if a process ID is already reserved.
        
        Args:
            process_id: The process ID to check
            
        Returns:
            True if the ID is reserved, False otherwise
        """
        return process_id in self._reserved_ids
    
    def reset(self) -> None:
        """Reset the generator state."""
        with self._lock:
            self._counter = 0
            self._reserved_ids.clear()

class TimestampedProcessIdGenerator(ProcessIdGenerator):
    """Process ID generator that includes timestamps in IDs."""
    
    def next_id(self, prefix: Optional[str] = None) -> str:
        """
        Generate a timestamped process ID.
        
        Returns:
            A unique process ID string in format: prefix_timestamp_number 
            (e.g., proc_20230815_1)
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        with self._lock:
            self._counter += 1
            while True:
                process_id = f"{prefix or self._prefix}_{timestamp}_{self._counter}"
                if process_id not in self._reserved_ids:
                    self._reserved_ids.add(process_id)
                    return process_id
                self._counter += 1