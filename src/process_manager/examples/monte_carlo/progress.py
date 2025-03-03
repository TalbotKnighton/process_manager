from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import asyncio
from enum import Enum

class ProgressState(Enum):
    """Detailed progress states for tracking"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    GENERATING_INPUTS = "generating_inputs"
    RUNNING_SIMULATIONS = "running_simulations"
    ANALYZING_RESULTS = "analyzing_results"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProgressStats:
    """Statistics for progress tracking"""
    total_cases: int = 0
    completed_cases: int = 0
    failed_cases: int = 0
    cached_cases: int = 0
    current_case: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def progress_percentage(self) -> float:
        """Get overall progress percentage"""
        if self.total_cases == 0:
            return 0.0
        return 100 * self.completed_cases / self.total_cases
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary format"""
        return {
            "total_cases": self.total_cases,
            "completed_cases": self.completed_cases,
            "failed_cases": self.failed_cases,
            "cached_cases": self.cached_cases,
            "current_case": self.current_case,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_time": self.elapsed_time,
            "progress_percentage": self.progress_percentage
        }

class ProgressTracker:
    """
    Tracks and reports progress of Monte Carlo simulation.
    
    Features:
    - Detailed progress states
    - Progress statistics
    - Progress file output
    - Progress callback support
    - ETA calculation
    """
    
    def __init__(self,
                 output_dir: Path,
                 progress_callback: Optional[Callable[[ProgressState, ProgressStats], None]] = None):
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.stats = ProgressStats()
        self.state = ProgressState.NOT_STARTED
        self.progress_file = output_dir / "progress.json"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self, total_cases: int):
        """Start progress tracking"""
        self.stats.total_cases = total_cases
        self.stats.start_time = datetime.now()
        self.state = ProgressState.INITIALIZING
        self._update()
    
    def update_state(self, state: ProgressState):
        """Update progress state"""
        self.state = state
        self._update()
    
    def complete_case(self, case_id: str, cached: bool = False):
        """Mark a case as completed"""
        self.stats.completed_cases += 1
        if cached:
            self.stats.cached_cases += 1
        self.stats.current_case = case_id
        self._update()
    
    def fail_case(self, case_id: str):
        """Mark a case as failed"""
        self.stats.failed_cases += 1
        self.stats.current_case = case_id
        self._update()
    
    def complete(self, success: bool = True):
        """Complete progress tracking"""
        self.stats.end_time = datetime.now()
        self.state = ProgressState.COMPLETED if success else ProgressState.FAILED
        self._update()
    
    def _update(self):
        """Update progress tracking"""
        # Write progress file
        progress_data = {
            "state": self.state.value,
            "stats": self.stats.to_dict(),
            "eta": self._calculate_eta()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Call progress callback if set
        if self.progress_callback:
            self.progress_callback(self.state, self.stats)
    
    def _calculate_eta(self) -> Optional[float]:
        """Calculate estimated time remaining in seconds"""
        if (self.stats.completed_cases == 0 or
            self.stats.start_time is None or
            self.stats.elapsed_time is None):
            return None
        
        # Calculate average time per case
        time_per_case = self.stats.elapsed_time / self.stats.completed_cases
        remaining_cases = self.stats.total_cases - self.stats.completed_cases
        
        return time_per_case * remaining_cases

class AsyncProgressTracker(ProgressTracker):
    """Asynchronous version of progress tracker"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = asyncio.Lock()
    
    async def start(self, total_cases: int):
        """Start progress tracking (async)"""
        async with self._lock:
            super().start(total_cases)
    
    async def update_state(self, state: ProgressState):
        """Update progress state (async)"""
        async with self._lock:
            super().update_state(state)
    
    async def complete_case(self, case_id: str, cached: bool = False):
        """Mark a case as completed (async)"""
        async with self._lock:
            super().complete_case(case_id, cached)
    
    async def fail_case(self, case_id: str):
        """Mark a case as failed (async)"""
        async with self._lock:
            super().fail_case(case_id)
    
    async def complete(self, success: bool = True):
        """Complete progress tracking (async)"""
        async with self._lock:
            super().complete(success)
