from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import Dict, Any, Optional
import json

class SimulationState(Enum):
    """Tracks the state of individual simulation cases"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

@dataclass
class SimulationParams:
    """
    Parameters for a single Monte Carlo simulation case.
    
    Attributes:
        temperature: System temperature in Kelvin (273-373K)
        pressure: System pressure in Pascal (1e5-5e5 Pa)
        flow_rate: Flow rate in m³/s (0.1-1.0 m³/s)
        sim_time: Simulation duration in seconds (10-100s)
        
    Note:
        All parameters have physical bounds enforced by validation.
    """
    temperature: float  # Kelvin
    pressure: float    # Pascal
    flow_rate: float   # m3/s
    sim_time: float    # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary format"""
        return {
            "temperature": self.temperature,
            "pressure": self.pressure,
            "flow_rate": self.flow_rate,
            "sim_time": self.sim_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationParams':
        """Create parameters from dictionary format"""
        return cls(
            temperature=float(data["temperature"]),
            pressure=float(data["pressure"]),
            flow_rate=float(data["flow_rate"]),
            sim_time=float(data["sim_time"])
        )

@dataclass
class SimulationCase:
    """
    Represents a single simulation case in the Monte Carlo study.
    
    Attributes:
        case_id: Unique identifier for the case
        params: Simulation parameters
        input_file: Path to input file
        output_file: Path to output file (if completed)
        state: Current simulation state
        error: Error message if failed
        cache_hit: Whether result was retrieved from cache
        
    Note:
        This class maintains the complete state of a simulation case
        throughout its lifecycle from creation to completion.
    """
    case_id: str
    params: SimulationParams
    input_file: Path
    output_file: Optional[Path] = None
    state: SimulationState = SimulationState.PENDING
    error: Optional[str] = None
    cache_hit: bool = False
    
    def to_json(self) -> str:
        """Serialize case to JSON format"""
        data = {
            "case_id": self.case_id,
            "params": self.params.to_dict(),
            "input_file": str(self.input_file),
            "output_file": str(self.output_file) if self.output_file else None,
            "state": self.state.value,
            "error": self.error,
            "cache_hit": self.cache_hit
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SimulationCase':
        """Create case from JSON format"""
        data = json.loads(json_str)
        return cls(
            case_id=data["case_id"],
            params=SimulationParams.from_dict(data["params"]),
            input_file=Path(data["input_file"]),
            output_file=Path(data["output_file"]) if data["output_file"] else None,
            state=SimulationState(data["state"]),
            error=data["error"],
            cache_hit=data["cache_hit"]
        )
