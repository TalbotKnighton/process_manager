"""Example of a process using composition rather than inheritance."""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Optional, Callable
from pydantic import BaseModel, Field
from functools import partial

from process_manager.workflow.core import create_workflow
from process_manager.workflow.process import BaseProcess, ProcessConfig
from process_manager.workflow.workflow_types import ProcessType
from process_manager import data_handlers as dh

class ProcessResult(BaseModel):
    """Generic process result that includes a NamedValueHash for data storage."""
    process_index: int
    timestamp: float
    data: dh.NamedValueHash

class RandomWaitConfig(BaseModel):
    """Configuration for a random wait process."""
    process_index: int
    seed: Optional[int] = None
    min_wait: float = Field(default=1.0, gt=0)
    max_wait: float = Field(default=2.0, gt=0)
    
    @classmethod
    def get_process_id(cls, process_index: int) -> str:
        return f"random_wait_{process_index}"
    
    @property
    def process_id(self) -> str:
        return self.get_process_id(self.process_index)

def random_wait_function(config: RandomWaitConfig, input_data: Any) -> ProcessResult:
    """Function that implements the random wait logic."""
    # Initialize random number generator
    rng = random.Random(config.seed)
    
    # Generate random wait time
    wait_time = rng.uniform(config.min_wait, config.max_wait)
    
    print(f"Process {config.process_index} starting with wait time: {wait_time:.2f} seconds")
    time.sleep(wait_time)
    print(f"Process {config.process_index} completed")
    
    # Create a NamedValueHash to store the process data
    data = dh.NamedValueHash()
    data.register_value(value=dh.NamedValue("wait_time", wait_time))
    data.register_value(value=dh.NamedValue("process_name", f"random_wait_{config.process_index}"))
    data.register_value(value=dh.NamedValue("random_seed", config.seed))  # Get first value from random state
    
    return ProcessResult(
        process_index=config.process_index,
        timestamp=time.time(),
        data=data
    )

def create_random_wait_process(config: RandomWaitConfig) -> BaseProcess:
    """Factory function to create a random wait process."""
    process = BaseProcess(
        config=ProcessConfig(
            process_type=ProcessType.PROCESS,
            process_id=config.process_id
        )
    )
    # Bind the config to the function and set it as the process method
    process.process = partial(random_wait_function, config)
    return process

async def main():
    # Create the main workflow
    workflow = create_workflow(
        process_id="random_wait_workflow",
        max_processes=2
    )
    
    # Create processes using the factory function
    processes = [
        create_random_wait_process(
            RandomWaitConfig(
                process_index=i,
                seed=42 + i,
                min_wait=1.0,
                max_wait=2.0
            )
        ).register_to(workflow)
        for i in range(4)
    ]
    
    # Execute the workflow and get results
    print("Starting workflow execution...")
    start_time = time.time()
    
    results = await workflow.execute()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print results
    print("\nWorkflow completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("\nProcess results:")
    for process in processes:
        if process.config.process_id in results:
            result: ProcessResult = results[process.config.process_id].data
            
            print(f"Process {result.process_index} ({result.data.get_value('process_name')}):")
            print(f"  - Wait time: {result.data.get_value('wait_time'):.2f} seconds")
            print(f"  - Random seed: {result.data.get_value('random_seed')}")
            print(f"  - Completed at: {time.strftime('%H:%M:%S', time.localtime(result.timestamp))}")

    # Demonstrate configuration validation
    print("\nPydantic validation example:")
    try:
        invalid_config = RandomWaitConfig(process_index=0, min_wait=0.1)
    except ValueError as e:
        print("Caught validation error:", e)

if __name__ == "__main__":
    asyncio.run(main())