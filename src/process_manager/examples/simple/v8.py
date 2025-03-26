"""Example of a process that uses ProcessResult with NamedValueHash for data handling."""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Optional
from pydantic import BaseModel

from process_manager.workflow.core import create_workflow
from process_manager.workflow.process import BaseProcess, ProcessConfig
from process_manager.workflow.workflow_types import ProcessType, WorkflowNode
# from process_manager.data_handlers.values import NamedValueHash, NamedValue
from process_manager import data_handlers as dh

class ProcessResult(BaseModel):
    """Generic process result that includes a NamedValueHash for data storage."""
    process_index: int
    timestamp: float
    data: dh.NamedValueHash

class RandomWaitProcess(BaseProcess):
    def __init__(self, process_index: int, seed: Optional[int] = None):
        super().__init__(
            config=ProcessConfig(
                process_type=ProcessType.PROCESS,
                process_id=f"random_wait_{process_index}"
            )
        )
        self.random = random.Random(seed)
        self.process_index = process_index

    def process(self, input_data: Any) -> ProcessResult:
        # Generate random wait time between 1 and 2 seconds
        # wait_time = self.random.uniform(1.0, 2.0)
        wait_time = dh.UniformDistribution(name='wait_time', low=1, high=2).sample().squeeze()

        
        print(f"Process {self.process_index} starting with wait time: {wait_time:.2f} seconds")
        time.sleep(int(wait_time))  # Using time.sleep to simulate CPU-bound work
        print(f"Process {self.process_index} completed")
        
        # Create a NamedValueHash to store the process data
        data = dh.NamedValueHash()
        data.register_value(value=dh.NamedValue("wait_time", wait_time))
        data.register_value(value=dh.NamedValue("process_name", f"random_wait_{self.process_index}"))
        data.register_value(value=dh.NamedValue("random_seed", self.random.getstate()[1][0]))  # Get first value from random state
        
        return ProcessResult(
            process_index=self.process_index,
            timestamp=time.time(),
            data=data
        )

async def main():
    # Create the main workflow
    workflow = create_workflow(
        process_id="random_wait_workflow",
        max_processes=6  # Allow 2 concurrent processes
    )
    
    # Create and register processes in a more fluent way
    processes = [
        RandomWaitProcess(
            process_index=i, 
            seed=42 + i
        ).register_to(workflow)
        for i in range(10)
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
            wait_time = result.data.get_value("wait_time")
            process_name = result.data.get_value("process_name")
            random_seed = result.data.get_value("random_seed")
            
            print(f"Process {result.process_index} ({process_name}):")
            print(f"  - Wait time: {wait_time:.2f} seconds")
            print(f"  - Random seed: {random_seed}")
            print(f"  - Completed at: {time.strftime('%H:%M:%S', time.localtime(result.timestamp))}")

    # Demonstrate data access and serialization
    print("\nNamedValueHash example:")
    first_result: ProcessResult = results[processes[0].config.process_id].data
    print("Available data keys:", first_result.data.get_value_names())
    print("JSON representation:")
    print(first_result.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())