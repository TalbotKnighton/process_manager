"""Example of a simple CPU-bound process that generates a random wait time and returns a Pydantic model."""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Optional
from pydantic import BaseModel

from process_manager.workflow.core import create_workflow
from process_manager.workflow.process import BaseProcess, ProcessConfig
from process_manager.workflow.workflow_types import ProcessType, WorkflowNode

class WaitResult(BaseModel):
    """Pydantic model to represent the result of a random wait process."""
    process_index: int
    wait_time: float
    timestamp: float

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

    def process(self, input_data: Any) -> WaitResult:
        # Generate random wait time between 1 and 2 seconds
        wait_time = self.random.uniform(1.0, 2.0)
        
        print(f"Process {self.process_index} starting with wait time: {wait_time:.2f} seconds")
        time.sleep(wait_time)  # Using time.sleep to simulate CPU-bound work
        print(f"Process {self.process_index} completed")
        
        return WaitResult(
            process_index=self.process_index,
            wait_time=wait_time,
            timestamp=time.time()
        )

async def main():
    # Create the main workflow
    workflow = create_workflow(
        process_id="random_wait_workflow",
        max_processes=2  # Allow 2 concurrent processes
    )
    
    # Create 4 random wait processes
    processes = [
        RandomWaitProcess(process_index=i, seed=42 + i)
        for i in range(4)
    ]
    
    # Add each process to the workflow
    for process in processes:
        workflow.add_node(WorkflowNode(
            process=process,
            dependencies=[],  # No dependencies between processes
            required=True
        ))
    
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
            result: WaitResult = results[process.config.process_id].data
            print(f"Process {result.process_index}: waited {result.wait_time:.2f} seconds "
                  f"(completed at {time.strftime('%H:%M:%S', time.localtime(result.timestamp))})")

    # Demonstrate Pydantic model validation and serialization
    print("\nPydantic model example:")
    first_result: WaitResult = results[processes[0].config.process_id].data
    print(f"JSON representation: {first_result.model_dump_json(indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())