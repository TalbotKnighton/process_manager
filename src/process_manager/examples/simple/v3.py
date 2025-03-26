"""Example of nested workflows for parallel random number generation."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import os

from process_manager.workflow.core import create_workflow
from process_manager.workflow.process import BaseProcess, ProcessConfig, ProcessId
from process_manager.workflow.workflow_types import ProcessType, WorkflowNode
from process_manager.data_handlers.random_variables import NormalDistribution

class RandomNumberGenerator(BaseProcess):
    def __init__(self, index: int, seed: Optional[int] = None):
        super().__init__(
            config=ProcessConfig(
                process_type=ProcessType.THREAD,
                process_id="generator",
            )
        )
        self.distribution = NormalDistribution(
            name="random_value",
            mu=0.0,
            sigma=1.0,
            seed=seed
        )
    
    def process(self, input_data: Any) -> float:
        return self.distribution.sample(size=1).item()

class FileWriter(BaseProcess):
    def __init__(self, index: int, output_dir: Path):
        super().__init__(
            config=ProcessConfig(
                process_type=ProcessType.THREAD,
                process_id="writer",
            )
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, input_data: Dict[str, Any]) -> Path:
        generator_id = ProcessId("generator", self.config.process_id.split('_')[-1])
        value = self.get_process_result(input_data, generator_id)
        
        output_file = self.output_dir / f"value_{self.config.process_id}.txt"
        with open(output_file, 'w') as f:
            f.write(f"{value}\n")
            
        return output_file

def execute_workflow_process(args: Tuple[int, Optional[int], Path]) -> Dict[str, Any]:
    """Create and execute a single workflow process."""
    index, seed, output_dir = args
    
    workflow = create_workflow(
        max_threads=1,
        process_id=f"workflow_process_{index}"
    )
    
    generator = RandomNumberGenerator(index=index, seed=seed)
    writer = FileWriter(index=index, output_dir=output_dir)
    
    workflow.add_node(WorkflowNode(
        process=generator,
        dependencies=[],
        required=True
    ))
    
    workflow.add_node(WorkflowNode(
        process=writer,
        dependencies=[generator.config.process_id],
        required=True
    ))
    
    # Framework handles execution and result wrapping
    return workflow.execute()

class ParallelRandomWorkflowCPUBound(BaseProcess):
    def __init__(self, 
                 num_samples: int,
                 output_dir: Path,
                 max_parallel: Optional[int] = None,
                 seed: Optional[int] = None):
        super().__init__(
            config = ProcessConfig(
                process_type=ProcessType.PROCESS,
                process_id="parallel_random_workflow_cpu",
            )
        )
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.max_parallel = max_parallel or max(1, os.cpu_count() - 1)
        self.base_seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, input_data: Any) -> Dict[int, Path]:
        workflow = create_workflow(
            max_processes=self.max_parallel,
            process_id=f"parallel_workflow_{os.getpid()}"
        )
        
        workflow_args = [
            (i, self.base_seed, self.output_dir) 
            for i in range(self.num_samples)
        ]
        
        results = {}
        # Only need pools for parallel execution
        with workflow.get_pools() as (_, process_pool):
            for result in process_pool.map(execute_workflow_process, workflow_args):
                writer_id = f"writer_{result['index']}"
                if writer_id in result:
                    results[result['index']] = result[writer_id].data
        
        return results

async def main():
    # Create and configure the main workflow
    main_workflow = create_workflow(process_id="main_workflow")
    
    # Create and add the parallel process
    parallel_process = ParallelRandomWorkflowCPUBound(
        num_samples=3,
        output_dir=Path("random_outputs_cpu"),
        max_parallel=2,
        seed=42
    )
    
    main_workflow.add_node(WorkflowNode(
        process=parallel_process,
        dependencies=[],
        required=True
    ))
    
    # Framework handles execution and cleanup
    results = await main_workflow.execute()
    
    # Print results
    if results and parallel_process.config.process_id in results:
        output_files = results[parallel_process.config.process_id].data
        print(f"Generated files: {output_files}")

if __name__ == '__main__':
    asyncio.run(main())