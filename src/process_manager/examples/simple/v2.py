"""Example of nested workflows for parallel random number generation."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, Dict, Any
import os
import traceback
from typing import Tuple

from process_manager.workflow.core import Workflow, create_workflow
from process_manager.workflow.process import BaseProcess, ProcessConfig, ProcessId
from process_manager.workflow.workflow_types import (
    ProcessResult,
    ProcessType,
    WorkflowNode,
)
from process_manager.data_handlers.random_variables import NormalDistribution
from process_manager.data_handlers.values import NamedValue

class RandomNumberGenerator(BaseProcess):
    def __init__(self, index: int, seed: Optional[int] = None):
        super().__init__(
            process_type=ProcessType.THREAD,
            base_name="generator",
            index=index
        )
        self.distribution = NormalDistribution(
            name="random_value",
            mu=0.0,
            sigma=1.0,
            seed=seed
        )
    
    def process(self, input_data: Any) -> float:
        """Just the core logic without error handling."""
        return self.distribution.sample(size=1).item()

class FileWriter(BaseProcess):
    def __init__(self, index: int, output_dir: Path):
        super().__init__(
            process_type=ProcessType.THREAD,
            base_name="writer",
            index=index
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, input_data: Dict[str, Any]) -> Path:
        """Just the core logic without error handling."""
        generator_id = ProcessId("generator", self.config.process_id.split('_')[-1])
        value = self.get_process_result(input_data, generator_id)
        
        output_file = self.output_dir / f"value_{self.config.process_id}.txt"
        with open(output_file, 'w') as f:
            f.write(f"{value}\n")
            
        return output_file
def execute_workflow_process(args: Tuple[int, Optional[int], Path]) -> Dict[str, Any]:
    """Create and execute a single workflow process."""
    index, seed, output_dir = args
    
    # Create workflow
    workflow = create_workflow(
        max_threads=1,
        process_id=f"workflow_process_{index}"
    )
    
    # Create and add nodes
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
    
    # The framework handles event loop and execution
    workflow_results = workflow.execute()
    return {
        'index': index,
        'success': True,
        'results': workflow_results
    }

class ParallelRandomWorkflowCPUBound(BaseProcess):
    def __init__(self, 
                 num_samples: int,
                 output_dir: Path,
                 max_parallel: Optional[int] = None,
                 seed: Optional[int] = None):
        super().__init__(ProcessConfig(
            process_type=ProcessType.PROCESS,
            process_id="parallel_random_workflow_cpu"
        ))
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.max_parallel = max_parallel or max(1, os.cpu_count() - 1)
        self.base_seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, input_data: Any) -> Dict[int, Path]:
        # Create workflow for parallel execution
        workflow = create_workflow(
            max_processes=self.max_parallel,
            process_id=f"parallel_workflow_{os.getpid()}"
        )
        
        # Prepare arguments for each workflow
        workflow_args = [
            (i, self.base_seed, self.output_dir) 
            for i in range(self.num_samples)
        ]
        
        results = {}
        # Execute workflows in parallel
        with workflow.get_pools() as (_, process_pool):
            workflow_results = list(process_pool.map(
                execute_workflow_process,
                workflow_args
            ))
        
        # Process results
        for result in workflow_results:
            if result['success']:
                index = result['index']
                writer_id = f"writer_{index}"
                if writer_id in result['results']:
                    results[index] = result['results'][writer_id].data
        
        return results
async def main_cpu_bound():
    """Example usage of parallel random workflow with CPU-bound processes."""
    try:
        # Create and configure the main workflow
        main_workflow = create_workflow(
            process_id="main_workflow"
        )
        
        # Create the parallel process
        parallel_process = ParallelRandomWorkflowCPUBound(
            num_samples=3,  # Reduced for testing
            output_dir=Path("random_outputs_cpu"),
            max_parallel=2,
            seed=42
        )
        
        # Create and add workflow node
        node = WorkflowNode(
            process=parallel_process,
            dependencies=[],
            required=True
        )
        main_workflow.add_node(node)
        
        try:
            # Execute workflow
            print("Starting main workflow execution...")
            results = await main_workflow.execute()
            
            # Print results
            if parallel_process.config.process_id in results and results[parallel_process.config.process_id].success:
                output_files = results[parallel_process.config.process_id].data
                print(f"Generated files: {output_files}")
            else:
                print("Workflow execution failed")
                
        finally:
            main_workflow.shutdown()  # Clean up main workflow pools
            
    except Exception as e:
        print("Error in main_cpu_bound:")
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == "__main__":
    print(f"Starting script in process {os.getpid()}")
    print("Running CPU-bound version...")
    asyncio.run(main_cpu_bound())

# # Module-level functions for multiprocessing
# def create_and_run_workflow(args: Tuple[int, Optional[int], Path]) -> Dict[str, Any]:
#     """Standalone function to create and run a workflow."""
#     index, seed, output_dir = args
#     try:
#         # Create workflow components
#         workflow = create_workflow(max_threads=1)
        
#         # Create generator
#         generator = RandomNumberGenerator(
#             seed=None if seed is None else seed + index
#         )
#         generator.config.process_id = f"generator_{index}"
        
#         # Create writer
#         writer = FileWriter(output_dir)
#         writer.config.process_id = f"writer_{index}"
        
#         # Create nodes
#         generator_node = WorkflowNode(
#             process=generator,
#             dependencies=[],
#             required=True
#         )
        
#         writer_node = WorkflowNode(
#             process=writer,
#             dependencies=[generator.config.process_id],
#             required=True,
#             input_mapping={
#                 f"generator_{index}": generator.config.process_id
#             }
#         )
        
#         # Add nodes to workflow
#         workflow.add_node(generator_node)
#         workflow.add_node(writer_node)
        
#         # Create and run event loop
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         try:
#             results = loop.run_until_complete(workflow.execute())
#             return {
#                 'index': index,
#                 'success': True,
#                 'results': results
#             }
#         finally:
#             loop.close()
            
#     except Exception as e:
#         print(f"Error in workflow {index}: {str(e)}")
#         traceback.print_exc()
#         return {
#             'index': index,
#             'success': False,
#             'error': str(e)
#         }
# import traceback
# import sys
# import os
# from typing import Optional, Dict, Any

# class ParallelRandomWorkflowCPUBound(BaseProcess):
#     """Process that runs multiple CPU-bound generate-and-save workflows in parallel."""
    
#     def __init__(self, 
#                  num_samples: int,
#                  output_dir: Path,
#                  max_parallel: Optional[int] = None,
#                  seed: Optional[int] = None):
#         super().__init__(ProcessConfig(
#             process_type=ProcessType.PROCESS,  # Try switching to THREAD to test
#             process_id="parallel_random_workflow_cpu"
#         ))
#         self.num_samples = num_samples
#         self.output_dir = output_dir
#         self.max_parallel = max_parallel or max(1, cpu_count() - 1)
#         self.base_seed = seed
    
#     def _sync_execute(self, input_data: Any) -> ProcessResult:
#         """Synchronous implementation."""
#         start_time = datetime.now()
#         results = {}
        
#         try:
#             print(f"Starting _sync_execute in process {os.getpid()}")
            
#             # Create the workflows sequentially first (for testing)
#             workflows = []
#             for i in range(self.num_samples):
#                 try:
#                     workflow = create_workflow(max_threads=1)
                    
#                     # Create generator
#                     generator = RandomNumberGenerator(
#                         seed=None if self.base_seed is None else self.base_seed + i
#                     )
#                     generator.config.process_id = f"generator_{i}"
                    
#                     # Create writer
#                     writer = FileWriter(self.output_dir)
#                     writer.config.process_id = f"writer_{i}"
                    
#                     # Create nodes
#                     generator_node = WorkflowNode(
#                         process=generator,
#                         dependencies=[],
#                         required=True
#                     )
                    
#                     writer_node = WorkflowNode(
#                         process=writer,
#                         dependencies=[generator.config.process_id],
#                         required=True,
#                         input_mapping={
#                             f"generator_{i}": generator.config.process_id
#                         }
#                     )
                    
#                     # Add nodes
#                     workflow.add_node(generator_node)
#                     workflow.add_node(writer_node)
                    
#                     workflows.append((i, workflow))
#                     print(f"Successfully created workflow {i}")
                    
#                 except Exception as e:
#                     print(f"Error creating workflow {i}:")
#                     print("".join(traceback.format_exception(*sys.exc_info())))
#                     raise
            
#             # Execute workflows sequentially first (for testing)
#             print("Starting workflow execution...")
#             for i, workflow in workflows:
#                 try:
#                     print(f"Executing workflow {i}")
#                     # Create new event loop for each workflow
#                     loop = asyncio.new_event_loop()
#                     asyncio.set_event_loop(loop)
#                     try:
#                         workflow_results = loop.run_until_complete(workflow.execute())
#                         print(f"Workflow {i} results: {workflow_results}")
                        
#                         writer_id = f"writer_{i}"
#                         if writer_id in workflow_results and workflow_results[writer_id].success:
#                             results[i] = workflow_results[writer_id].data
#                         else:
#                             print(f"Failed to get results for workflow {i}")
#                     finally:
#                         loop.close()
                        
#                 except Exception as e:
#                     print(f"Error executing workflow {i}:")
#                     print("".join(traceback.format_exception(*sys.exc_info())))
#                     raise
            
#             end_time = datetime.now()
#             return ProcessResult(
#                 success=True,
#                 data=results,
#                 execution_time=(end_time - start_time).total_seconds(),
#                 start_time=start_time,
#                 end_time=end_time
#             )
            
#         except Exception as e:
#             end_time = datetime.now()
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             formatted_tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
#             print("Error in _sync_execute:")
#             print(formatted_tb)
            
#             return ProcessResult(
#                 success=False,
#                 data=None,
#                 execution_time=(end_time - start_time).total_seconds(),
#                 start_time=start_time,
#                 end_time=end_time,
#                 error=str(e),
#                 error_type=type(e).__name__,
#                 traceback=formatted_tb
#             )

#     async def execute(self, input_data: Any) -> Dict[int, Path]:
#         """Async interface that runs the sync implementation."""
#         try:
#             print(f"Starting execute in process {os.getpid()}")
#             return await self._run_process(input_data)
#         except Exception as e:
#             print("Error in execute:")
#             print("".join(traceback.format_exception(*sys.exc_info())))
#             raise

# async def main_cpu_bound():
#     """Example usage of parallel random workflow with CPU-bound processes."""
#     try:
#         print(f"Starting main_cpu_bound in process {os.getpid()}")
        
#         # Create and configure the main workflow
#         main_workflow = create_workflow()
        
#         # Create the parallel process
#         parallel_process = ParallelRandomWorkflowCPUBound(
#             num_samples=3,  # Reduced for testing
#             output_dir=Path("random_outputs_cpu"),
#             max_parallel=2,
#             seed=42
#         )
        
#         # Create and add workflow node
#         node = WorkflowNode(
#             process=parallel_process,
#             dependencies=[],
#             required=True
#         )
#         main_workflow.add_node(node)
        
#         # Execute workflow
#         print("Starting main workflow execution...")
#         results = await main_workflow.execute()
        
#         # Print results
#         if parallel_process.config.process_id in results and results[parallel_process.config.process_id].success:
#             output_files = results[parallel_process.config.process_id].data
#             print(f"Generated files: {output_files}")
#         else:
#             print("Workflow execution failed")
#             if parallel_process.config.process_id in results:
#                 result = results[parallel_process.config.process_id]
#                 print(f"Error: {result.error}")
#                 print(f"Error type: {result.error_type}")
#                 if hasattr(result, 'traceback'):
#                     print(f"Traceback: {result.traceback}")
                    
#     except Exception as e:
#         print("Error in main_cpu_bound:")
#         print("".join(traceback.format_exception(*sys.exc_info())))
#         raise

# if __name__ == "__main__":
#     print(f"Starting script in process {os.getpid()}")
#     print("Running CPU-bound version...")
#     asyncio.run(main_cpu_bound())