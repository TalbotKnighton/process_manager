"""Example of nested workflows for parallel random number generation."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from process_manager.workflow.core import Workflow, create_workflow
from process_manager.workflow.process import BaseProcess, ProcessConfig
from process_manager.workflow.workflow_types import (
    ProcessResult,
    ProcessType,
    WorkflowNode,
)
from process_manager.data_handlers.random_variables import NormalDistribution
from process_manager.data_handlers.values import NamedValue

class RandomNumberGenerator(BaseProcess):
    """Process that generates a random number using data_handlers."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(ProcessConfig(
            process_type=ProcessType.THREAD,
            process_id="random_generator"
        ))
        self.distribution = NormalDistribution(
            name="random_value",
            mu=0.0,
            sigma=1.0,
            seed=seed
        )
    
    async def execute(self, input_data: Any) -> float:
        """Async interface for the process."""
        return await self._run_threaded(input_data)
    
    def _sync_execute(self, input_data: Any) -> ProcessResult:
        """Synchronous implementation."""
        start_time = datetime.now()
        try:
            value = self.distribution.sample(size=1).item()
            end_time = datetime.now()
            return ProcessResult(
                success=True,
                data=value,
                execution_time=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            end_time = datetime.now()
            return ProcessResult(
                success=False,
                data=None,
                execution_time=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time,
                error=str(e),
                error_type=type(e).__name__
            )

class FileWriter(BaseProcess):
    """Process that writes a value to a file."""
    
    def __init__(self, output_dir: Path):
        super().__init__(ProcessConfig(
            process_type=ProcessType.THREAD,
            process_id="file_writer"
        ))
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def execute(self, input_data: Dict[str, Any]) -> Path:
        """Async interface for the process."""
        return await self._run_threaded(input_data)
    
    def _sync_execute(self, input_data: Dict[str, Any]) -> ProcessResult:
        """Synchronous implementation."""
        start_time = datetime.now()
        try:
            # Print debugging information
            print(f"Process ID: {self.config.process_id}")
            print(f"Input data keys: {input_data.keys()}")
            print(f"Input data content: {input_data}")  # Add this debug line
            
            # Get the index from the writer's process ID
            writer_index = self.config.process_id.split('_')[-1]
            generator_id = f"generator_{writer_index}"
            
            print(f"Looking for generator ID: {generator_id}")
            
            # Check if the generator data exists
            if generator_id not in input_data:
                raise ValueError(f"Generator ID {generator_id} not found in input data")
            
            # Get the value directly from the generator's output
            generator_output = input_data[generator_id]
            
            # If generator_output is already a float, use it directly
            if isinstance(generator_output, (float, int)):
                value = generator_output
            # If it's a ProcessResult, get the data from it
            elif hasattr(generator_output, 'data'):
                value = generator_output.data
            else:
                raise ValueError(f"Unexpected data format from {generator_id}: {type(generator_output)}")
            
            output_file = self.output_dir / f"value_{self.config.process_id}.txt"
            
            with open(output_file, 'w') as f:
                f.write(f"{value}\n")
                
            end_time = datetime.now()
            return ProcessResult(
                success=True,
                data=output_file,
                execution_time=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            end_time = datetime.now()
            print(f"Error in FileWriter: {str(e)}")
            return ProcessResult(
                success=False,
                data=None,
                execution_time=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time,
                error=str(e),
                error_type=type(e).__name__
            )

class ParallelRandomWorkflow(BaseProcess):
    """Process that runs multiple generate-and-save workflows in parallel."""
    
    def __init__(self, 
                 num_samples: int,
                 output_dir: Path,
                 max_parallel: int = 4,
                 seed: Optional[int] = None):
        super().__init__(ProcessConfig(
            process_type=ProcessType.ASYNC,
            process_id="parallel_random_workflow"
        ))
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.max_parallel = max_parallel
        self.base_seed = seed
    
    def _create_single_workflow(self, index: int) -> Workflow:
        """Create a workflow for generating and saving one random number."""
        workflow = create_workflow(max_threads=1)
        
        # Create processes with unique IDs
        generator = RandomNumberGenerator(
            seed=None if self.base_seed is None else self.base_seed + index
        )
        generator.config.process_id = f"generator_{index}"
        
        writer = FileWriter(self.output_dir)
        writer.config.process_id = f"writer_{index}"
        
        # Create nodes and add to workflow
        generator_node = WorkflowNode(
            process=generator,
            dependencies=[],
            required=True
        )
        
        writer_node = WorkflowNode(
            process=writer,
            dependencies=[generator.config.process_id],
            required=True,
            input_mapping={
                f"generator_{index}": generator.config.process_id
            }
        )
        
        # Add nodes in the correct order
        workflow.add_node(generator_node)
        workflow.add_node(writer_node)
        
        return workflow
    
    async def execute(self, input_data: Any) -> Dict[int, Path]:
        """Execute multiple workflows in parallel."""
        results = {}
        sem = asyncio.Semaphore(self.max_parallel)
        
        async def run_workflow(index: int):
            async with sem:
                workflow = self._create_single_workflow(index)
                print(f"Starting workflow {index}")
                workflow_results = await workflow.execute()
                print(f"Workflow {index} results: {workflow_results}")
                
                # Get the writer's result
                writer_id = f"writer_{index}"
                if writer_id in workflow_results and workflow_results[writer_id].success:
                    results[index] = workflow_results[writer_id].data
                else:
                    print(f"Failed to get results for workflow {index}")
        
        tasks = [run_workflow(i) for i in range(self.num_samples)]
        await asyncio.gather(*tasks)
        
        return results

async def main():
    """Example usage of parallel random workflow."""
    # Create and configure the main workflow
    main_workflow = create_workflow()
    
    # Create the parallel process
    parallel_process = ParallelRandomWorkflow(
        num_samples=10,
        output_dir=Path("random_outputs"),
        max_parallel=3,
        seed=42
    )
    
    # Create and add workflow node
    node = WorkflowNode(
        process=parallel_process,
        dependencies=[],
        required=True
    )
    main_workflow.add_node(node)
    
    # Execute workflow
    results = await main_workflow.execute()
    
    # Print results
    if parallel_process.config.process_id in results and results[parallel_process.config.process_id].success:
        output_files = results[parallel_process.config.process_id].data
        print(f"Generated files: {output_files}")

# if __name__ == "__main__":
#     asyncio.run(main())

# from multiprocessing import Pool, cpu_count
# import concurrent.futures

# class ParallelRandomWorkflowCPUBound(BaseProcess):
#     """Process that runs multiple CPU-bound generate-and-save workflows in parallel using multiprocessing."""
    
#     def __init__(self, 
#                  num_samples: int,
#                  output_dir: Path,
#                  max_parallel: Optional[int] = None,
#                  seed: Optional[int] = None):
#         super().__init__(ProcessConfig(
#             process_type=ProcessType.PROCESS,  # Use PROCESS type for CPU-bound work
#             process_id="parallel_random_workflow_cpu"
#         ))
#         self.num_samples = num_samples
#         self.output_dir = output_dir
#         self.max_parallel = max_parallel or max(1, cpu_count() - 1)  # Default to CPU count - 1
#         self.base_seed = seed
    
#     def _create_single_workflow(self, index: int) -> Workflow:
#         """Create a workflow for generating and saving one random number."""
#         workflow = create_workflow(max_threads=1)
        
#         # Create processes with unique IDs
#         generator = RandomNumberGenerator(
#             seed=None if self.base_seed is None else self.base_seed + index
#         )
#         generator.config.process_id = f"generator_{index}"
        
#         writer = FileWriter(self.output_dir)
#         writer.config.process_id = f"writer_{index}"
        
#         # Create nodes and add to workflow
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
        
#         # Add nodes in the correct order
#         workflow.add_node(generator_node)
#         workflow.add_node(writer_node)
        
#         return workflow

#     def _run_single_workflow(self, index: int) -> Dict[str, Any]:
#         """Execute a single workflow synchronously."""
#         try:
#             workflow = self._create_single_workflow(index)
#             # Use synchronous execution for the subprocess
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             try:
#                 results = loop.run_until_complete(workflow.execute())
#             finally:
#                 loop.close()
            
#             return {
#                 'index': index,
#                 'success': True,
#                 'results': results
#             }
#         except Exception as e:
#             return {
#                 'index': index,
#                 'success': False,
#                 'error': str(e)
#             }

#     def _sync_execute(self, input_data: Any) -> ProcessResult:
#         """Synchronous implementation using multiprocessing."""
#         start_time = datetime.now()
#         results = {}
        
#         try:
#             # Use ProcessPoolExecutor for true parallel execution
#             with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_parallel) as executor:
#                 # Submit all workflows
#                 future_to_index = {
#                     executor.submit(self._run_single_workflow, i): i 
#                     for i in range(self.num_samples)
#                 }
                
#                 # Collect results as they complete
#                 for future in concurrent.futures.as_completed(future_to_index):
#                     index = future_to_index[future]
#                     try:
#                         workflow_result = future.result()
#                         if workflow_result['success']:
#                             writer_id = f"writer_{index}"
#                             if (writer_id in workflow_result['results'] and 
#                                 workflow_result['results'][writer_id].success):
#                                 results[index] = workflow_result['results'][writer_id].data
#                             else:
#                                 print(f"Failed to get results for workflow {index}")
#                         else:
#                             print(f"Workflow {index} failed: {workflow_result.get('error')}")
#                     except Exception as e:
#                         print(f"Workflow {index} generated an exception: {str(e)}")

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
#             return ProcessResult(
#                 success=False,
#                 data=None,
#                 execution_time=(end_time - start_time).total_seconds(),
#                 start_time=start_time,
#                 end_time=end_time,
#                 error=str(e),
#                 error_type=type(e).__name__
#             )

#     async def execute(self, input_data: Any) -> Dict[int, Path]:
#         """Async interface that runs the sync implementation."""
#         return await self._run_process(input_data)

# # Example usage:
# async def main_cpu_bound():
#     """Example usage of parallel random workflow with CPU-bound processes."""
#     # Create and configure the main workflow
#     main_workflow = create_workflow()
    
#     # Create the parallel process
#     parallel_process = ParallelRandomWorkflowCPUBound(
#         num_samples=10,
#         output_dir=Path("random_outputs_cpu"),
#         max_parallel=3,  # Use 3 processes
#         seed=42
#     )
    
#     # Create and add workflow node
#     node = WorkflowNode(
#         process=parallel_process,
#         dependencies=[],
#         required=True
#     )
#     main_workflow.add_node(node)
    
#     # Execute workflow
#     results = await main_workflow.execute()
    
#     # Print results
#     if parallel_process.config.process_id in results and results[parallel_process.config.process_id].success:
#         output_files = results[parallel_process.config.process_id].data
#         print(f"Generated files: {output_files}")

# if __name__ == "__main__":
#     # Run both versions for comparison
#     print("Running CPU-bound version...")
#     asyncio.run(main_cpu_bound())

from multiprocessing import Pool, cpu_count
import concurrent.futures
import functools

def run_single_workflow_wrapper(workflow_creator, index: int) -> Dict[str, Any]:
    """Standalone function to run a single workflow."""
    try:
        # Create a new workflow for this process
        workflow = workflow_creator(index)
        
        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(workflow.execute())
            loop.close()
            
            return {
                'index': index,
                'success': True,
                'results': results
            }
        finally:
            loop.close()
    except Exception as e:
        return {
            'index': index,
            'success': False,
            'error': str(e)
        }

class ParallelRandomWorkflowCPUBound(BaseProcess):
    """Process that runs multiple CPU-bound generate-and-save workflows in parallel using multiprocessing."""
    
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
        self.max_parallel = max_parallel or max(1, cpu_count() - 1)
        self.base_seed = seed
    
    def _create_single_workflow(self, index: int) -> Workflow:
        """Create a workflow for generating and saving one random number."""
        workflow = create_workflow(max_threads=1)
        
        # Create processes with unique IDs
        generator = RandomNumberGenerator(
            seed=None if self.base_seed is None else self.base_seed + index
        )
        generator.config.process_id = f"generator_{index}"
        
        writer = FileWriter(self.output_dir)
        writer.config.process_id = f"writer_{index}"
        
        # Create nodes and add to workflow
        generator_node = WorkflowNode(
            process=generator,
            dependencies=[],
            required=True
        )
        
        writer_node = WorkflowNode(
            process=writer,
            dependencies=[generator.config.process_id],
            required=True,
            input_mapping={
                f"generator_{index}": generator.config.process_id
            }
        )
        
        # Add nodes in the correct order
        workflow.add_node(generator_node)
        workflow.add_node(writer_node)
        
        return workflow

    def _sync_execute(self, input_data: Any) -> ProcessResult:
        """Synchronous implementation using multiprocessing."""
        start_time = datetime.now()
        results = {}
        
        try:
            # Create a pool with the specified number of processes
            with Pool(processes=self.max_parallel) as pool:
                # Create partial function with fixed arguments
                workflow_runner = functools.partial(
                    run_single_workflow_wrapper,
                    self._create_single_workflow
                )
                
                # Map the workflow runner over the indices
                workflow_results = pool.map(
                    workflow_runner,
                    range(self.num_samples)
                )
                
                # Process results
                for result in workflow_results:
                    if result['success']:
                        index = result['index']
                        writer_id = f"writer_{index}"
                        if (writer_id in result['results'] and 
                            result['results'][writer_id].success):
                            results[index] = result['results'][writer_id].data
                        else:
                            print(f"Failed to get results for workflow {index}")
                    else:
                        print(f"Workflow {result['index']} failed: {result.get('error')}")

            end_time = datetime.now()
            return ProcessResult(
                success=True,
                data=results,
                execution_time=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            end_time = datetime.now()
            return ProcessResult(
                success=False,
                data=None,
                execution_time=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time,
                error=str(e),
                error_type=type(e).__name__
            )

    async def execute(self, input_data: Any) -> Dict[int, Path]:
        """Async interface that runs the sync implementation."""
        return await self._run_process(input_data)

# Example usage remains the same
async def main_cpu_bound():
    """Example usage of parallel random workflow with CPU-bound processes."""
    # Create and configure the main workflow
    main_workflow = create_workflow()
    
    # Create the parallel process
    parallel_process = ParallelRandomWorkflowCPUBound(
        num_samples=10,
        output_dir=Path("random_outputs_cpu"),
        max_parallel=3,
        seed=42
    )
    
    # Create and add workflow node
    node = WorkflowNode(
        process=parallel_process,
        dependencies=[],
        required=True
    )
    main_workflow.add_node(node)
    
    # Execute workflow
    results = await main_workflow.execute()
    
    # Print results
    if parallel_process.config.process_id in results and results[parallel_process.config.process_id].success:
        output_files = results[parallel_process.config.process_id].data
        print(f"Generated files: {output_files}")

if __name__ == "__main__":
    print("Running CPU-bound version...")
    asyncio.run(main_cpu_bound())