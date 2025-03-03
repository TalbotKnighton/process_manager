import asyncio
from process_manager import wf1 as wf

async def main():
    # Create workflow with custom pool sizes
    with Workflow(max_processes=4, max_threads=10) as workflow:
        # CPU-intensive process
        data_processing = CPUIntensiveProcess(
            ProcessConfig(
                process_id="process_data",
                process_type=ProcessType.PROCESS,
                timeout=300
            )
        )
        
        # I/O-bound process
        file_handling = IOBoundProcess(
            ProcessConfig(
                process_id="handle_files",
                process_type=ProcessType.THREAD
            )
        )
        
        # Add nodes to workflow
        workflow.add_node(WorkflowNode(process=data_processing))
        workflow.add_node(WorkflowNode(
            process=file_handling,
            dependencies=["process_data"]
        ))
        
        # Execute workflow
        results = await workflow.execute({
            "process_data": initial_data
        })

if __name__ == "__main__":
    asyncio.run(main())