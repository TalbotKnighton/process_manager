"""Debug test for parallel flow data passing."""
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from flow.core.flow import Flow, FlowConfig, FlowTree
from flow.core.types import FlowType
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeneratorInput(BaseModel):
    """Input for number generator."""
    start: int = Field(..., description="Start number")
    end: int = Field(..., description="End number")

class GeneratorOutput(BaseModel):
    """Output from number generator."""
    numbers: List[int] = Field(..., description="Generated numbers")

def generator(input_data: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug(f"Generator processing with input: {input_data}")
    validated = GeneratorInput.model_validate(input_data)
    numbers = list(range(validated.start, validated.end + 1))
    output = GeneratorOutput(numbers=numbers)
    logger.debug(f"Generator produced output: {output}")
    return output.model_dump()

class MultiplierInput(BaseModel):
    """Input for number multiplier."""
    numbers: List[int] = Field(..., description="Numbers to multiply")
    factor: float = Field(..., description="Multiplication factor")

class MultiplierOutput(BaseModel):
    """Output from number multiplier."""
    result: List[float] = Field(..., description="Multiplied numbers")

def multiplier(input_data: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug(f"Multiplier processing with input: {input_data}")
    validated = MultiplierInput.model_validate(input_data)
    result = [n * validated.factor for n in validated.numbers]
    output = MultiplierOutput(result=result)
    logger.debug(f"Multiplier produced output: {output}")
    return output.model_dump()

async def run_example():
    # Create registry
    registry = FlowTree(max_workers=4)

    # Create flows with simplified initialization
    generator1 = Flow(
        name="Generator1",
        callable=generator,
        flow_tree=registry
    )

    multiplier1 = Flow(
        name="Multiplier1",
        callable=multiplier,
        flow_tree=registry
    )

    # Add prerequisites
    multiplier1.add_prerequisite(generator1)

    # Execute all flows and wait for results
    logger.info("Starting flow execution")
    results = await registry.execute_all({
        "start": 1,
        "end": 5,
        "factor": 2
    })
    logger.info("Flow execution completed")

    # Verify we have results
    logger.info(f"Number of results: {len(results)}")
    for flow_id, result in results.items():
        flow = registry._flows[flow_id]
        logger.info(f"Result for {flow.config.name}: {result.output}")

    # Get individual results
    gen_result = await registry.get_result(generator1.id)
    if gen_result:
        print(f"\nGenerator Result: {gen_result.output}")
    else:
        print("\nNo generator result found!")

    mult_results = await registry.get_results_by_name("Multiplier1")
    if mult_results:
        print(f"\nMultiplier Results: {mult_results[0].output}")
    else:
        print("\nNo multiplier results found!")

    # Show execution order
    print("\nResults in execution order:")
    for flow_id in registry._completed_flows:
        flow = registry._flows[flow_id]
        result = await registry.get_result(flow_id)
        if result:
            print(f"{flow.config.name}: {result.output}")

    # Show execution summary
    summary = await registry.get_execution_summary()
    print("\nExecution Summary:")
    print(f"Total flows: {summary['total_flows']}")
    print(f"Completed flows: {summary['completed_flows']}")
    print(f"Failed flows: {summary['failed_flows']}")
    print(f"Success rate: {summary['success_rate'] * 100:.1f}%")

if __name__ == "__main__":
    asyncio.run(run_example())