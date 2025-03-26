"""Debugging version of parallel flow execution."""
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from flow.core.flow import Flow, FlowConfig
from flow.core.types import FlowType
from datetime import datetime
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# ============= Data Models =============

class NumberGeneratorInput(BaseModel):
    """Input validation for NumberGenerator."""
    start: int = Field(
        default=1,
        gt=0,
        description="Start of number sequence"
    )
    end: int = Field(
        ...,  # Required field
        gt=0,
        description="End of number sequence"
    )
    
    model_config = ConfigDict(extra='forbid')
    
    @property
    def sequence_length(self) -> int:
        return self.end - self.start + 1
    
    # def model_validate(self, *args, **kwargs):
    #     validated = type(self).model_validate(*args, **kwargs)
    #     if validated.start >= validated.end:
    #         raise ValueError("start must be less than end")
    #     return validated

class NumberGeneratorOutput(BaseModel):
    """Output validation for NumberGenerator."""
    numbers: List[int] = Field(
        ...,
        min_items=1,
        description="Generated sequence of numbers"
    )

class NumberMultiplierInput(BaseModel):
    """Input validation for NumberMultiplier."""
    numbers: List[int] = Field(
        ...,
        min_items=1,
        description="Numbers to multiply"
    )
    factor: float = Field(
        default=2.0,
        gt=0,
        description="Multiplication factor"
    )
    
    model_config = ConfigDict(extra='forbid')

class NumberMultiplierOutput(BaseModel):
    """Output validation for NumberMultiplier."""
    result: List[float] = Field(
        ...,
        min_items=1,
        description="Multiplied numbers"
    )
    stats: Dict[str, float] = Field(
        default_factory=dict,
        description="Statistics about the results"
    )

# ============= Processors =============

class NumberGenerator(BaseModel):
    """Generates a sequence of numbers."""
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"NumberGenerator processing with input: {input_data}")
        
        # Validate input
        print(f'\n\n\n{input_data = }\n\n\n')
        validated_input = NumberGeneratorInput.model_validate(input_data)
        
        # Generate numbers immediately (no sleep for debugging)
        numbers = list(range(validated_input.start, validated_input.end + 1))
        
        # Validate output
        output = NumberGeneratorOutput(numbers=numbers)
        logger.debug(f"NumberGenerator produced: {output}")
        
        return output.model_dump()

class NumberMultiplier(BaseModel):
    """Multiplies numbers by a factor."""
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"NumberMultiplier processing with input: {input_data}")
        
        # Validate input
        validated_input = NumberMultiplierInput.model_validate(input_data)
        
        # Process immediately (no sleep for debugging)
        result = [n * validated_input.factor for n in validated_input.numbers]
        
        # Calculate stats
        stats = {
            "mean": sum(result) / len(result),
            "min": min(result),
            "max": max(result)
        }
        
        # Validate output
        output = NumberMultiplierOutput(result=result, stats=stats)
        logger.debug(f"NumberMultiplier produced: {output}")
        
        return output.model_dump()

async def create_parallel_flows(num_flows: int = 3) -> List[Flow]:
    """Create multiple parallel flows with dependencies."""
    flows = []
    
    for i in range(num_flows):
        logger.debug(f"Creating flow pair {i}")
        
        # Create generator flow
        generator = NumberGenerator()
        gen_flow = Flow(
            callable=generator,
            config=FlowConfig(
                name=f"Generator-{i}",
                flow_type=FlowType.INLINE,  # Changed to INLINE for debugging
                timeout=10.0
            )
        )
        logger.debug(f"Created generator flow with ID: {gen_flow.process_id}")
        
        # Create multiplier flow
        multiplier = NumberMultiplier()
        mult_flow = Flow(
            callable=multiplier,
            config=FlowConfig(
                name=f"Multiplier-{i}",
                flow_type=FlowType.INLINE,  # Changed to INLINE for debugging
                timeout=10.0
            )
        )
        logger.debug(f"Created multiplier flow with ID: {mult_flow.process_id}")
        
        # Register dependency
        logger.debug(f"Registering dependency between {gen_flow.process_id} and {mult_flow.process_id}")
        mult_flow.register_to(
            gen_flow,
            required_deps=[gen_flow.process_id]
        )
        
        flows.extend([gen_flow, mult_flow])
    
    return flows

async def execute_parallel_flows(flows: List[Flow]) -> None:
    """Execute generator flows in parallel."""
    try:
        # Get all generator flows (even indexed flows)
        generator_flows = flows[::2]
        logger.debug(f"Preparing to execute {len(generator_flows)} generator flows")
        
        # Prepare execution data for each generator
        execution_data = [
            {
                "start": i * 10 + 1,
                "end": (i + 1) * 10,
                "factor": i + 2
            }
            for i in range(len(generator_flows))
        ]
        
        # Execute all generator flows in parallel
        start_time = time.time()
        logger.info("Starting parallel execution of generators")
        
        # Execute one by one first for debugging
        results = []
        for flow, data in zip(generator_flows, execution_data):
            logger.debug(f"Executing flow {flow.config.name} with data: {data}")
            try:
                result = await flow.execute(data)
                results.append(result)
                logger.debug(f"Flow {flow.config.name} completed with result: {result}")
            except Exception as e:
                logger.error(f"Flow {flow.config.name} failed: {e}", exc_info=True)
                results.append(e)
        
        end_time = time.time()
        logger.info(f"All flows completed in {end_time - start_time:.2f} seconds")
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Generator-{i} failed: {result}")
            else:
                logger.info(f"Generator-{i} completed: {result.output}")
                
                # Get corresponding multiplier result
                mult_flow = flows[i*2 + 1]
                mult_result = await mult_flow.context.results_manager.get_result(
                    mult_flow.process_id
                )
                if mult_result:
                    logger.info(f"Multiplier-{i} completed: {mult_result.output}")
                else:
                    logger.warning(f"No result found for Multiplier-{i}")
        
    except Exception as e:
        logger.error(f"Error in parallel execution: {e}", exc_info=True)
        raise

async def run_parallel_example():
    """Run the complete parallel flow example."""
    flows = []
    try:
        logger.info("Starting parallel flow example")
        
        # Create flows
        flows = await create_parallel_flows(num_flows=3)
        logger.info(f"Created {len(flows)} flows")
        
        # Execute flows
        await execute_parallel_flows(flows)
        
        logger.info("Parallel execution completed successfully")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up flows")
        for flow in flows:
            if flow and flow.context:
                try:
                    flow.context.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up flow: {e}")
        logger.info("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(run_parallel_example())