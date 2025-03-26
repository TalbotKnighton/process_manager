"""Debug test for parallel flow data passing."""
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from flow.core.flow import Flow, FlowConfig
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

class Generator(BaseModel):
    """Generates a sequence of numbers."""
    Input = GeneratorInput
    Output = GeneratorOutput

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Generator processing with input: {input_data}")
        validated = self.Input.model_validate(input_data)
        numbers = list(range(validated.start, validated.end + 1))
        output = self.Output(numbers=numbers)
        logger.debug(f"Generator produced output: {output}")
        return output.model_dump()

class MultiplierInput(BaseModel):
    """Input for number multiplier."""
    numbers: List[int] = Field(..., description="Numbers to multiply")
    factor: float = Field(..., description="Multiplication factor")

class MultiplierOutput(BaseModel):
    """Output from number multiplier."""
    result: List[float] = Field(..., description="Multiplied numbers")

class Multiplier(BaseModel):
    """Multiplies numbers by a factor."""
    Input = MultiplierInput
    Output = MultiplierOutput

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Multiplier processing with input: {input_data}")
        validated = self.Input.model_validate(input_data)
        result = [n * validated.factor for n in validated.numbers]
        output = self.Output(result=result)
        logger.debug(f"Multiplier produced output: {output}")
        return output.model_dump()
async def test_parallel_data_flow():
    """Test parallel execution with explicit data flow tracking."""
    try:
        logger.info("Creating parallel flows")
        main_flow = Flow(
            callable=lambda i: None,
            config=FlowConfig(
                name=f"RootFlow",
                flow_type=FlowType.INLINE,
                process_id='RootFlow',
            )
        )
        generator = Flow(
            callable=lambda i: print(f'\n\n\n{i=}\n\n\n'),
            config=FlowConfig(
                name=f'generator',
                flow_type=FlowType.INLINE,
                process_id='generator',
            )
        ).register_to(main_flow)
        await main_flow.execute(None)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(test_parallel_data_flow())