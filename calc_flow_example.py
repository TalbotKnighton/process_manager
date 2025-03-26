"""Simple example of nested flows with visualization."""
import asyncio
from typing import Dict, Any
from pydantic import BaseModel, Field
import logging
from flow.core.flow import Flow, FlowConfig
from flow.core.types import FlowType
from flow.visualization.graph import FlowVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumberGenerator(BaseModel):
    """Generates a sequence of numbers."""
    class Input(BaseModel):
        start: int = Field(default=1)
        end: int = Field(default=10)

    class Output(BaseModel):
        numbers: list[int]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start = input_data.get('start', 1)
        end = input_data.get('end', 10)
        numbers = list(range(start, end + 1))
        return self.Output(numbers=numbers).model_dump()

class NumberMultiplier(BaseModel):
    """Multiplies each number by a factor."""
    class Input(BaseModel):
        numbers: list[int]
        factor: int = Field(default=2)

    class Output(BaseModel):
        result: list[int]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        numbers = input_data['numbers']
        factor = input_data.get('factor', 2)
        result = [n * factor for n in numbers]
        return self.Output(result=result).model_dump()

class NumberSummarizer(BaseModel):
    """Calculates sum and average of numbers."""
    class Input(BaseModel):
        numbers: list[int]

    class Output(BaseModel):
        sum: int
        average: float

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        numbers = input_data['numbers']
        total = sum(numbers)
        avg = total / len(numbers)
        return self.Output(sum=total, average=avg).model_dump()

async def run_example():
    # Create processors
    generator = NumberGenerator()
    multiplier = NumberMultiplier()
    summarizer = NumberSummarizer()

    # Create flows
    generator_flow = Flow(
        callable=generator,
        config=FlowConfig(
            name="Number Generator",
            flow_type=FlowType.PROCESS
        )
    )

    multiplier_flow = Flow(
        callable=multiplier,
        config=FlowConfig(
            name="Number Multiplier",
            flow_type=FlowType.PROCESS
        )
    )

    summarizer_flow = Flow(
        callable=summarizer,
        config=FlowConfig(
            name="Number Summarizer",
            flow_type=FlowType.PROCESS
        )
    )

    # Register dependencies
    multiplier_flow.register_to(
        generator_flow,
        required_deps=[generator_flow.process_id]
    )
    
    summarizer_flow.register_to(
        multiplier_flow,
        required_deps=[multiplier_flow.process_id]
    )

    # Create and display visualization
    logger.info("Creating flow visualization...")
    visualizer = FlowVisualizer(generator_flow)
    
    # Also save Mermaid diagram
    mermaid_diagram = visualizer.to_mermaid()
    with open("simple_flow.mmd", "w") as f:
        f.write(mermaid_diagram)
    logger.info("Flow diagram saved to simple_flow.mmd")

    # Execute the flow
    logger.info("Executing flow...")
    result = await generator_flow.execute({
        "start": 1,
        "end": 5,
        "factor": 3
    })

    # Print results
    logger.info("Flow results:")
    print(result.output)

    # Show interactive Plotly visualization
    flow_chart = visualizer.to_plotly()
    flow_chart.show()

async def main():
    try:
        await run_example()
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)
    finally:
        logger.info("Example completed")

if __name__ == "__main__":
    asyncio.run(main())
