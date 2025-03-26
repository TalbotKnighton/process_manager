"""Example usage of the Flow system."""
from typing import Dict, Any, List
import asyncio
from pydantic import BaseModel, Field

from flow.core.flow import Flow, FlowConfig
from flow.core.types import FlowType

class DataLoader(BaseModel):
    """Example data loader processor."""
    class Input(BaseModel):
        file_path: str
        batch_size: int = Field(default=100)

    class Output(BaseModel):
        data: List[float]
        metadata: Dict[str, Any]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Validate input
        validated_input = self.Input(**input_data)
        
        # Simulate loading data
        data = [float(i) for i in range(validated_input.batch_size)]
        metadata = {"source": validated_input.file_path, "size": len(data)}
        
        # Validate and return output
        output = self.Output(data=data, metadata=metadata)
        return output.model_dump()

class DataValidator(BaseModel):
    """Example data validator processor."""
    class Input(BaseModel):
        data: List[float]
        threshold: float = Field(default=0.0)

    class Output(BaseModel):
        valid_data: List[float]
        stats: Dict[str, float]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated_input = self.Input(**input_data)
        
        # Filter and compute stats
        valid_data = [x for x in validated_input.data if x > validated_input.threshold]
        stats = {
            "mean": sum(valid_data) / len(valid_data) if valid_data else 0,
            "count": len(valid_data)
        }
        
        output = self.Output(valid_data=valid_data, stats=stats)
        return output.model_dump()

class DataProcessor(BaseModel):
    """Example data processor."""
    class Input(BaseModel):
        data: List[float]
        valid_data: Optional[List[float]] = None  # Optional from validator
        multiplier: float = Field(default=2.0)

    class Output(BaseModel):
        processed_data: List[float]
        stats: Dict[str, float]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated_input = self.Input(**input_data)
        
        # Use validated data if available, otherwise use raw data
        data_to_process = validated_input.valid_data or validated_input.data
        
        # Process data
        processed_data = [x * validated_input.multiplier for x in data_to_process]
        stats = {
            "mean": sum(processed_data) / len(processed_data),
            "min": min(processed_data),
            "max": max(processed_data)
        }
        
        output = self.Output(processed_data=processed_data, stats=stats)
        return output.model_dump()

async def main():
    # Create processors
    loader = DataLoader()
    validator = DataValidator()
    processor = DataProcessor()

    # Create flows
    loader_flow = Flow(
        callable=loader,
        config=FlowConfig(
            name="data_loader",
            flow_type=FlowType.PROCESS
        )
    )

    validator_flow = Flow(
        callable=validator,
        config=FlowConfig(
            name="validator",
            flow_type=FlowType.PROCESS
        )
    )

    processor_flow = Flow(
        callable=processor,
        config=FlowConfig(
            name="processor",
            flow_type=FlowType.PROCESS
        )
    )

    # Register dependencies
    validator_flow.register_to(
        loader_flow,
        required_deps=[loader_flow.process_id]
    )
    
    processor_flow.register_to(
        loader_flow,
        required_deps=[loader_flow.process_id],
        optional_deps=[validator_flow.process_id]
    )

    # Execute
    try:
        # Start with loader
        loader_result = await loader_flow.execute({
            "file_path": "data.csv",
            "batch_size": 10
        })
        
        # Validator and processor can run in parallel
        validator_task = asyncio.create_task(validator_flow.execute({
            "threshold": 5.0
        }))
        
        processor_task = asyncio.create_task(processor_flow.execute({
            "multiplier": 2.0
        }))
        
        # Wait for all to complete
        await asyncio.gather(validator_task, processor_task)
        
    except Exception as e:
        print(f"Flow execution failed: {e}")
    finally:
        # Cleanup
        FlowContext.get_instance().cleanup()

if __name__ == "__main__":
    asyncio.run(main())