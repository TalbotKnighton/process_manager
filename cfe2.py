"""Simplified flow example."""
import asyncio
import logging
from typing import Dict, Any
from pydantic import BaseModel
from flow.core.flow import Flow, FlowConfig
from flow.core.types import FlowType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleProcessor(BaseModel):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Processing with {type(self).__name__}")
        return {"result": input_data["value"] * 2}

async def run_example():
    # Create a simple flow
    processor = SimpleProcessor()
    flow = Flow(
        callable=processor,
        config=FlowConfig(
            name="Simple Flow",
            flow_type=FlowType.PROCESS
        )
    )
    
    try:
        # Execute flow
        result = await flow.execute({"value": 21})
        print(f"Result: {result.output}")
    finally:
        if flow.context:
            flow.context.cleanup()

if __name__ == "__main__":
    asyncio.run(run_example())