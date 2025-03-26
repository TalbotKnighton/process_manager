"""Minimal test case for flow execution."""
import asyncio
from typing import Dict, Any
from pydantic import BaseModel
from flow.core.flow import Flow, FlowConfig
from flow.core.types import FlowType
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleProcessor(BaseModel):
    """Simple processor that just returns its input."""
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Processing input: {input_data}")
        return {"output": input_data["value"] * 2}

async def test_simple_flow():
    """Test a single flow execution."""
    try:
        logger.debug("Creating flow")
        flow = Flow(
            callable=SimpleProcessor(),
            config=FlowConfig(
                name="Simple Flow",
                flow_type=FlowType.INLINE  # Use INLINE to avoid process pool
            )
        )
        
        logger.debug("Executing flow")
        result = await flow.execute({"value": 21})
        
        logger.debug(f"Execution completed with result: {result}")
        print(f"Result: {result.output}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        logger.debug("Test completed")

if __name__ == "__main__":
    asyncio.run(test_simple_flow())