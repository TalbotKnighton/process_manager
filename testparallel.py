"""Flow example with parallel execution support."""
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel
import logging
from flow.core.flow import Flow, FlowConfig
from flow.core.types import FlowType, FlowStatus
from flow.core.context import FlowContext
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NumberGenerator(BaseModel):
    """Generates a sequence of numbers with simulated processing time."""
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("NumberGenerator.process called")
        # Simulate some processing time
        time.sleep(1)
        numbers = list(range(input_data.get('start', 1), input_data.get('end', 5)))
        logger.debug(f"Generated numbers: {numbers}")
        return {"numbers": numbers}

class NumberMultiplier(BaseModel):
    """Multiplies numbers with simulated processing time."""
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("NumberMultiplier.process called")
        numbers = input_data.get('numbers', [])
        factor = input_data.get('factor', 2)
        # Simulate some processing time
        time.sleep(1)
        result = [n * factor for n in numbers]
        return {"result": result}

class ParallelFlow(Flow):
    """Flow subclass with parallel execution support."""
    
    async def _execute_flow(
        self,
        input_data: Dict[str, Any],
        fail_quickly: bool = False
    ) -> Any:
        """Internal flow execution method with parallel support."""
        from flow.core.flow import FlowResult
        
        logger.debug(f"ParallelFlow._execute_flow started for {self.config.name}")
        result = FlowResult(
            process_id=self.process_id,
            status=FlowStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Execute processor using process pool
            logger.debug("Submitting to process pool")
            if hasattr(self.function, 'process'):
                output = await self.context.pool_manager.submit_task(
                    self.process_id,
                    self.config.flow_type,
                    self.function.process,
                    input_data,
                    timeout=self.config.timeout
                )
            else:
                output = await self.context.pool_manager.submit_task(
                    self.process_id,
                    self.config.flow_type,
                    self.function,
                    input_data,
                    timeout=self.config.timeout
                )
            
            logger.debug(f"Processor output: {output}")
            
            # Update result
            result.output = output
            result.status = FlowStatus.COMPLETED
            result.end_time = datetime.now()
            
            # Store result
            await self.context.results_manager.save_result(self.process_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in _execute_flow: {e}", exc_info=True)
            result.status = FlowStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            await self.context.results_manager.save_result(self.process_id, result)
            raise

async def run_parallel_example():
    """Run example with parallel flows."""
    logger.debug("Starting parallel example")
    context = None
    
    try:
        # Initialize context
        context = FlowContext.get_instance()

        # Create multiple processors and flows
        flows = []
        for i in range(3):  # Create 3 parallel number generators
            processor = NumberGenerator()
            flow = ParallelFlow(
                callable=processor,
                config=FlowConfig(
                    name=f"Number Generator {i}",
                    flow_type=FlowType.PROCESS  # Use process pool
                )
            )
            flows.append(flow)

        # Execute flows in parallel
        logger.debug("Starting parallel execution")
        start_time = time.time()
        
        tasks = [
            flow._execute_flow({
                "start": i * 5,
                "end": (i + 1) * 5
            }) for i, flow in enumerate(flows)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        logger.debug(f"All flows completed in {end_time - start_time:.2f} seconds")

        # Print results
        for i, result in enumerate(results):
            print(f"Flow {i} result: {result.output}")

    except Exception as e:
        logger.error(f"Error in parallel example: {e}", exc_info=True)
    finally:
        logger.debug("Cleaning up")
        if context:
            context.cleanup()
        logger.debug("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(run_parallel_example())