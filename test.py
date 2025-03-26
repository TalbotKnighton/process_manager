"""Diagnostic flow example with fixed status handling."""
import asyncio
from typing import Dict, Any
from pydantic import BaseModel
import logging
from flow.core.flow import Flow, FlowConfig
from flow.core.types import FlowType, LoggingLevel, FlowStatus
from flow.core.context import FlowContext
from datetime import datetime

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NumberGenerator(BaseModel):
    """Generates a sequence of numbers."""
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("NumberGenerator.process called")
        numbers = list(range(input_data.get('start', 1), input_data.get('end', 5)))
        logger.debug(f"Generated numbers: {numbers}")
        return {"numbers": numbers}

class DiagnosticFlow(Flow):
    """Flow subclass with detailed logging."""
    
    async def _execute_flow(
        self,
        input_data: Dict[str, Any],
        fail_quickly: bool = False
    ) -> Any:
        """Internal flow execution method."""
        from flow.core.flow import FlowResult
        
        logger.debug(f"DiagnosticFlow._execute_flow started for {self.config.name}")
        result = FlowResult(
            process_id=self.process_id,
            status=FlowStatus.RUNNING,  # Using enum value
            start_time=datetime.now()
        )
        
        try:
            # Execute processor
            logger.debug("Executing processor")
            if hasattr(self.function, 'process'):
                output = self.function.process(input_data)
            else:
                output = self.function(input_data)
            
            logger.debug(f"Processor output: {output}")
            
            # Update result
            result.output = output
            result.status = FlowStatus.COMPLETED  # Using enum value
            result.end_time = datetime.now()
            
            # Store result
            await self.context.results_manager.save_result(self.process_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in _execute_flow: {e}", exc_info=True)
            result.status = FlowStatus.FAILED  # Using enum value
            result.error = str(e)
            result.end_time = datetime.now()
            await self.context.results_manager.save_result(self.process_id, result)
            raise

async def run_diagnostic():
    """Run with detailed logging of initialization and execution."""
    logger.debug("Starting diagnostic run")
    context = None
    
    try:
        # Initialize context
        logger.debug("Initializing FlowContext")
        context = FlowContext.get_instance()
        logger.debug("FlowContext initialized")

        # Create processor and flow
        logger.debug("Creating processor and flow")
        processor = NumberGenerator()
        flow = DiagnosticFlow(
            callable=processor,
            config=FlowConfig(
                name="Number Generator",
                flow_type=FlowType.PROCESS
            )
        )
        logger.debug(f"Flow created with ID: {flow.process_id}")

        # Execute with timeout and explicit error handling
        logger.debug("Starting flow execution")
        try:
            result = await asyncio.wait_for(
                flow._execute_flow({"start": 1, "end": 5}),
                timeout=5.0
            )
            logger.debug(f"Flow execution completed with result: {result}")
            print(f"Execution result: {result.output if result else 'No result'}")
        except asyncio.TimeoutError:
            logger.error("Flow execution timed out!")
        except Exception as e:
            logger.error(f"Flow execution failed: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error in diagnostic run: {e}", exc_info=True)
    finally:
        logger.debug("Cleaning up")
        if context:
            context.cleanup()
        logger.debug("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(run_diagnostic())