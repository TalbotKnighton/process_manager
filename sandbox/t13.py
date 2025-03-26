"""Example usage of the flow package with random number generation."""
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import numpy as np
import logging
import json
from datetime import datetime

from flow.core.flow import Flow, FlowConfig
from flow.core.types import FlowType, FlowStatus, VisFormat
from flow.visualization.graph import FlowVisualizer
from process_manager.data_handlers.random_variables import UniformDistribution
from process_manager.data_handlers.random_variables import RandomVariableHash

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RandomGeneratorConfig(BaseModel):
    """Configuration for random number generation."""
    size: int = Field(default=1000, gt=0)
    seed: Optional[int] = None
    min_value: float = Field(default=0.0)
    max_value: float = Field(default=1.0)

class RandomGenerator(BaseModel):
    """Generates random numbers using UniformDistribution."""
    class Input(BaseModel):
        config: RandomGeneratorConfig

    class Output(BaseModel):
        data: List[float]
        stats: Dict[str, float]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Processing RandomGenerator")
        config = RandomGeneratorConfig(**input_data['config'])
        
        # Create random variable
        rv = UniformDistribution(
            min_value=config.min_value,
            max_value=config.max_value
        )
        
        if config.seed is not None:
            np.random.seed(config.seed)
            
        # Generate random numbers
        data = [rv.get_random() for _ in range(config.size)]
        
        # Calculate statistics
        stats = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data))
        }
        logger.info("Completed RandomGenerator processing")
        return self.Output(data=data, stats=stats).model_dump()

class DataHasher(BaseModel):
    """Hashes random numbers using RandomVariableHash."""
    class Input(BaseModel):
        data: List[float]
        num_bins: int = Field(default=10, gt=0)

    class Output(BaseModel):
        hash_values: List[int]
        bin_counts: List[int]
        stats: Dict[str, float]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Processing DataHasher")
        data = input_data['data']
        num_bins = input_data.get('num_bins', 10)
        
        # Create hasher
        hasher = RandomVariableHash(num_bins=num_bins)
        
        # Hash values
        hash_values = [hasher.hash(x) for x in data]
        
        # Count bins
        bin_counts = [0] * num_bins
        for h in hash_values:
            bin_counts[h] += 1
            
        # Calculate statistics
        total = len(hash_values)
        expected_per_bin = total / num_bins
        chi_square = sum(
            ((count - expected_per_bin) ** 2) / expected_per_bin 
            for count in bin_counts
        )
        
        stats = {
            "chi_square": float(chi_square),
            "mean_bin_count": float(np.mean(bin_counts)),
            "std_bin_count": float(np.std(bin_counts))
        }
        logger.info("Completed DataHasher processing")
        return self.Output(
            hash_values=hash_values,
            bin_counts=bin_counts,
            stats=stats
        ).model_dump()

class DataAnalyzer(BaseModel):
    """Analyzes both raw and hashed data."""
    class Input(BaseModel):
        raw_data: List[float]
        hash_values: List[int]
        bin_counts: List[int]
        raw_stats: Dict[str, float]
        hash_stats: Dict[str, float]

    class Output(BaseModel):
        analysis: Dict[str, Any]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Processing DataAnalyzer")
        # Combine statistics and add additional analysis
        analysis = {
            "raw_data_stats": input_data['raw_stats'],
            "hash_stats": input_data['hash_stats'],
            "distribution_quality": {
                "uniformity_score": input_data['hash_stats']['chi_square'],
                "bin_variation": input_data['hash_stats']['std_bin_count'],
            }
        }
        logger.info("Completed DataAnalyzer processing")
        return self.Output(analysis=analysis).model_dump()

async def run_flow_example():
    """Run the flow example with proper cleanup."""
    flows = []  # Keep track of all flows
    try:
        logger.info("Creating flows...")
        # Create processors
        generator = RandomGenerator()
        hasher = DataHasher()
        analyzer = DataAnalyzer()

        # Create flows
        generator_flow = Flow(
            callable=generator,
            config=FlowConfig(
                name="Random Generator",
                flow_type=FlowType.PROCESS
            )
        )
        flows.append(generator_flow)

        hasher_flow = Flow(
            callable=hasher,
            config=FlowConfig(
                name="Data Hasher",
                flow_type=FlowType.PROCESS
            )
        )
        flows.append(hasher_flow)

        analyzer_flow = Flow(
            callable=analyzer,
            config=FlowConfig(
                name="Data Analyzer",
                flow_type=FlowType.PROCESS
            )
        )
        flows.append(analyzer_flow)

        logger.info("Registering dependencies...")
        # Register dependencies
        hasher_flow.register_to(
            generator_flow,
            required_deps=[generator_flow.process_id]
        )
        
        analyzer_flow.register_to(
            generator_flow,
            required_deps=[generator_flow.process_id, hasher_flow.process_id]
        )

        logger.info("Creating flow visualization...")
        visualizer = FlowVisualizer(generator_flow)
        flow_chart = visualizer.to_plotly()
        flow_chart.show()
        # print("\nInitial Flow Structure (Mermaid):")
        # print(flow_chart)
        
        # with open("flow.mmd", "w") as f:
        #     f.write(flow_chart)
        # logger.info("Flow diagram saved to flow.mmd")

        # Execute just the root flow - it should handle dependencies
        logger.info("Executing flow graph...")
        result = await generator_flow.execute({
            "config": {
                "size": 10,
                "seed": 42,
                "min_value": 0.0,
                "max_value": 1.0
            }
        })
        logger.info("Flow execution completed")

        # Print results
        logger.info("\nResults:")
        print(json.dumps(result.output, indent=2))

        # Show final flow structure
        logger.info("\nFinal Flow Structure (Mermaid):")
        mermaid_final = visualizer.to_mermaid()
        print(mermaid_final)
        
        with open("flow_final.mmd", "w") as f:
            f.write(mermaid_final)
        logger.info("Final flow diagram saved to flow_final.mmd")

    except Exception as e:
        logger.error(f"Error executing flows: {e}", exc_info=True)
        raise
    finally:
        logger.info("Starting cleanup...")
        # Cleanup all flows
        for flow in flows:
            if flow and flow.context:
                logger.info(f"Cleaning up flow {flow.config.name}")
                try:
                    flow.context.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up flow {flow.config.name}: {e}")
        logger.info("All flows cleaned up")

async def main():
    """Main entry point with proper error handling."""
    try:
        logger.info("Starting flow example...")
        await run_flow_example()
        logger.info("Flow example completed successfully")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Example completed, exiting...")
        import sys
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())