"""
Registry for managing synthetic data producers.
"""

from typing import Dict, List, Any, Optional
import logging
from .base_producer import BaseProducer, AggregatingProducer
from .process_producer import ProcessProducer
from .network_producer import NetworkProducer
from .kernel_params_producer import KernelParamsProducer
from .modules_producer import ModulesProducer
from .world_writable_producer import WorldWritableProducer
from .suid_producer import SuidProducer
from .ioc_producer import IocProducer
from .mac_producer import MacProducer
from .dns_producer import DnsProducer
from .endpoint_behavior_producer import EndpointBehaviorProducer
from .context_aware_producer import ContextAwareProducer

logger = logging.getLogger(__name__)

# Import parallel processing utilities
try:
    from .parallel_processor import process_producers_parallel, get_parallel_processor
    PARALLEL_AVAILABLE = True
    logger.info("Parallel processing module imported successfully")
except ImportError as e:
    PARALLEL_AVAILABLE = False
    logger.warning("Parallel processing import failed: %s", e)

class ProducerRegistry:
    """Registry for all synthetic data producers."""

    def __init__(self):
        self.producers: Dict[str, BaseProducer] = {}
        self._register_default_producers()

    def _register_default_producers(self):
        """Register all default producers."""
        self.register_producer("processes", ProcessProducer())
        self.register_producer("network", NetworkProducer())
        self.register_producer("kernel_params", KernelParamsProducer())
        self.register_producer("modules", ModulesProducer())
        self.register_producer("world_writable", WorldWritableProducer())
        self.register_producer("suid", SuidProducer())
        self.register_producer("ioc", IocProducer())
        self.register_producer("mac", MacProducer())
        self.register_producer("dns", DnsProducer())
        self.register_producer("endpoint_behavior", EndpointBehaviorProducer())
        self.register_producer("context", ContextAwareProducer())

    def register_producer(self, name: str, producer: BaseProducer):
        """Register a producer."""
        self.producers[name] = producer

    def get_producer(self, name: str) -> BaseProducer:
        """Get a producer by name."""
        if name not in self.producers:
            raise ValueError(f"Producer '{name}' not found")
        return self.producers[name]

    def list_producers(self) -> List[str]:
        """List all registered producers."""
        return list(self.producers.keys())

    def generate_all_findings(self, counts: Optional[Dict[str, int]] = None, conservative_parallel: bool = True, gpu_optimized: Optional[bool] = None, max_workers: Optional[int] = None, density_mode: str = "high") -> Dict[str, List[Dict[str, Any]]]:
        """Generate findings from all producers.

        Args:
            counts: Dictionary mapping producer names to number of findings to generate.
                   If None, generates 100 findings per producer.
            conservative_parallel: Whether to use conservative parallel processing
            gpu_optimized: Whether to use GPU-optimized parallel processing
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary mapping producer names to their findings.
        """
        if counts is None:
            if density_mode == "high":
                counts = {
                    "processes": 500,
                    "network": 500,
                    "kernel_params": 300,
                    "modules": 300,
                    "world_writable": 400,
                    "suid": 300,
                    "ioc": 400,
                    "mac": 200,
                    "dns": 400,
                    "endpoint_behavior": 500,
                    "context": 400,
                }
            else:
                counts = {name: 100 for name in self.producers.keys()}

        # Use parallel processing if available and beneficial
        if PARALLEL_AVAILABLE and len(self.producers) > 2:
            processor = get_parallel_processor(conservative_parallel, gpu_optimized, max_workers)
            logger.info(
                "Using parallel processing for %d producers (%d workers)",
                len(self.producers),
                processor.max_workers,
            )
            return process_producers_parallel(self.producers, counts, "Generating findings", processor)
        else:
            # Fallback to sequential processing for small numbers or when parallel not available
            if not PARALLEL_AVAILABLE:
                logger.info("Parallel processing not available, using sequential processing")
            else:
                logger.info("Small number of producers (%d), using sequential processing", len(self.producers))

            results = {}
            for name, producer in self.producers.items():
                count = counts.get(name, 100)
                try:
                    raw_findings = producer.generate_findings(count)
                    if isinstance(producer, AggregatingProducer):
                        results[name] = producer.aggregate_findings(raw_findings)
                    else:
                        results[name] = raw_findings
                except Exception as exc:
                    logger.error("Error generating findings for producer %s: %s", name, exc)
                    results[name] = []
            return results

# Global registry instance
registry = ProducerRegistry()