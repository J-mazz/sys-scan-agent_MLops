"""Parallel processing utilities optimized for diverse execution environments."""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Setup logging
logger = logging.getLogger(__name__)

# Global worker functions for multiprocessing (must be at module level)
def _process_single_producer(producer_name: str, producers: Dict[str, Any], counts: Dict[str, int]) -> tuple[str, List[Dict[str, Any]]]:
    """Process a single producer (module-level function for multiprocessing)."""
    producer = producers[producer_name]
    count = counts.get(producer_name, 10)
    try:
        results = producer.generate_findings(count)
        return producer_name, results
    except Exception as e:
        logger.error(f"Error processing producer {producer_name}: {e}")
        return producer_name, []

def _process_single_correlation_producer(producer_name: str, correlation_producers: Dict[str, Any], findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Process a single correlation producer (module-level function for multiprocessing)."""
    producer = correlation_producers[producer_name]
    try:
        correlations = producer.analyze_correlations(findings)
        return correlations
    except Exception as e:
        logger.error(f"Error processing correlation producer {producer_name}: {e}")
        return []


def _producer_task(payload: Tuple[str, Dict[str, Any], Dict[str, int]]) -> Tuple[str, List[Dict[str, Any]]]:
    name, producers, counts = payload
    return _process_single_producer(name, producers, counts)


def _correlation_task(payload: Tuple[str, Dict[str, Any], Dict[str, List[Dict[str, Any]]]]) -> Tuple[str, List[Dict[str, Any]]]:
    name, correlation_producers, findings = payload
    return name, _process_single_correlation_producer(name, correlation_producers, findings)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
    logger.info("CuPy available for GPU acceleration")
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    logger.info("CuPy not available - GPU acceleration limited")


@dataclass(frozen=True)
class SystemProfile:
    """Snapshot of the current host capabilities."""

    cpu_count: int
    available_memory_gb: float
    is_gpu: bool

class ParallelProcessor:
    """System-aware parallel executor with sensible defaults for Azure CPU hosts."""

    def __init__(self, conservative_mode: bool = True, gpu_optimized: bool = False, max_workers: Optional[int] = None):
        self.conservative_mode = conservative_mode
        self.gpu_optimized = gpu_optimized
        self.profile = self._detect_system_profile()

        self.max_workers = self._determine_worker_count(max_workers)
        self.use_processes = self._should_use_processes()
        self.chunk_size = 64 if self.use_processes else 16

        logger.info(
            "Parallel processor initialized: %d workers (conservative: %s, GPU: %s)",
            self.max_workers,
            conservative_mode,
            gpu_optimized,
        )
        logger.info(
            "System profile: %d CPU cores, %.1fGB RAM available",
            self.profile.cpu_count,
            self.profile.available_memory_gb,
        )

    def _detect_system_profile(self) -> SystemProfile:
        cpu_count = os.cpu_count() or multiprocessing.cpu_count() or 8
        available_memory_gb = self._get_available_memory_gb()
        return SystemProfile(cpu_count=cpu_count, available_memory_gb=available_memory_gb, is_gpu=_is_gpu_env)

    def _determine_worker_count(self, override: Optional[int]) -> int:
        if override is not None:
            return max(1, override)

        cpu_count = self.profile.cpu_count
        mem = self.profile.available_memory_gb

        # Tailor defaults for 8-core / 16GB Azure CPU hosts
        if cpu_count >= 8 and 12 <= mem <= 24:
            target = 8 if not self.conservative_mode else 6
        elif cpu_count >= 16 and mem >= 32:
            target = min(cpu_count, int(mem // 2))
        else:
            target = min(cpu_count, max(4, int(mem // 2)))

        if self.gpu_optimized and self.profile.is_gpu:
            target = min(max(4, target), cpu_count * 2)

        return max(1, target)

    def _should_use_processes(self) -> bool:
        if self.profile.available_memory_gb < 8:
            return False
        if self.gpu_optimized and self.profile.is_gpu:
            return True
        return self.max_workers >= 4

    def _get_available_memory_gb(self) -> float:
        """Get available memory in GB."""
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                mem = psutil.virtual_memory()
                return mem.available / (1024 ** 3)
            except Exception:
                logger.debug("psutil memory probe failed", exc_info=True)
        return 8.0

    def _check_system_resources(self) -> bool:
        """Check if system has sufficient resources for parallel processing."""
        if not PSUTIL_AVAILABLE or psutil is None:
            return True

        try:
            available_memory_gb = self._get_available_memory_gb()
            cpu_percent = psutil.cpu_percent(interval=0.2)

            cpu_threshold = 95 if available_memory_gb >= 32 else 90 if available_memory_gb >= 16 else 85
            if cpu_percent > cpu_threshold:
                logger.warning("High CPU usage detected (%.1f%%), throttling workers", cpu_percent)
                self.max_workers = max(1, self.max_workers // 2)
                return False

            memory = psutil.virtual_memory()
            memory_threshold = 80 if available_memory_gb < 12 else 90
            if memory.percent > memory_threshold:
                logger.warning("High memory usage detected (%.1f%%), throttling workers", memory.percent)
                self.max_workers = max(1, self.max_workers // 2)
                return False

            return True
        except Exception as exc:
            logger.debug("Resource check failed", exc_info=exc)
            return True

    def _get_executor(self, max_workers: int):
        """Get the appropriate executor based on configuration."""
        if self.use_processes:
            return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def process_items_parallel(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        description: str = "Processing items"
    ) -> List[R]:
        """
        Process a list of items in parallel with GPU optimizations.

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            description: Description for logging

        Returns:
            List of results in the same order as input items
        """
        if not items:
            return []

        # Check system resources before starting
        self._check_system_resources()

        logger.info("%s (%d items) using %d workers", description, len(items), self.max_workers)

        # For small datasets, don't bother with parallel processing
        if len(items) <= 2:
            logger.info("Small dataset detected, processing sequentially")
            results: List[R] = []
            for item in items:
                try:
                    result = process_func(item)
                    results.append(result)
                except Exception as exc:
                    logger.error("Error processing item: %s", exc)
            return results

        # GPU optimization: Use ProcessPoolExecutor for CPU-bound tasks
        with self._get_executor(self.max_workers) as executor:
            futures = [executor.submit(process_func, item) for item in items]
            results: List[R] = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("Error processing item: %s", exc)

        logger.info("%s completed: %d/%d items processed", description, len(results), len(items))
        return results

    def process_dict_parallel(
        self,
        items_dict: Dict[str, T],
        process_func: Callable[[str, T], tuple[str, R]],
        description: str = "Processing dictionary items"
    ) -> Dict[str, R]:
        """
        Process a dictionary of items in parallel with GPU optimizations.

        Args:
            items_dict: Dictionary of items to process
            process_func: Function that takes (key, value) and returns (key, result)
            description: Description for logging

        Returns:
            Dictionary mapping keys to results
        """
        if not items_dict:
            return {}

        # Check system resources before starting
        self._check_system_resources()

        logger.info("%s (%d items) using %d workers", description, len(items_dict), self.max_workers)

        # For small datasets, don't bother with parallel processing
        if len(items_dict) <= 2:
            logger.info("Small dataset detected, processing sequentially")
            results: Dict[str, R] = {}
            for key, value in items_dict.items():
                try:
                    result_key, result_value = process_func(key, value)
                    results[result_key] = result_value
                except Exception as exc:
                    logger.error("Error processing %s: %s", key, exc)
            return results

        # GPU optimization: Process in larger chunks for better throughput
        with self._get_executor(self.max_workers) as executor:
            futures = [executor.submit(process_func, key, value) for key, value in items_dict.items()]
            results: Dict[str, R] = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result_key, result_value = future.result()
                    results[result_key] = result_value
                except Exception as exc:
                    logger.error("Error processing item: %s", exc)

        logger.info("%s completed: %d/%d items processed", description, len(results), len(items_dict))
        return results

# Environment-specific processor instances
def detect_gpu_environment() -> bool:
    """Detect if running in a GPU environment (L4, T4, A100, or similar)."""
    try:
        # Check for NVIDIA GPU
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Check for NVIDIA GPUs including L4 and T4
            gpu_indicators = ['L4', 'T4', 'A100', 'V100', 'P100', 'K80', 'Tesla', 'GeForce', 'Quadro']
            if any(gpu in result.stdout for gpu in gpu_indicators):
                return True
            # Also check for CUDA capability
            if 'CUDA' in result.stdout or 'NVIDIA' in result.stdout:
                return True
    except (ImportError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: Check environment variables
    gpu_env_vars = ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES']
    for var in gpu_env_vars:
        if os.getenv(var) is not None:
            return True

    # Colab-specific detection
    try:
        # Check if we're in Google Colab
        import sys
        if 'google.colab' in sys.modules:
            return True
    except ImportError:
        pass

    return False

# Auto-detect environment and create optimized processors
_is_gpu_env = detect_gpu_environment()

parallel_processor_local = ParallelProcessor(conservative_mode=True, gpu_optimized=False)  # For local CPU
parallel_processor_cloud = ParallelProcessor(conservative_mode=False, gpu_optimized=_is_gpu_env)  # Auto-detect GPU
parallel_processor_gpu = ParallelProcessor(conservative_mode=False, gpu_optimized=True)  # Explicit GPU optimization

# Default processor: prioritize high-memory systems, then GPU, then conservative CPU
def _get_default_processor():
    """Get the best default processor based on system capabilities."""
    available_memory_gb = 8.0
    if PSUTIL_AVAILABLE and psutil is not None:
        try:
            mem = psutil.virtual_memory()
            available_memory_gb = mem.available / (1024 ** 3)
        except Exception:
            logger.debug("psutil memory probe failed", exc_info=True)

    if _is_gpu_env:
        return parallel_processor_gpu
    if available_memory_gb >= 32:
        return ParallelProcessor(conservative_mode=False, gpu_optimized=False)
    if available_memory_gb >= 16:
        return parallel_processor_cloud
    return parallel_processor_local

parallel_processor = _get_default_processor()

def get_parallel_processor(conservative: bool = True, gpu_optimized: Optional[bool] = None, max_workers: Optional[int] = None):
    """Get the appropriate parallel processor based on execution environment."""
    if gpu_optimized is None:
        gpu_optimized = _is_gpu_env

    # Auto-determine conservative mode based on available memory if not specified
    if conservative:
        available_memory_gb = 8.0
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                mem = psutil.virtual_memory()
                available_memory_gb = mem.available / (1024 ** 3)
            except Exception:
                logger.debug("psutil memory probe failed", exc_info=True)

        if available_memory_gb >= 32:
            conservative = False

    if gpu_optimized:
        return ParallelProcessor(conservative_mode=conservative, gpu_optimized=True, max_workers=max_workers)
    elif conservative:
        return parallel_processor_local
    else:
        return parallel_processor_cloud


def process_producers_parallel(producers: Dict[str, Any], counts: Dict[str, int], description: str, processor: ParallelProcessor) -> Dict[str, List[Dict[str, Any]]]:
    """Process multiple producers in parallel using the given processor.

    Args:
        producers: Dictionary of producer name -> producer instance
        counts: Dictionary of producer name -> number of items to generate
        description: Description for progress reporting
        processor: The parallel processor to use

    Returns:
        Dictionary of producer name -> list of generated items
    """
    producer_names = list(producers.keys())

    tasks = [(name, producers, counts) for name in producer_names]

    results = processor.process_items_parallel(tasks, _producer_task, description=description)

    mapped: Dict[str, List[Dict[str, Any]]] = {name: findings for name, findings in results}
    logger.info("Parallel processing completed: %s producers processed", len(mapped))
    return mapped


def process_correlations_parallel(findings: Dict[str, List[Dict[str, Any]]], correlation_producers: Dict[str, Any], description: str, processor: ParallelProcessor) -> List[Dict[str, Any]]:
    """Process correlation analysis in parallel using the given processor.

    Args:
        findings: Dictionary of scanner type -> list of findings
        correlation_producers: Dictionary of correlation producer name -> producer instance
        description: Description for progress reporting
        processor: The parallel processor to use

    Returns:
        List of generated correlations
    """
    producer_names = list(correlation_producers.keys())

    tasks = [(name, correlation_producers, findings) for name in producer_names]

    results = processor.process_items_parallel(tasks, _correlation_task, description=description)

    merged: List[Dict[str, Any]] = []
    for name, correlations in results:
        logger.debug("Completed correlation producer %s: %d correlations", name, len(correlations))
        merged.extend(correlations)

    logger.info(
        "Parallel correlation processing completed: %d total correlations from %d producers",
        len(merged),
        len(producer_names)
    )
    return merged