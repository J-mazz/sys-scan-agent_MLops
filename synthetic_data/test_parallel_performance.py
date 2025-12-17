"""
Performance test for parallel vs sequential processing.
"""

import time
from synthetic_data.producer_registry import registry
from synthetic_data.correlation_registry import correlation_registry
from synthetic_data.parallel_processor import get_parallel_processor

def test_parallel_performance():
    """Test performance difference between parallel and sequential processing."""

    print("🚀 Performance Test: Parallel vs Sequential Processing")
    print("=" * 60)

    # Test data
    producer_counts = {
        "processes": 50,
        "network": 30,
        "kernel_params": 20,
        "modules": 15,
        "world_writable": 10,
        "suid": 10,
        "ioc": 10,
        "mac": 5
    }

    # Test parallel processing
    print("\n🔄 Testing PARALLEL processing...")
    start_time = time.time()

    parallel_processor = get_parallel_processor(conservative=True)
    parallel_results = registry.generate_all_findings(producer_counts, conservative_parallel=True)

    parallel_time = time.time() - start_time
    parallel_findings = sum(len(f) for f in parallel_results.values())

    print(".2f")
    print(f"📊 Generated {parallel_findings} findings in parallel")

    # Test sequential processing (simulate by setting small threshold)
    print("\n📝 Testing SEQUENTIAL processing...")
    start_time = time.time()

    # Temporarily modify registry to force sequential
    original_producers = registry.producers.copy()
    # Create a smaller subset to simulate sequential behavior
    small_producers = dict(list(original_producers.items())[:2])  # Only 2 producers
    registry.producers = small_producers

    sequential_results = registry.generate_all_findings(
        {k: v for k, v in producer_counts.items() if k in small_producers},
        conservative_parallel=True
    )

    # Restore original producers
    registry.producers = original_producers

    # Generate remaining findings sequentially
    remaining_counts = {k: v for k, v in producer_counts.items() if k not in small_producers}
    if remaining_counts:
        for name, count in remaining_counts.items():
            producer = registry.get_producer(name)
            sequential_results[name] = producer.generate_findings(count)

    sequential_time = time.time() - start_time
    sequential_findings = sum(len(f) for f in sequential_results.values())

    print(".2f")
    print(f"📊 Generated {sequential_findings} findings sequentially")

    # Calculate performance improvement
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        improvement = (speedup - 1) * 100

        print("\n🎯 PERFORMANCE RESULTS")
        print("=" * 40)
        print(".2f")
        print(".2f")
        print(".1f")
        print(".2f")

        if improvement > 0:
            print(".1f")
        else:
            print(".1f")

    print("\n✅ Performance test completed!")

if __name__ == "__main__":
    test_parallel_performance()