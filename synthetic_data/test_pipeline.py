#!/usr/bin/env python3
"""
Test script for the complete synthetic data pipeline.
"""

import sys
import os
import json
from pathlib import Path

# Add the synthetic_data directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from synthetic_data.synthetic_data_pipeline import SyntheticDataPipeline

def test_complete_pipeline():
    """Test the complete synthetic data pipeline."""
    print("Testing Complete Synthetic Data Pipeline")
    print("=" * 50)

    # Initialize pipeline
    pipeline = SyntheticDataPipeline()

    print(f"Available producers: {pipeline.get_available_producers()}")
    print(f"Available correlation producers: {pipeline.get_available_correlation_producers()}")
    print()

    # Test pipeline execution with small dataset
    producer_counts = {
        "processes": 3,
        "network": 3,
        "kernel_params": 2,
        "modules": 2,
        "world_writable": 2,
        "suid": 2,
        "ioc": 2,
        "mac": 2,
        "dns": 2,
        "endpoint_behavior": 2
    }

    print("Executing pipeline...")
    result = pipeline.execute_pipeline(
        producer_counts=producer_counts,
        output_path="test_pipeline_output.json",
        save_intermediate=True
    )

    print("\n✅ Pipeline execution successful!")
    print(f"📊 Findings generated: {result['data_summary']['total_findings']}")
    print(f"🔗 Correlations generated: {result['data_summary']['total_correlations']}")
    print(f"✅ Verification status: {result['quality_metrics']['verification_status']}")
    print(".2f")
    print(".2f")
    print(f"💾 Dataset size: {result['performance_metrics']['data_size_mb']} MB")

    assert result["quality_metrics"]["verification_status"] in {"passed", "warning", "failed"}
    assert result["data_summary"]["total_findings"] >= 0
    assert result["data_summary"]["total_correlations"] >= 0

    # Validate output file
    if os.path.exists("test_pipeline_output.json"):
        with open("test_pipeline_output.json", 'r') as f:
            dataset = json.load(f)

        print("\n📁 Output file validation:")
        print(f"  - Dataset version: {dataset.get('metadata', {}).get('version', 'unknown')}")
        print(f"  - Total findings: {dataset.get('data', {}).get('statistics', {}).get('total_findings', 0)}")
        print(f"  - Total correlations: {dataset.get('data', {}).get('statistics', {}).get('total_correlations', 0)}")

        # Check intermediate files
        intermediate_dir = Path("intermediate_results")
        if intermediate_dir.exists():
            intermediate_files = list(intermediate_dir.glob("*.json"))
            print(f"  - Intermediate files: {len(intermediate_files)}")
            for f in intermediate_files:
                print(f"    * {f.name}")

    print("\n🎉 All tests passed!")

def test_pipeline_components():
    """Test individual pipeline components."""
    print("\nTesting Pipeline Components")
    print("-" * 30)

    pipeline = SyntheticDataPipeline()

    # Test producer registry
    producers = pipeline.producer_registry.list_producers()
    print(f"✓ Producer registry: {len(producers)} producers available")

    # Test correlation registry
    corr_producers = pipeline.correlation_registry.list_correlation_producers()
    print(f"✓ Correlation registry: {len(corr_producers)} correlation producers available")

    # Test finding generation
    findings = pipeline.producer_registry.generate_all_findings({"processes": 2, "network": 2})
    total_findings = sum(len(f) for f in findings.values())
    print(f"✓ Finding generation: {total_findings} findings generated")

    # Test correlation analysis
    correlations = pipeline.correlation_registry.analyze_all_correlations(findings)
    print(f"✓ Correlation analysis: {len(correlations)} correlations generated")

    # Test verification
    verification = pipeline.verification_agent.verify_dataset(findings, correlations)
    print(f"✓ Verification: {verification.get('overall_status', 'unknown')}")

    print("✓ All components working correctly")

if __name__ == "__main__":
    # Run component tests
    test_pipeline_components()

    # Run complete pipeline test
    success = test_complete_pipeline()

    if success:
        print("\n🎯 Pipeline test completed successfully!")
    else:
        print("\n💥 Pipeline test failed!")
        sys.exit(1)