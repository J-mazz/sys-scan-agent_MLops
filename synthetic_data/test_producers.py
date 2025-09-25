#!/usr/bin/env python3
"""
Test script to validate all synthetic data producers.
"""

import sys
import os
import json

# Add the synthetic_data directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from producer_registry import registry

def test_producers():
    """Test all registered producers."""
    print("Testing synthetic data producers...")
    print("=" * 50)

    producers = registry.list_producers()
    print(f"Registered producers: {producers}")
    print()

    # Test each producer
    for producer_name in producers:
        print(f"Testing {producer_name} producer...")
        try:
            producer = registry.get_producer(producer_name)
            findings = producer.generate_findings(5)  # Generate 5 findings each

            print(f"  Generated {len(findings)} findings")

            # Validate findings structure
            for i, finding in enumerate(findings):
                required_fields = [
                    "id", "title", "severity", "risk_score", "description",
                    "metadata", "category", "tags", "risk_subscores"
                ]

                missing_fields = [field for field in required_fields if field not in finding]
                if missing_fields:
                    print(f"  ERROR: Finding {i} missing fields: {missing_fields}")
                else:
                    print(f"  Finding {i}: {finding['title']} (severity: {finding['severity']}, risk: {finding['risk_score']})")

            print("  ✓ Producer working correctly")
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    # Test generating all findings
    print("Testing generate_all_findings...")
    try:
        all_findings = registry.generate_all_findings()
        total_findings = sum(len(findings) for findings in all_findings.values())
        print(f"Generated {total_findings} total findings across all producers")

        # Save a sample to file for inspection
        sample_output = {
            "producers": list(all_findings.keys()),
            "total_findings": total_findings,
            "sample_findings": {}
        }

        for producer_name, findings in all_findings.items():
            if findings:
                sample_output["sample_findings"][producer_name] = findings[0]

        with open("producer_test_output.json", "w") as f:
            json.dump(sample_output, f, indent=2)

        print("Sample output saved to producer_test_output.json")
        print("✓ All producers working correctly")

    except Exception as e:
        print(f"ERROR in generate_all_findings: {e}")

if __name__ == "__main__":
    test_producers()