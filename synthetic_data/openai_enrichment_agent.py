"""Legacy OpenAI enrichment agent removed in favor of batch builder."""

raise ImportError("openai_enrichment_agent has been removed; use batch_enrichment_agent instead.")
        findings_map = dataset["findings"]

        # Flatten findings
        all_findings: List[Dict[str, Any]] = []
        for severity_dict in findings_map.values():
            for finding_list in severity_dict.values():
                all_findings.extend(finding_list)

        logger.info("Phase 2: Enriching %d findings using OpenAI...", len(all_findings))

        tasks = [self.enrich_finding(f) for f in all_findings]
        try:  # pragma: no cover - optional progress
            from tqdm.asyncio import tqdm_asyncio

            enriched_findings_list = await tqdm_asyncio.gather(*tasks)
        except Exception:
            enriched_findings_list = await asyncio.gather(*tasks)

        # Correlation processing (reuse same enrichment prompt/logic)
        all_correlations: List[Dict[str, Any]] = dataset.get("correlations", []) if isinstance(dataset, dict) else []
        logger.info("Phase 2b: Enriching %d correlations using OpenAI...", len(all_correlations))
        corr_tasks = [self.enrich_finding(c) for c in all_correlations]
        if all_correlations:
            try:  # pragma: no cover
                from tqdm.asyncio import tqdm_asyncio

                await tqdm_asyncio.gather(*corr_tasks)
            except Exception:
                await asyncio.gather(*corr_tasks)

        # Update metadata
        report["transformed_dataset"].setdefault("metadata", {})
        meta = report["transformed_dataset"]["metadata"]

        processing = meta.get("processing_pipeline", [])
        if "openai_enrichment" not in processing:
            processing = list(processing) + ["openai_enrichment"]

        meta.update(
            {
                "enrichment_agent": "openai_sdk",
                "enrichment_model": self.model,
                "openai_enriched": True,
                "langchain_enriched": False,
                "enriched_findings": len(enriched_findings_list),
                "enriched_correlations": len(all_correlations),
                "processing_pipeline": processing,
            }
        )

        logger.info("Phase 3: Saving enriched dataset to %s", output_path)
        with open(output_path, "w") as f:
            json.dump(report["transformed_dataset"], f, indent=2)

        logger.info("✅ Agent Run Complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OpenAI SDK enrichment pipeline")
    parser.add_argument("--model", help="OpenAI model for structured outputs", default=None)
    parser.add_argument("--output", help="Path for enriched dataset", default="final_enriched_security_data.json")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent OpenAI calls")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per finding")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Cap on completion tokens")
    parser.add_argument(
        "--producer-counts",
        type=str,
        default=None,
        help="JSON mapping of producer -> count (e.g., '{\\"processes\\": 10, \\"network\\": 5}')",
    )

    args = parser.parse_args()

    counts = None
    if args.producer_counts:
        try:
            counts = json.loads(args.producer_counts)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON for --producer-counts; ignoring")

    agent = EnrichmentAgent(
        model=args.model,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
    )

    asyncio.run(
        agent.run(
            output_path=args.output,
            producer_counts=counts,
        )
    )