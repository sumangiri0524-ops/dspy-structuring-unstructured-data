"""Main CLI entrypoint for the DSPy LLM Pipeline."""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

from src.pipeline import DSPyPipeline
from src.validation import PipelineValidator, print_validation_report

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DSPy Knowledge Pipeline: Entity Extraction, Deduplication, and Graph Generation"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in deterministic mock mode for structural testing without calling live LLMs"
    )
    parser.add_argument(
        "--urls",
        type=str,
        default="data/urls.txt",
        help="Path to URLs list (default: data/urls.txt)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
        help="Target confidence for deduplication loop (default: 0.90)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for deduplication (default: 10)"
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    print("\n" + "=" * 80)
    print("      DSPY PRACTICAL ASSIGNMENT: STRUCTURED DATA COMPILER PIPELINE")
    print("=" * 80)

    has_longcat = bool(os.getenv("LONGCAT_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    use_mock = args.mock
    if not use_mock and not (has_longcat or has_openai):
        print("\n[!] NOTICE: No LONGCAT_API_KEY or OPENAI_API_KEY detected in .env file.")
        print("    Running in offline deterministic MOCK mode for structural validation.")
        print("    To run with real LLM: add LONGCAT_API_KEY=... to your .env file.\n")
        use_mock = True
    elif not use_mock:
        mode_name = "LongCat API (" + os.getenv("LONGCAT_MODEL", "LongCat-2.0") + ")" if has_longcat else "OpenAI API"
        print(f"\n[*] Live LLM Mode Active: Using {mode_name}\n")

    # Initialize and execute pipeline
    pipeline = DSPyPipeline(
        urls_file=args.urls,
        target_confidence=args.confidence,
        batch_size=args.batch_size,
        mock=use_mock
    )

    results, summary = pipeline.run()

    # Run automated validation suite
    print("\nRunning automated validation suite...")
    validator = PipelineValidator()
    all_passed, val_results = validator.validate_all()
    print_validation_report(val_results)

    # Print summary statistics
    print("=" * 80)
    print("                         PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Execution Mode:             {summary['execution_mode'].upper()}")
    print(f"Target Confidence:          {summary['target_confidence']:.2f}")
    print(f"URLs Processed:             {summary['total_urls_processed']}")
    print(f"Scrape Success / Fail:      {summary['successful_scrapes']} / {summary['failed_scrapes']}")
    print(f"Raw Entities Extracted:     {summary['total_raw_entities']}")
    print(f"Deduplicated Entities:      {summary['total_deduplicated_entities']}")
    print(f"Avg Confidence Achieved:    {summary['average_confidence_achieved']:.4f}")
    print(f"Triples Extracted:          {summary['total_triples_extracted']}")
    print(f"Mermaid Files Created:      {summary['mermaid_files_generated']}")
    print(f"Tags CSV Records:           {summary['tags_csv_rows']}")
    print("=" * 80)
    print(f"Deliverables generated at:")
    print(f"  - Mermaid Diagrams:       outputs/mermaid/ (10 files)")
    print(f"  - Structured Tags:        outputs/tags.csv")
    print(f"  - Execution Summary:      outputs/run_summary.json")
    print("=" * 80 + "\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
