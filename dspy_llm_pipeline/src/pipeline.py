"""End-to-end DSPy pipeline for scraping, entity extraction, deduplication, and graph generation."""

import os
import re
import csv
import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from src.schemas import EntityWithAttr, Triple, TagRow, URLPipelineResult, ScrapedURLData
from src.scraper import WebScraper
from src.dspy_modules import (
    configure_lm,
    extract_entities_from_text,
    deduplicate_with_lm,
    extract_triples_from_text
)
from src.mermaid import triples_to_mermaid

logger = logging.getLogger(__name__)


class DSPyPipeline:
    """Production DSPy pipeline implementing the practical assignment workflow."""

    def __init__(
        self,
        urls_file: str = "data/urls.txt",
        outputs_dir: str = "outputs",
        target_confidence: float = 0.90,
        batch_size: int = 10,
        mock: bool = False
    ):
        self.urls_file = Path(urls_file)
        self.outputs_dir = Path(outputs_dir)
        self.mermaid_dir = self.outputs_dir / "mermaid"
        self.tags_csv_path = self.outputs_dir / "tags.csv"
        self.summary_json_path = self.outputs_dir / "run_summary.json"

        self.target_confidence = target_confidence
        self.batch_size = batch_size
        self.mock = mock

        # Ensure output directories exist
        self.mermaid_dir.mkdir(parents=True, exist_ok=True)

        # Configure DSPy LM
        self.lm = configure_lm(mock=self.mock)
        self.scraper = WebScraper(timeout=15, max_retries=3)

    def load_urls(self) -> List[str]:
        """Read target URLs from file, preserving order and removing empty lines."""
        if not self.urls_file.exists():
            raise FileNotFoundError(f"URLs file not found at: {self.urls_file}")

        with open(self.urls_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        logger.info(f"Loaded {len(urls)} URLs from {self.urls_file}")
        return urls

    def process_url(self, url: str, index: int) -> URLPipelineResult:
        """Process a single URL through the complete extraction pipeline."""
        logger.info(f"\n==========================================")
        logger.info(f"Processing URL [{index}/10]: {url}")
        logger.info(f"==========================================")

        # Step 1: Scrape
        scrape_data: ScrapedURLData = self.scraper.scrape_url(url, index=index)

        # Step 2: Handle scraping failures (never fabricate information)
        if not scrape_data.success or not scrape_data.cleaned_text:
            logger.warning(f"URL [{index}] scraping was unsuccessful: {scrape_data.error_message}")
            mermaid_content = (
                f"```mermaid\nflowchart TD\n"
                f"    %% Knowledge Graph for URL {index}\n"
                f"    %% Source: {url}\n"
                f'    error_node["Scraping Failed or Blocked: {scrape_data.error_message or "No Content"}"]\n'
                f"```"
            )
            mermaid_file = self.mermaid_dir / f"mermaid_{index}.md"
            with open(mermaid_file, "w", encoding="utf-8") as mf:
                mf.write(mermaid_content)

            return URLPipelineResult(
                url=url,
                url_index=index,
                raw_entities=[],
                deduplicated_entities=[],
                entity_to_type={},
                triples=[],
                mermaid_path=str(mermaid_file),
                mermaid_syntax=mermaid_content,
                tags_count=0,
                confidence_achieved=1.0,
                dedup_iterations=0,
                scrape_success=False,
                error_message=scrape_data.error_message or "No content available"
            )

        # Step 3: Entity Extraction
        logger.info(f"[{index}] Extracting entities from {len(scrape_data.chunks)} text chunks...")
        raw_entities = extract_entities_from_text(scrape_data.chunks, mock=self.mock)
        logger.info(f"[{index}] Extracted {len(raw_entities)} raw entities.")

        # Build map of raw entity -> attr_type
        type_mapping: Dict[str, str] = {}
        for ent in raw_entities:
            key = ent.entity.strip().lower()
            if key not in type_mapping:
                type_mapping[key] = ent.attr_type

        # Step 4: Intelligent Deduplication with Confidence Loop (Target >= 0.90)
        raw_entity_strings = [ent.entity for ent in raw_entities]
        logger.info(f"[{index}] Running confidence-loop deduplication (target >= {self.target_confidence})...")
        dedup_entities, confidence, iterations = deduplicate_with_lm(
            items=raw_entity_strings,
            batch_size=self.batch_size,
            target_confidence=self.target_confidence,
            mock=self.mock
        )

        # Normalize and sanitize deduplicated entities
        clean_dedup = []
        seen_dedup = set()
        for e in dedup_entities:
            c = re.sub(r"[\r\n\t]+", " ", str(e)).strip()
            c = re.sub(r"\s+", " ", c).strip("\"'`.,;:()[]{}")
            if c and c.lower() not in seen_dedup:
                seen_dedup.add(c.lower())
                clean_dedup.append(c)
        dedup_entities = clean_dedup

        logger.info(
            f"[{index}] Deduplicated {len(raw_entity_strings)} -> {len(dedup_entities)} entities "
            f"(Confidence: {confidence:.2f}, Iterations: {iterations})."
        )

        # Map canonical deduplicated entity to semantic type
        canonical_type_map: Dict[str, str] = {}
        for de in dedup_entities:
            de_clean = de.strip().lower()
            # Find best matching type
            if de_clean in type_mapping:
                canonical_type_map[de] = type_mapping[de_clean]
            else:
                # Substring or fallback match
                matched_type = "Concept"
                for raw_key, raw_type in type_mapping.items():
                    if raw_key in de_clean or de_clean in raw_key:
                        matched_type = raw_type
                        break
                canonical_type_map[de] = matched_type

        # Step 5: Relationship / Triple Extraction
        logger.info(f"[{index}] Extracting relationship triples strictly from deduplicated entities...")
        triples = extract_triples_from_text(
            chunks=scrape_data.chunks,
            deduplicated_entities=dedup_entities,
            mock=self.mock
        )
        logger.info(f"[{index}] Extracted {len(triples)} valid relationship triples.")

        # Step 6: Mermaid Diagram Generation
        mermaid_syntax = triples_to_mermaid(
            triples=triples,
            entity_list=dedup_entities,
            graph_title=f"Knowledge Graph: {scrape_data.title or 'URL ' + str(index)}",
            source_url=url
        )
        mermaid_file = self.mermaid_dir / f"mermaid_{index}.md"
        with open(mermaid_file, "w", encoding="utf-8") as mf:
            mf.write(mermaid_syntax)
        logger.info(f"[{index}] Saved Mermaid diagram to: {mermaid_file}")

        return URLPipelineResult(
            url=url,
            url_index=index,
            raw_entities=raw_entities,
            deduplicated_entities=dedup_entities,
            entity_to_type=canonical_type_map,
            triples=triples,
            mermaid_path=str(mermaid_file),
            mermaid_syntax=mermaid_syntax,
            tags_count=len(dedup_entities),
            confidence_achieved=confidence,
            dedup_iterations=iterations,
            scrape_success=True,
            error_message=None
        )

    def write_tags_csv(self, results: List[URLPipelineResult]) -> int:
        """Write outputs/tags.csv with exact columns: link,tag,tag_type.

        Rules:
        - link: exact source URL
        - tag: exact deduplicated entity string
        - tag_type: semantic category
        - No duplicate tag for the same URL
        """
        seen_pairs = set()
        rows: List[TagRow] = []

        for res in results:
            url = res.url
            for entity in res.deduplicated_entities:
                clean_e = re.sub(r"[\r\n\t]+", " ", str(entity)).strip()
                clean_e = re.sub(r"\s+", " ", clean_e).strip("\"'`.,;:()[]{}")
                if not clean_e:
                    continue
                pair_key = (url, clean_e.lower())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                tag_type = res.entity_to_type.get(clean_e, res.entity_to_type.get(entity, "Concept"))
                rows.append(TagRow(link=url, tag=clean_e, tag_type=tag_type))

        with open(self.tags_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["link", "tag", "tag_type"])
            for r in rows:
                writer.writerow([r.link, r.tag, r.tag_type])

        logger.info(f"Wrote {len(rows)} tag records to: {self.tags_csv_path}")
        return len(rows)

    def write_run_summary(
        self,
        results: List[URLPipelineResult],
        total_tags: int
    ) -> Dict:
        """Generate outputs/run_summary.json with actual execution statistics."""
        total_urls = len(results)
        successful_scrapes = sum(1 for r in results if r.scrape_success)
        failed_scrapes = total_urls - successful_scrapes
        total_raw_entities = sum(len(r.raw_entities) for r in results)
        total_dedup_entities = sum(len(r.deduplicated_entities) for r in results)
        total_triples = sum(len(r.triples) for r in results)

        confidences = [r.confidence_achieved for r in results if r.dedup_iterations > 0]
        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 1.0

        mermaid_files = list(self.mermaid_dir.glob("mermaid_*.md"))

        summary = {
            "execution_mode": "mock" if (self.mock or self.lm is None) else "live_llm",
            "model_configured": os.getenv("LONGCAT_MODEL", "LongCat-2.0") if not self.mock else "mock_engine",
            "target_confidence": self.target_confidence,
            "total_urls_processed": total_urls,
            "successful_scrapes": successful_scrapes,
            "failed_scrapes": failed_scrapes,
            "total_raw_entities": total_raw_entities,
            "total_deduplicated_entities": total_dedup_entities,
            "average_confidence_achieved": round(avg_confidence, 4),
            "total_triples_extracted": total_triples,
            "mermaid_files_generated": len(mermaid_files),
            "tags_csv_rows": total_tags,
            "results_per_url": [
                {
                    "index": r.url_index,
                    "url": r.url,
                    "scrape_success": r.scrape_success,
                    "error_message": r.error_message,
                    "raw_entities_count": len(r.raw_entities),
                    "deduplicated_entities_count": len(r.deduplicated_entities),
                    "confidence": round(r.confidence_achieved, 4),
                    "triples_count": len(r.triples),
                    "mermaid_file": f"mermaid_{r.url_index}.md"
                }
                for r in results
            ]
        }

        with open(self.summary_json_path, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)

        logger.info(f"Saved run summary to: {self.summary_json_path}")
        return summary

    def run(self) -> Tuple[List[URLPipelineResult], Dict]:
        """Execute the complete pipeline across all 10 URLs."""
        urls = self.load_urls()
        results: List[URLPipelineResult] = []

        for i, url in enumerate(urls, start=1):
            res = self.process_url(url, index=i)
            results.append(res)

        total_tags = self.write_tags_csv(results)
        summary = self.write_run_summary(results, total_tags)

        return results, summary
