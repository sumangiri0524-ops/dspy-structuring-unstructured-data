"""Automated validation suite verifying all constraints specified in the assignment PDF."""

import re
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class PipelineValidator:
    """Automated validator enforcing all 9 invariant rules from the assignment."""

    def __init__(
        self,
        urls_file: str = "data/urls.txt",
        outputs_dir: str = "outputs",
        tags_csv: str = "outputs/tags.csv",
        summary_json: str = "outputs/run_summary.json"
    ):
        self.urls_file = Path(urls_file)
        self.outputs_dir = Path(outputs_dir)
        self.mermaid_dir = self.outputs_dir / "mermaid"
        self.tags_csv_path = Path(tags_csv)
        self.summary_json_path = Path(summary_json)
        self.results: Dict[str, Dict[str, Any]] = {}

    def validate_all(self) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        """Run all validation checks and return (overall_success, details)."""
        checks = [
            ("exact_10_urls", self.check_urls_count),
            ("exact_10_mermaid_files", self.check_mermaid_files_count),
            ("valid_mermaid_structure", self.check_mermaid_structure),
            ("mermaid_edge_labels_length", self.check_mermaid_edge_labels),
            ("no_duplicate_mermaid_edges", self.check_no_duplicate_edges),
            ("only_dedup_entities_as_nodes", self.check_only_dedup_entities_in_mermaid),
            ("csv_exact_columns", self.check_csv_columns),
            ("csv_no_duplicate_link_tag_pairs", self.check_csv_duplicates),
            ("all_required_urls_represented", self.check_all_urls_represented),
            ("run_summary_json_valid", self.check_run_summary_json),
        ]

        all_passed = True
        for name, check_fn in checks:
            try:
                passed, msg, details = check_fn()
                self.results[name] = {
                    "passed": passed,
                    "message": msg,
                    "details": details
                }
                if not passed:
                    all_passed = False
            except Exception as e:
                all_passed = False
                self.results[name] = {
                    "passed": False,
                    "message": f"Exception raised: {str(e)}",
                    "details": {}
                }

        return all_passed, self.results

    def check_urls_count(self) -> Tuple[bool, str, Any]:
        """Check that data/urls.txt contains exactly 10 URLs."""
        if not self.urls_file.exists():
            return False, f"URLs file missing: {self.urls_file}", {}
        with open(self.urls_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        count = len(urls)
        passed = (count == 10)
        return passed, f"Found {count} URLs (expected 10)", {"count": count, "urls": urls}

    def check_mermaid_files_count(self) -> Tuple[bool, str, Any]:
        """Check that exactly 10 Mermaid files (mermaid_1.md to mermaid_10.md) exist."""
        if not self.mermaid_dir.exists():
            return False, "Mermaid directory does not exist", {}
        files = list(self.mermaid_dir.glob("mermaid_*.md"))
        expected_names = {f"mermaid_{i}.md" for i in range(1, 11)}
        actual_names = {f.name for f in files}

        missing = expected_names - actual_names
        passed = (len(files) == 10 and not missing)
        return passed, f"Found {len(files)} Mermaid files (missing: {missing or 'none'})", {
            "found_count": len(files),
            "missing": list(missing)
        }

    def check_mermaid_structure(self) -> Tuple[bool, str, Any]:
        """Verify that each Mermaid file contains valid flowchart syntax."""
        files = sorted(self.mermaid_dir.glob("mermaid_*.md"))
        if not files:
            return False, "No Mermaid files found to validate", {}

        invalid = []
        for f in files:
            content = f.read_text(encoding="utf-8")
            # Must contain ```mermaid and flowchart
            if "```mermaid" not in content or "flowchart" not in content:
                invalid.append(f.name)
            # Must be closed with ```
            if not content.strip().endswith("```"):
                invalid.append(f.name)

        passed = (len(invalid) == 0)
        return passed, f"All {len(files)} files have valid Mermaid block structure" if passed else f"Invalid syntax in {invalid}", {
            "invalid_files": invalid
        }

    def check_mermaid_edge_labels(self) -> Tuple[bool, str, Any]:
        """Verify that all relationship labels are <= 40 characters."""
        files = sorted(self.mermaid_dir.glob("mermaid_*.md"))
        violating_labels = []

        for f in files:
            content = f.read_text(encoding="utf-8")
            # Extract labels in -- "..." --> or -- ... --> or -->|...|
            matches = re.findall(r'--\s*"?([^"\n\r>]+?)"?\s*-->', content)
            for m in matches:
                lbl = m.strip()
                if len(lbl) > 40:
                    violating_labels.append({"file": f.name, "label": lbl, "length": len(lbl)})

        passed = (len(violating_labels) == 0)
        return passed, "All edge labels trimmed to <= 40 characters" if passed else f"Found {len(violating_labels)} labels exceeding 40 chars", {
            "violations": violating_labels
        }

    def check_no_duplicate_edges(self) -> Tuple[bool, str, Any]:
        """Verify that no Mermaid diagram contains duplicate directed edges."""
        files = sorted(self.mermaid_dir.glob("mermaid_*.md"))
        duplicates = []

        for f in files:
            content = f.read_text(encoding="utf-8")
            edge_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)(?:\[.*?\])?\s*--\s*"?(.*?)"?\s*-->\s*([a-zA-Z0-9_]+)', re.M)
            seen_edges = set()
            for match in edge_pattern.finditer(content):
                src, pred, dst = match.group(1), match.group(2).lower(), match.group(3)
                key = (src, pred, dst)
                if key in seen_edges:
                    duplicates.append({"file": f.name, "edge": key})
                seen_edges.add(key)

        passed = (len(duplicates) == 0)
        return passed, "Zero duplicate edges across all Mermaid diagrams" if passed else f"Found duplicate edges in {len(duplicates)} occurrences", {
            "duplicates": duplicates
        }

    def check_only_dedup_entities_in_mermaid(self) -> Tuple[bool, str, Any]:
        """Verify that all node text in Mermaid graphs matches deduplicated entity tags."""
        if not self.tags_csv_path.exists():
            return False, "tags.csv does not exist to verify entity membership", {}

        # Load valid deduplicated tags per URL
        url_to_tags: Dict[str, Set[str]] = {}
        with open(self.tags_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row["link"].strip()
                tag = re.sub(r"[\r\n\t]+", " ", row["tag"]).strip()
                tag = re.sub(r"\s+", " ", tag).lower().strip("\"'`.,;:()[]{}")
                if url not in url_to_tags:
                    url_to_tags[url] = set()
                url_to_tags[url].add(tag)

        # Load URLs to match with index
        with open(self.urls_file, "r", encoding="utf-8") as uf:
            urls = [line.strip() for line in uf if line.strip() and not line.strip().startswith("#")]

        unauthorized_nodes = []
        for i, url in enumerate(urls, start=1):
            mermaid_file = self.mermaid_dir / f"mermaid_{i}.md"
            if not mermaid_file.exists():
                continue
            content = mermaid_file.read_text(encoding="utf-8")
            valid_set = url_to_tags.get(url, set())

            # Extract node labels in format id["Label"]
            node_labels = re.findall(r'[a-zA-Z0-9_]+\["([^"\n\r]+)"\]', content)
            for lbl in node_labels:
                lbl_clean = re.sub(r"[\r\n\t]+", " ", lbl).strip()
                lbl_clean = re.sub(r"\s+", " ", lbl_clean).lower().strip("\"'`.,;:()[]{}")
                # Ignore system messages like error / empty node
                if "scraping failed" in lbl_clean or "no entities" in lbl_clean or "no content" in lbl_clean:
                    continue
                if lbl_clean not in valid_set:
                    unauthorized_nodes.append({
                        "file": mermaid_file.name,
                        "url": url,
                        "node": lbl
                    })

        passed = (len(unauthorized_nodes) == 0)
        return passed, "All Mermaid graph nodes strictly originate from deduplicated entity list" if passed else f"Found unauthorized nodes: {unauthorized_nodes}", {
            "unauthorized_nodes": unauthorized_nodes
        }

    def check_csv_columns(self) -> Tuple[bool, str, Any]:
        """Check tags.csv has EXACT columns: link,tag,tag_type."""
        if not self.tags_csv_path.exists():
            return False, f"tags.csv not found at: {self.tags_csv_path}", {}

        with open(self.tags_csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)

        expected = ["link", "tag", "tag_type"]
        passed = (header == expected)
        return passed, f"CSV header is {header} (expected {expected})", {"header": header}

    def check_csv_duplicates(self) -> Tuple[bool, str, Any]:
        """Check tags.csv has NO duplicate (link, tag) pairs."""
        if not self.tags_csv_path.exists():
            return False, "tags.csv not found", {}

        seen = set()
        duplicates = []
        total_rows = 0

        with open(self.tags_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                total_rows += 1
                pair = (row.get("link", "").strip(), row.get("tag", "").strip().lower())
                if pair in seen:
                    duplicates.append({"row": row_idx, "pair": pair})
                seen.add(pair)

        passed = (len(duplicates) == 0)
        return passed, f"No duplicate (link, tag) pairs across {total_rows} rows" if passed else f"Found {len(duplicates)} duplicates", {
            "total_rows": total_rows,
            "duplicates": duplicates
        }

    def check_all_urls_represented(self) -> Tuple[bool, str, Any]:
        """Check all 10 required URLs are processed in summary and accounted for."""
        if not self.summary_json_path.exists():
            return False, "run_summary.json missing", {}

        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        total = summary.get("total_urls_processed", 0)
        passed = (total == 10)
        return passed, f"Summary accounts for {total}/10 processed URLs", {"summary_total": total}

    def check_run_summary_json(self) -> Tuple[bool, str, Any]:
        """Check outputs/run_summary.json exists, is valid JSON, and has all metrics."""
        if not self.summary_json_path.exists():
            return False, "run_summary.json missing", {}

        try:
            with open(self.summary_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            required_keys = [
                "execution_mode", "target_confidence", "total_urls_processed",
                "successful_scrapes", "failed_scrapes", "total_raw_entities",
                "total_deduplicated_entities", "average_confidence_achieved",
                "total_triples_extracted", "mermaid_files_generated", "tags_csv_rows"
            ]
            missing = [k for k in required_keys if k not in data]
            passed = (len(missing) == 0)
            return passed, f"run_summary.json has valid metrics (missing: {missing or 'none'})", data
        except Exception as e:
            return False, f"JSON parsing error: {e}", {}


def print_validation_report(results: Dict[str, Dict[str, Any]]):
    """Print a clean CLI validation table."""
    print("\n" + "=" * 75)
    print("                    AUTOMATED VALIDATION AUDIT                    ")
    print("=" * 75)
    print(f"{'Check / Invariant Rule':<38} | {'Status':<8} | {'Details'}")
    print("-" * 75)

    all_pass = True
    for rule, res in results.items():
        status = "PASS" if res["passed"] else "FAIL"
        if not res["passed"]:
            all_pass = False
        msg = res["message"][:45]
        print(f"{rule:<38} | {status:<8} | {msg}")

    print("=" * 75)
    print(f"OVERALL RESULT: {'ALL PASS [100%]' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass
