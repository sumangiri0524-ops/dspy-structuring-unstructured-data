"""Unit tests for PipelineValidator rules."""

import json
from pathlib import Path
from src.validation import PipelineValidator


def test_urls_count_validation(tmp_path):
    urls_file = tmp_path / "urls.txt"
    # Exactly 10 URLs
    urls_file.write_text("\n".join([f"https://example.com/{i}" for i in range(1, 11)]))

    validator = PipelineValidator(urls_file=str(urls_file), outputs_dir=str(tmp_path))
    passed, msg, details = validator.check_urls_count()
    assert passed is True


def test_csv_columns_validation(tmp_path):
    csv_file = tmp_path / "tags.csv"
    csv_file.write_text("link,tag,tag_type\nhttps://example.com,test,Concept\n")

    validator = PipelineValidator(tags_csv=str(csv_file))
    passed, msg, details = validator.check_csv_columns()
    assert passed is True

    # Bad headers
    csv_file.write_text("url,entity,type\nhttps://example.com,test,Concept\n")
    passed_bad, _, _ = validator.check_csv_columns()
    assert passed_bad is False


def test_csv_duplicate_pairs_detection(tmp_path):
    csv_file = tmp_path / "tags.csv"
    csv_file.write_text(
        "link,tag,tag_type\n"
        "https://example.com,soil health,Concept\n"
        "https://example.com,soil health,Concept\n"
    )
    validator = PipelineValidator(tags_csv=str(csv_file))
    passed, msg, details = validator.check_csv_duplicates()
    assert passed is False
    assert len(details["duplicates"]) == 1
