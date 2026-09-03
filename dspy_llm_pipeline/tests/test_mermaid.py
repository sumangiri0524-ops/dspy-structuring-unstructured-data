"""Unit tests for Mermaid Knowledge Graph generation."""

import re
from src.schemas import Triple
from src.mermaid import triples_to_mermaid, _clean_label


def test_clean_label_truncation():
    short = _clean_label("enhances", max_chars=40)
    assert short == "enhances"
    assert len(short) <= 40

    long_lbl = "is extensively responsible for facilitating and supporting nitrogen transfer"
    truncated = _clean_label(long_lbl, max_chars=40)
    assert len(truncated) <= 40
    assert truncated.endswith("...")


def test_triples_to_mermaid_enforces_entity_whitelist():
    triples = [
        Triple(subject="Sustainable Agriculture", predicate="improves", object="Soil Health"),
        Triple(subject="Sustainable Agriculture", predicate="mentions", object="Unauthorized Random Entity"),
    ]
    entity_list = ["Sustainable Agriculture", "Soil Health"]

    mermaid_str = triples_to_mermaid(triples, entity_list)

    assert "flowchart TD" in mermaid_str
    assert "Sustainable Agriculture" in mermaid_str
    assert "Soil Health" in mermaid_str
    # Unauthorized entity should NOT be included
    assert "Unauthorized Random Entity" not in mermaid_str


def test_triples_to_mermaid_prevents_duplicate_edges():
    triples = [
        Triple(subject="Node A", predicate="connects", object="Node B"),
        Triple(subject="Node A", predicate="connects", object="Node B"),
    ]
    entity_list = ["Node A", "Node B"]

    mermaid_str = triples_to_mermaid(triples, entity_list)
    # Count occurrences of the edge connector
    matches = re.findall(r'-- "connects" -->', mermaid_str)
    assert len(matches) == 1
