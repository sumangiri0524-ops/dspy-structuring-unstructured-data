"""Unit tests for Pydantic data schemas."""

import pytest
from pydantic import ValidationError
from src.schemas import EntityWithAttr, Triple, DeduplicationPrediction, TagRow


def test_entity_with_attr_valid():
    ent = EntityWithAttr(entity="pea-barley intercrop", attr_type="crop")
    assert ent.entity == "pea-barley intercrop"
    assert ent.attr_type == "Crop"  # Capitalized normalized


def test_entity_with_attr_empty_fails():
    with pytest.raises(ValidationError):
        EntityWithAttr(entity="", attr_type="Drug")


def test_triple_valid():
    t = Triple(subject="Intercrop", predicate="enhances soil health", object="Soil")
    assert t.subject == "Intercrop"
    assert t.predicate == "enhances soil health"
    assert t.object == "Soil"


def test_triple_predicate_trimmed_to_40_chars():
    long_predicate = "this is an excessively long predicate relationship that exceeds forty characters"
    t = Triple(subject="A", predicate=long_predicate, object="B")
    assert len(t.predicate) <= 40
    assert t.predicate.endswith("...")


def test_deduplication_prediction():
    pred = DeduplicationPrediction(
        deduplicated=["pea-barley intercrop", "soil health"],
        confidence=0.95
    )
    assert len(pred.deduplicated) == 2
    assert pred.confidence >= 0.90


def test_tag_row_fields():
    row = TagRow(link="https://en.wikipedia.org/wiki/Sustainable_agriculture", tag="Soil Health", tag_type="Concept")
    assert row.link.startswith("https://")
    assert row.tag == "Soil Health"
    assert row.tag_type == "Concept"
