"""Unit tests for DSPy extraction signatures, confidence loop, and triple extraction."""

from src.schemas import EntityWithAttr
from src.dspy_modules import (
    extract_entities_from_text,
    deduplicate_with_lm,
    extract_triples_from_text,
    MockPredictor
)


def test_mock_predictor_extracts_entities():
    sample_text = (
        "Sustainable agriculture enhances soil health and protects against chronic pain. "
        "The FAO recommends pea-barley intercrop for nitrogen uptake."
    )
    entities = MockPredictor.extract_entities(sample_text)
    assert len(entities) > 0
    names = [e.entity.lower() for e in entities]
    assert any("sustainable agriculture" in n or "soil health" in n or "intercrop" in n or "fao" in n for n in names)


def test_confidence_loop_achieves_target_confidence():
    noisy_items = [
        "PB IC",
        "pea-barley intercrop",
        "pea-barley intercrops",
        "soil health",
        "Soil Health",
        "nitrogen uptake"
    ]
    dedup_list, confidence, iterations = deduplicate_with_lm(
        items=noisy_items,
        batch_size=10,
        target_confidence=0.90,
        mock=True
    )
    # Deduplication must reduce duplicates
    assert len(dedup_list) < len(noisy_items)
    # Confidence loop requirement: target confidence >= 0.90
    assert confidence >= 0.90
    assert iterations >= 1


def test_triples_strictly_use_valid_entities():
    chunks = [
        "Sustainable agriculture directly enhances soil health and improves nitrogen uptake across fields."
    ]
    valid_entities = ["Sustainable agriculture", "soil health", "nitrogen uptake"]

    triples = extract_triples_from_text(chunks, valid_entities, mock=True)
    valid_set = {e.lower() for e in valid_entities}

    for t in triples:
        assert t.subject.lower() in valid_set
        assert t.object.lower() in valid_set
        assert len(t.predicate) <= 40
