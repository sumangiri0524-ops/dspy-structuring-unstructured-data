"""Pydantic data schemas for DSPy Entity Extraction, Deduplication, and Triples."""

import re
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator


class EntityWithAttr(BaseModel):
    """Named entity with semantic attribute type as defined in the assignment PDF."""

    entity: str = Field(
        ...,
        description="the named entity (exact string from text)",
        min_length=1,
        max_length=80
    )
    attr_type: str = Field(
        ...,
        description="semantic type (e.g. Drug, Disease, Crop, Process, Measurement, Concept, Organization, Location)",
        min_length=1
    )

    @field_validator("entity", mode="before")
    @classmethod
    def clean_entity(cls, v: Any) -> str:
        cleaned = re.sub(r"[\r\n\t]+", " ", str(v)).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip("\"'`.,;:()[]{}")
        if not cleaned:
            raise ValueError("Entity string cannot be empty")
        if len(cleaned) > 80:
            cleaned = cleaned[:80].rstrip()
        return cleaned

    @field_validator("attr_type", mode="before")
    @classmethod
    def clean_attr_type(cls, v: Any) -> str:
        cleaned = re.sub(r"[\r\n\t]+", " ", str(v)).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip("\"'`.,;:()[]{}")
        if not cleaned:
            return "Concept"
        # Normalize casing (e.g. "crop" -> "Crop")
        return cleaned[:1].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()


class Triple(BaseModel):
    """Knowledge graph relation triple (subject, predicate, object)."""

    subject: str = Field(
        ...,
        description="Source entity (must match deduplicated entity)",
        min_length=1
    )
    predicate: str = Field(
        ...,
        description="Relationship predicate between subject and object (trimmed to max 40 chars)",
        min_length=1,
        max_length=40
    )
    object: str = Field(
        ...,
        description="Target entity (must match deduplicated entity)",
        min_length=1
    )

    @field_validator("subject", "object", mode="before")
    @classmethod
    def clean_nodes(cls, v: Any) -> str:
        cleaned = re.sub(r"[\r\n\t]+", " ", str(v)).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip("\"'`.,;:()[]{}")

    @field_validator("predicate", mode="before")
    @classmethod
    def trim_predicate(cls, v: Any) -> str:
        cleaned = re.sub(r"[\r\n\t]+", " ", str(v)).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        trimmed = cleaned.strip("\"'`.,;:()[]{}")
        if len(trimmed) > 40:
            trimmed = trimmed[:37].rstrip() + "..."
        return trimmed or "relates_to"


class DeduplicationPrediction(BaseModel):
    """Structured result returned by entity deduplication module."""

    deduplicated: List[str] = Field(
        default_factory=list,
        description="List of canonical deduplicated entity strings"
    )
    confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence score of deduplication, target >= 0.90"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Explanation or criteria applied for grouping aliases"
    )


class TagRow(BaseModel):
    """Row model for tags.csv: link, tag, tag_type."""

    link: str = Field(..., description="Exact source URL")
    tag: str = Field(..., description="Exact deduplicated entity string")
    tag_type: str = Field(..., description="Semantic category")


class ScrapedURLData(BaseModel):
    """Scraped content and metadata for a single URL."""

    url: str
    index: int
    title: str = ""
    status_code: Optional[int] = None
    success: bool = False
    error_message: Optional[str] = None
    cleaned_text: str = ""
    chunks: List[str] = Field(default_factory=list)


class URLPipelineResult(BaseModel):
    """Complete extraction and graph generation results for a single URL."""

    url: str
    url_index: int
    raw_entities: List[EntityWithAttr] = Field(default_factory=list)
    deduplicated_entities: List[str] = Field(default_factory=list)
    entity_to_type: Dict[str, str] = Field(default_factory=dict)
    triples: List[Triple] = Field(default_factory=list)
    mermaid_path: Optional[str] = None
    mermaid_syntax: str = ""
    tags_count: int = 0
    confidence_achieved: float = 0.0
    dedup_iterations: int = 1
    scrape_success: bool = True
    error_message: Optional[str] = None
