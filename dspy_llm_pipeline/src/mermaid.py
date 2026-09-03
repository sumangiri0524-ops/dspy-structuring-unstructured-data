"""Mermaid Knowledge Graph generation with strict node whitelisting and label constraints."""

import re
import hashlib
from typing import List, Set, Tuple
from src.schemas import Triple


def _clean_label(label: str, max_chars: int = 40) -> str:
    """Trim and sanitize edge relationship label to <= 40 characters."""
    # Remove characters that break Mermaid syntax
    clean = re.sub(r'["`\[\]\(\)\{\}\n\r\t]', " ", label)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) > max_chars:
        clean = clean[: max_chars - 3].rstrip() + "..."
    return clean or "relates to"


def _clean_node_text(text: str) -> str:
    """Escape characters inside node text representation."""
    clean = re.sub(r'["`\n\r\t]', " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _make_node_id(text: str) -> str:
    """Create a deterministic safe alphanumeric ID for Mermaid nodes."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", text.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    # Add hash suffix to avoid collisions while keeping readability
    h = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:6]
    return f"id_{safe[:20]}_{h}" if safe else f"id_{h}"


def triples_to_mermaid(
    triples: List[Triple],
    entity_list: List[str],
    graph_title: str = "Knowledge Graph",
    source_url: str = "",
    max_edges: int = 40
) -> str:
    """Generate valid Mermaid flowchart syntax from triples and deduplicated entities.

    Strict Requirements:
    1. Valid Mermaid flowchart syntax testable in Mermaid Live Editor.
    2. Only allows entities from our deduplicated list as nodes (entity_set check).
    3. Edge labels trimmed to <= 40 chars.
    4. Avoid duplicate edges.
    """
    # Whitelist of deduplicated entities (case-insensitive lookup)
    entity_lookup = {
        re.sub(r"[\r\n\t]+", " ", e).strip().lower(): re.sub(r"[\r\n\t]+", " ", e).strip().strip("\"'`.,;:()[]{}")
        for e in entity_list
        if e.strip()
    }
    entity_set = set(entity_lookup.keys())

    lines = [
        "```mermaid",
        "flowchart TD",
        f"    %% {graph_title}",
    ]
    if source_url:
        lines.append(f"    %% Source: {source_url}")

    if not entity_list:
        lines.append('    empty_node["No entities extracted or scraping blocked"]')
        lines.append("```")
        return "\n".join(lines)

    # Track defined nodes and seen edges
    seen_edges: Set[Tuple[str, str, str]] = set()
    node_id_map = {}
    valid_edge_count = 0

    for triple in triples:
        if valid_edge_count >= max_edges:
            break

        src_clean = re.sub(r"[\r\n\t]+", " ", triple.subject).strip().lower()
        dst_clean = re.sub(r"[\r\n\t]+", " ", triple.object).strip().lower()

        # Strict validation: ONLY entities from deduplicated list are allowed as nodes
        if src_clean not in entity_set or dst_clean not in entity_set:
            continue

        if src_clean == dst_clean:
            continue

        # Get canonical casing from entity list
        canonical_src = entity_lookup[src_clean]
        canonical_dst = entity_lookup[dst_clean]

        lbl = _clean_label(triple.predicate, max_chars=40)
        edge_key = (src_clean, lbl.lower(), dst_clean)

        # Avoid duplicate edges
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        # Ensure node IDs exist
        if src_clean not in node_id_map:
            node_id_map[src_clean] = (
                _make_node_id(canonical_src),
                _clean_node_text(canonical_src)
            )
        if dst_clean not in node_id_map:
            node_id_map[dst_clean] = (
                _make_node_id(canonical_dst),
                _clean_node_text(canonical_dst)
            )

        src_id, src_label = node_id_map[src_clean]
        dst_id, dst_label = node_id_map[dst_clean]

        # Valid Mermaid syntax: src_id["Label"] -- "pred" --> dst_id["Label"]
        lines.append(f'    {src_id}["{src_label}"] -- "{lbl}" --> {dst_id}["{dst_label}"]')
        valid_edge_count += 1

    # If no valid edges were found but entities exist, render standalone entity nodes
    if valid_edge_count == 0:
        for e in entity_list[:10]:
            e_clean = e.strip().lower()
            if e_clean in entity_lookup:
                can_name = entity_lookup[e_clean]
                n_id = _make_node_id(can_name)
                n_lbl = _clean_node_text(can_name)
                lines.append(f'    {n_id}["{n_lbl}"]')

    lines.append("```")
    return "\n".join(lines)
