#!/usr/bin/env python3
"""
Starter code for CSE 132A HW3.

Students must implement:
  * infer_fds_from_sample
  * minimal_cover
  * synthesize_3nf
  * decompose_bcnf

Use the provided helper functions to keep the CLI contract and output format.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3NF / BCNF decomposition tool")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--to", required=True, choices=["3nf", "bcnf"], help="Target normal form")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_fd_json(fd_json: Sequence[dict]) -> List[Tuple[frozenset, frozenset]]:
    """Convert JSON list of {left: [...], right: [...]} into frozensets."""
    fds: List[Tuple[frozenset, frozenset]] = []
    for idx, fd in enumerate(fd_json):
        if "left" not in fd or "right" not in fd:
            raise ValueError(f"FD at index {idx} must contain 'left' and 'right'.")
        left = tuple(fd["left"])
        right = tuple(fd["right"])
        if not left or not right:
            raise ValueError(f"FD at index {idx} must have non-empty left/right sides.")
        fds.append((frozenset(left), frozenset(right)))
    return fds


# ---------------------------------------------------------------------------
# TODO functions for students
# ---------------------------------------------------------------------------

def infer_fds_from_sample(attributes: Sequence[str], sample_rows: Sequence[dict]) -> List[Tuple[frozenset, frozenset]]:
    """
    TODO: infer functional dependencies from sample_rows.

    Return a list of (lhs, rhs) where lhs and rhs are frozensets of attribute names.
    RHS entries must be singleton sets in the final minimal cover.
    """
    raise NotImplementedError("infer_fds_from_sample must be implemented by the student.")


def minimal_cover(fds: Sequence[Tuple[frozenset, frozenset]]) -> List[Tuple[frozenset, frozenset]]:
    """
    TODO: compute a minimal cover for the given functional dependencies.

    Requirements:
      * Each RHS must be a singleton.
      * No redundant attributes on LHS.
      * No redundant FDs.
    """
    raise NotImplementedError("minimal_cover must be implemented by the student.")


def synthesize_3nf(attributes: Sequence[str], minimal_fds: Sequence[Tuple[frozenset, frozenset]]) -> Iterable[Iterable[str]]:
    """
    TODO: implement the 3NF synthesis algorithm.

    Return an iterable of relations (each relation is itself an iterable of attributes).
    """
    raise NotImplementedError("synthesize_3nf must be implemented by the student.")


def decompose_bcnf(attributes: Sequence[str], minimal_fds: Sequence[Tuple[frozenset, frozenset]]) -> Iterable[Iterable[str]]:
    """
    TODO: implement the BCNF decomposition algorithm.

    Return an iterable of relations (each relation is itself an iterable of attributes).
    """
    raise NotImplementedError("decompose_bcnf must be implemented by the student.")


# ---------------------------------------------------------------------------
# Output validation helpers
# ---------------------------------------------------------------------------

def _ensure_attribute_subset(attrs: Iterable[str], universe: Sequence[str], context: str) -> List[str]:
    allowed = set(universe)
    normalized = sorted(attrs)
    if not normalized:
        raise AssertionError(f"{context}: relation must contain at least one attribute")
    for attr in normalized:
        if attr not in allowed:
            raise AssertionError(f"{context}: attribute '{attr}' not in original schema {universe}")
    return normalized


def normalize_relations(relations: Iterable[Iterable[str]], attributes: Sequence[str]) -> List[List[str]]:
    if relations is None:
        raise AssertionError("Expected a collection of relations, got None")
    normalized: List[List[str]] = []
    seen = set()
    for idx, rel in enumerate(relations):
        if isinstance(rel, dict):
            raise AssertionError(f"Relation #{idx} should be an iterable of attribute names, not a dict")
        normalized_attrs = _ensure_attribute_subset(rel, attributes, f"Relation #{idx}")
        key = tuple(normalized_attrs)
        if key in seen:
            raise AssertionError(f"Duplicate relation detected: {normalized_attrs}")
        seen.add(key)
        normalized.append(normalized_attrs)
    normalized.sort(key=lambda xs: "".join(xs))
    return normalized


def normalize_fds_for_output(fds: Iterable[Tuple[Iterable[str], Iterable[str]]], attributes: Sequence[str]) -> List[Tuple[List[str], List[str]]]:
    normalized = []
    for idx, (lhs, rhs) in enumerate(fds):
        lhs_norm = _ensure_attribute_subset(lhs, attributes, f"FD #{idx} (LHS)")
        rhs_norm = _ensure_attribute_subset(rhs, attributes, f"FD #{idx} (RHS)")
        if len(rhs_norm) != 1:
            raise AssertionError(f"FD #{idx} must have singleton RHS after minimal cover, got {rhs_norm}")
        normalized.append((lhs_norm, rhs_norm))
    return normalized


def build_output_payload(relations: List[List[str]], fds: List[Tuple[List[str], List[str]]]) -> dict:
    payload = {
        "relations": [
            {"name": f"R{i+1}", "attributes": rel}
            for i, rel in enumerate(relations)
        ],
        "inferredFDs": [
            {"left": lhs, "right": rhs}
            for lhs, rhs in fds
        ],
    }
    # Assertions to guarantee the schema is correct.
    assert isinstance(payload["relations"], list), "relations must be a list"
    for rel in payload["relations"]:
        assert isinstance(rel, dict), "Each relation must be a dict"
        assert "attributes" in rel, "Missing 'attributes' key in relation"
        assert rel["attributes"] == sorted(rel["attributes"]), "Attributes must be sorted"
    assert isinstance(payload["inferredFDs"], list), "inferredFDs must be a list"
    for fd in payload["inferredFDs"]:
        assert set(fd.keys()) == {"left", "right"}, "Each FD must have 'left' and 'right'"
        assert fd["left"] == sorted(fd["left"]), "FD LHS must be sorted"
        assert fd["right"] == sorted(fd["right"]), "FD RHS must be sorted"
        assert len(fd["right"]) == 1, "FD RHS must be singleton"
    return payload


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    relation_name = data.get("relationName", "R")
    attributes = data.get("attributes", [])
    fd_json = data.get("functionalDependencies") or []
    sample_rows = data.get("sample_data") or []

    if not attributes:
        raise ValueError("Input must include a non-empty list of attributes.")

    fds = parse_fd_json(fd_json)
    if not fds:
        if not sample_rows:
            raise ValueError("Either functionalDependencies or sample_data must be provided.")
        print("# No FDs supplied. You must implement infer_fds_from_sample().", file=sys.stderr)
        fds = infer_fds_from_sample(attributes, sample_rows)

    print("# Computing a minimal cover...", file=sys.stderr)
    minimal = minimal_cover(fds)

    print(f"# Decomposing relation {relation_name} to {args.to.upper()}...", file=sys.stderr)
    if args.to == "3nf":
        raw_relations = synthesize_3nf(attributes, minimal)
    else:
        raw_relations = decompose_bcnf(attributes, minimal)

    normalized_relations = normalize_relations(raw_relations, attributes)
    normalized_fds = normalize_fds_for_output(minimal, attributes)
    payload = build_output_payload(normalized_relations, normalized_fds)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_file:
            json.dump(payload, out_file, ensure_ascii=False, indent=2 if args.pretty else None)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
