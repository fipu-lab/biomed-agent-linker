import json
import os
import argparse
from typing import List, Dict, Set, Optional

# Global cache for loaded artifacts
_cui2types = None

ST21PV_TYPES = {
    "ANAT": [
        "T017",
        "T029",
        "T023",
        "T030",
        "T031",
        "T022",
        "T025",
        "T026",
        "T018",
        "T021",
        "T024",
    ],
    "CHEM": [
        "T116",
        "T195",
        "T123",
        "T122",
        "T118",
        "T103",
        "T120",
        "T104",
        "T111",
        "T196",
        "T126",
        "T131",
        "T125",
        "T129",
        "T130",
        "T197",
        "T119",
        "T124",
        "T114",
        "T109",
        "T115",
        "T121",
        "T192",
        "T110",
        "T127",
    ],
    "DEVI": ["T203", "T074", "T075"],
    "DISO": [
        "T020",
        "T190",
        "T049",
        "T019",
        "T047",
        "T050",
        "T033",
        "T037",
        "T048",
        "T191",
        "T046",
        "T184",
    ],
    "GENE": ["T087", "T088", "T028", "T085", "T086"],
    "GEOG": ["T083"],
    "LIVB": [
        "T100",
        "T011",
        "T008",
        "T194",
        "T007",
        "T012",
        "T204",
        "T099",
        "T013",
        "T004",
        "T096",
        "T016",
        "T015",
        "T001",
        "T101",
        "T014",
        "T010",
        "T005",
        "T002",
    ],
    "OBJC": ["T071", "T168", "T073", "T072", "T167"],
    "OCCU": ["T091"],
    "ORGA": ["T093", "T092", "T094", "T095"],
    "PHEN": [
        "T038",
        "T069",
        "T068",
        "T034",
        "T070",
        "T067",
        "T066",
        "T065",
        "T052",
        "T053",
        "T063",
        "T062",
        "T061",
        "T060",
        "T059",
        "T058",
        "T057",
        "T056",
        "T055",
        "T054",
        "T051",
        "T041",
        "T040",
        "T039",
        "T043",
        "T201",
        "T045",
        "T044",
        "T042",
        "T032",
        "T064",
    ],
    "PHYS": ["T032", "T201", "T033"],
    "PROC": [
        "T060",
        "T065",
        "T058",
        "T059",
        "T063",
        "T062",
        "T061",
        "T057",
        "T056",
        "T055",
        "T054",
        "T053",
        "T052",
        "T051",
        "T064",
    ],
}


def _load_artifacts():
    """Load UMLS artifacts if not already loaded."""
    global _cui2types

    if _cui2types is None:
        artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        cui2types_path = os.path.join(artifacts_dir, "cui2types.json")

        with open(cui2types_path, "r", encoding="utf-8") as f:
            # cui2types contains sets as lists, convert back to sets
            data = json.load(f)
            _cui2types = {cui: set(types) for cui, types in data.items()}


def get_cui_semantic_types(cui: str) -> Set[str]:
    """
    Get semantic types (TUIs) for a given CUI.

    Args:
        cui: UMLS Concept Unique Identifier

    Returns:
        Set of TUIs (semantic type unique identifiers) for the CUI
    """
    _load_artifacts()
    return _cui2types.get(cui, set())


def filter_by_semantic_type(
    candidates: List[Dict], target_types: List[str] = None
) -> List[Dict]:
    """
    Filter candidates by semantic types.

    Args:
        candidates: List of candidate dictionaries (must contain 'cui' field)
        target_types: List of target ST21pv type codes (e.g., ['DISO', 'CHEM'])
                     If None, no filtering is applied

    Returns:
        Filtered list of candidates that match target semantic types
    """
    if target_types is None:
        return candidates

    _load_artifacts()

    # Get target TUIs from ST21pv types
    target_tuis = set()
    for st21_type in target_types:
        if st21_type in ST21PV_TYPES:
            target_tuis.update(ST21PV_TYPES[st21_type])

    if not target_tuis:
        return candidates

    filtered_candidates = []
    for candidate in candidates:
        cui = candidate.get("cui")
        if not cui:
            continue

        cui_types = get_cui_semantic_types(cui)

        # Check if any of the CUI's types match target types
        if cui_types.intersection(target_tuis):
            # Add type information to candidate
            candidate_copy = candidate.copy()
            candidate_copy["semantic_types"] = list(cui_types)
            candidate_copy["matching_st21_types"] = [
                st21_type
                for st21_type, tuis in ST21PV_TYPES.items()
                if cui_types.intersection(set(tuis))
            ]
            filtered_candidates.append(candidate_copy)

    return filtered_candidates


def boost_by_semantic_type(
    candidates: List[Dict], target_types: List[str] = None, boost_factor: float = 1.25
) -> List[Dict]:
    """
    Boost candidate scores based on semantic type matching.

    Args:
        candidates: List of candidate dictionaries with 'cui' and 'similarity_score' fields
        target_types: List of target ST21pv type codes
        boost_factor: Factor to multiply scores for matching types

    Returns:
        List of candidates with boosted scores, sorted by score
    """
    if target_types is None:
        return candidates

    _load_artifacts()

    # Get target TUIs
    target_tuis = set()
    for st21_type in target_types:
        if st21_type in ST21PV_TYPES:
            target_tuis.update(ST21PV_TYPES[st21_type])

    boosted_candidates = []
    for candidate in candidates:
        candidate_copy = candidate.copy()
        cui = candidate.get("cui")
        original_score = candidate.get("similarity_score", 0.0)

        if cui and target_tuis:
            cui_types = get_cui_semantic_types(cui)
            if cui_types.intersection(target_tuis):
                # Clip boosted score to [0, 1.0]
                candidate_copy["similarity_score"] = min(
                    original_score * boost_factor, 1.0
                )
                candidate_copy["type_boosted"] = True
            else:
                candidate_copy["type_boosted"] = False

        boosted_candidates.append(candidate_copy)

    # Sort by boosted scores
    boosted_candidates.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    return boosted_candidates


def get_st21_type_mapping(cui: str) -> List[str]:
    """
    Get ST21pv type codes for a given CUI.

    Args:
        cui: UMLS Concept Unique Identifier

    Returns:
        List of ST21pv type codes that this CUI belongs to
    """
    cui_types = get_cui_semantic_types(cui)

    st21_types = []
    for st21_type, tuis in ST21PV_TYPES.items():
        if cui_types.intersection(set(tuis)):
            st21_types.append(st21_type)

    return st21_types


def main():
    parser = argparse.ArgumentParser(prog="semantic_tools", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_types = sub.add_parser("types")
    p_types.add_argument("cui")

    p_map = sub.add_parser("st21map")
    p_map.add_argument("cui")

    p_filter = sub.add_parser("filter")
    p_filter.add_argument("-t", "--types", nargs="+", required=True)
    p_filter.add_argument(
        "-c", "--candidates-json", required=True, help="Path to candidates JSON list"
    )

    p_boost = sub.add_parser("boost")
    p_boost.add_argument("-t", "--types", nargs="+", required=True)
    p_boost.add_argument("-c", "--candidates-json", required=True)
    p_boost.add_argument("--factor", type=float, default=1.5)

    args = parser.parse_args()

    if args.cmd == "types":
        out = list(get_cui_semantic_types(args.cui))
        print(json.dumps(out, ensure_ascii=False))
        return

    if args.cmd == "st21map":
        out = get_st21_type_mapping(args.cui)
        print(json.dumps(out, ensure_ascii=False))
        return

    if args.cmd in {"filter", "boost"}:
        with open(args.candidates_json, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        if args.cmd == "filter":
            out = filter_by_semantic_type(candidates, target_types=args.types)
        else:
            out = boost_by_semantic_type(
                candidates, target_types=args.types, boost_factor=args.factor
            )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
