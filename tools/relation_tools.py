import json
import os
import argparse
from typing import List, Dict, Set, Optional

try:
    from .lexical_tools import get_cui_preferred_term
except ImportError:
    # Fallback for direct script execution
    import sys as _sys
    import os as _os

    _sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
    from tools.lexical_tools import get_cui_preferred_term
try:
    from .semantic_tools import get_cui_semantic_types, ST21PV_TYPES
except ImportError:
    import sys as _sys
    import os as _os

    _sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
    from tools.semantic_tools import get_cui_semantic_types, ST21PV_TYPES

_cui2neighbors = None


def _load_artifacts():
    global _cui2neighbors

    if _cui2neighbors is None:
        artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        cui2neighbors_path = os.path.join(artifacts_dir, "cui2neighbors.json")

        with open(cui2neighbors_path, "r", encoding="utf-8") as f:
            _cui2neighbors = json.load(f)


def _is_noise_term(term: str) -> bool:
    # Filter administrative/terminology scaffolding
    if not term:
        return True
    t = term.strip()
    admin_patterns = [
        "Terminology",
        "Codelist",
        "Value Terminology",
        "Research Participant Metadata",
        "Yes (indicator)",
        "Not Applicable",
        "Unknown",
        "Qualifier",
        "Body Site Modifier",
    ]
    for p in admin_patterns:
        if p in t:
            return True

    # LOINC-like axis tokens
    loinc_tokens = [":Find:", ":PrThr:", ":Qn", ":Ord", ":Nom", ":Temp:", "Pt:"]
    if any(tok in t for tok in loinc_tokens):
        return True
    return False


def get_cui_neighbors(cui: str) -> List[str]:
    _load_artifacts()
    return _cui2neighbors.get(cui, [])


def find_related_concepts(
    cui: str, max_depth: int = 1, max_results: int = 10
) -> List[Dict[str, any]]:
    """
    Find concepts related to a given CUI through UMLS relations.

    Args:
        cui: UMLS Concept Unique Identifier
        max_depth: Maximum depth of relation traversal
        max_results: Maximum number of related concepts to return

    Returns:
        List of related concept dictionaries with CUI, preferred term, and relation depth
    """
    _load_artifacts()

    visited = set()
    related = []
    queue = [(cui, 0)]  # (cui, depth)

    while queue and len(related) < max_results:
        current_cui, depth = queue.pop(0)

        if current_cui in visited or depth > max_depth:
            continue

        visited.add(current_cui)

        # Skip the original CUI from results
        if depth > 0:
            pref_term = get_cui_preferred_term(current_cui)
            if not _is_noise_term(pref_term):
                related.append(
                    {
                        "cui": current_cui,
                        "preferred_term": pref_term,
                        "relation_depth": depth,
                    }
                )

        # Add neighbors to queue for next depth level
        if depth < max_depth:
            neighbors = get_cui_neighbors(current_cui)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

    return related[:max_results]


def expand_candidates_with_neighbors(
    candidates: List[Dict],
    expansion_factor: int = 1,
    additive_bump: float = 0.05,
    target_types: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Expand candidate list by adding neighboring concepts.

    Args:
        candidates: List of candidate dictionaries with 'cui' field
        expansion_factor: Number of neighbors to add per candidate

    Returns:
        Expanded list of candidates including neighbors
    """
    _load_artifacts()

    expanded = candidates.copy()
    seen_cuis = {candidate["cui"] for candidate in candidates}

    target_tuis: Optional[Set[str]] = None
    if target_types:
        target_tuis = set()
        for st21_type in target_types:
            if st21_type in ST21PV_TYPES:
                target_tuis.update(ST21PV_TYPES[st21_type])

    for candidate in candidates:
        cui = candidate.get("cui")
        if not cui:
            continue

        neighbors = get_cui_neighbors(cui)

        # Add top neighbors that haven't been seen
        added_count = 0
        for neighbor in neighbors:
            if neighbor not in seen_cuis and added_count < expansion_factor:
                pref_term = get_cui_preferred_term(neighbor)
                if _is_noise_term(pref_term):
                    continue
                if target_tuis is not None:
                    cui_types = get_cui_semantic_types(neighbor)
                    if not cui_types.intersection(target_tuis):
                        continue
                neighbor_candidate = {
                    "cui": neighbor,
                    "preferred_term": pref_term,
                    # Additive bump keeps neighbors low-priority
                    "similarity_score": min(
                        candidate.get("similarity_score", 0.0) + additive_bump, 1.0
                    ),
                    "source": "neighbor_expansion",
                    "parent_cui": cui,
                }
                expanded.append(neighbor_candidate)
                seen_cuis.add(neighbor)
                added_count += 1

    return expanded


def find_common_neighbors(cui_list: List[str]) -> List[Dict[str, any]]:
    """
    Find concepts that are neighbors to multiple CUIs in the list.

    Args:
        cui_list: List of CUIs to find common neighbors for

    Returns:
        List of common neighbor dictionaries with counts
    """
    _load_artifacts()

    if not cui_list:
        return []

    # Count how many CUIs each neighbor appears with
    neighbor_counts = {}

    for cui in cui_list:
        neighbors = get_cui_neighbors(cui)
        for neighbor in neighbors:
            if neighbor not in cui_list:  # Don't include the original CUIs
                neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1

    # Filter for neighbors that appear with multiple CUIs
    common_neighbors = []
    for neighbor, count in neighbor_counts.items():
        if count > 1:  # At least 2 CUIs share this neighbor
            pref_term = get_cui_preferred_term(neighbor)
            common_neighbors.append(
                {
                    "cui": neighbor,
                    "preferred_term": pref_term,
                    "shared_count": count,
                    "coverage": count / len(cui_list),
                }
            )

    # Sort by shared count (descending)
    common_neighbors.sort(key=lambda x: x["shared_count"], reverse=True)
    return common_neighbors


def check_cui_connectivity(
    cui1: str, cui2: str, max_depth: int = 2
) -> Optional[List[str]]:
    """
    Check if two CUIs are connected through UMLS relations within max_depth.

    Args:
        cui1: First CUI
        cui2: Second CUI
        max_depth: Maximum depth to search for connection

    Returns:
        Path of CUIa connecting cui1 to cui2, or None if no connection found
    """
    _load_artifacts()

    if cui1 == cui2:
        return [cui1]

    # BFS to find shortest path
    queue = [([cui1], cui1)]  # (path, current_cui)
    visited = {cui1}

    while queue:
        path, current = queue.pop(0)

        # path has N nodes and N-1 edges; allow paths up to max_depth edges
        if len(path) - 1 > max_depth:
            continue

        neighbors = get_cui_neighbors(current)

        for neighbor in neighbors:
            if neighbor == cui2:
                return path + [neighbor]

            if neighbor not in visited and (len(path) - 1) < max_depth:
                visited.add(neighbor)
                queue.append((path + [neighbor], neighbor))

    return None


def main():
    parser = argparse.ArgumentParser(prog="relation_tools", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_neighbors = sub.add_parser("neighbors")
    p_neighbors.add_argument("cui")

    p_related = sub.add_parser("related")
    p_related.add_argument("cui")
    p_related.add_argument("--depth", type=int, default=1)
    p_related.add_argument("-n", "--max-results", type=int, default=10)

    p_common = sub.add_parser("common")
    p_common.add_argument("cuis", nargs="+")

    p_connect = sub.add_parser("connect")
    p_connect.add_argument("cui1")
    p_connect.add_argument("cui2")
    p_connect.add_argument("--depth", type=int, default=2)

    args = parser.parse_args()

    if args.cmd == "neighbors":
        out = get_cui_neighbors(args.cui)
        print(json.dumps(out, ensure_ascii=False))
        return

    if args.cmd == "related":
        out = find_related_concepts(
            args.cui, max_depth=args.depth, max_results=args.max_results
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "common":
        out = find_common_neighbors(args.cuis)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "connect":
        out = check_cui_connectivity(args.cui1, args.cui2, max_depth=args.depth)
        print(json.dumps(out, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
