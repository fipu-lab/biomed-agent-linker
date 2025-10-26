import json
import os
import argparse
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import re
from rapidfuzz import process, fuzz

_cui2strings = None
_cui2pref = None
_string2cuis = None
_all_strings_lower = None


def _load_artifacts():
    global _cui2strings, _cui2pref, _string2cuis, _all_strings_lower

    if _cui2strings is None:
        artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")

        # Load cui2strings
        cui2strings_path = os.path.join(artifacts_dir, "cui2strings.json")
        with open(cui2strings_path, "r", encoding="utf-8") as f:
            _cui2strings = json.load(f)

        # Load cui2pref
        cui2pref_path = os.path.join(artifacts_dir, "cui2pref.json")
        with open(cui2pref_path, "r", encoding="utf-8") as f:
            _cui2pref = json.load(f)

        # Create reverse index: string -> list of CUIs
        _string2cuis = {}
        for cui, strings in _cui2strings.items():
            for string in strings:
                string_lower = string.lower().strip()
                if string_lower not in _string2cuis:
                    _string2cuis[string_lower] = []
                _string2cuis[string_lower].append(cui)

        # Cache list of all strings for rapidfuzz
        _all_strings_lower = list(_string2cuis.keys())


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_short_form(text: str) -> bool:
    alnum = re.sub(r"[^a-z0-9]", "", text)
    return len(alnum) <= 3


def get_cui_strings(cui: str) -> List[str]:
    _load_artifacts()
    return _cui2strings.get(cui, [])


def get_cui_preferred_term(cui: str) -> Optional[str]:
    _load_artifacts()
    return _cui2pref.get(cui)


def search_by_string(
    query: str, max_results: int = 25, fuzzy: bool = True
) -> List[Tuple[str, str, float]]:

    _load_artifacts()

    query_norm = _normalize_text(query)
    short_mode = _is_short_form(query_norm)

    # Use cached core search
    results = list(_search_core(query_norm, max_results, fuzzy, short_mode))
    return results


@lru_cache(maxsize=10000)
def _search_core(
    query_norm: str, max_results: int, fuzzy: bool, short_mode: bool
) -> tuple:
    results: List[Tuple[str, str, float]] = []

    # Exact matches
    if query_norm in _string2cuis:
        for cui in _string2cuis[query_norm]:
            results.append((cui, query_norm, 1.0))

    # Short-form handling: prefer strict partial matches, slightly relaxed cutoff
    if short_mode and len(results) < max_results:
        candidates = process.extract(
            query_norm,
            _all_strings_lower,
            scorer=fuzz.partial_ratio,
            limit=max(max_results * 20, 50),
            score_cutoff=90,
        )
        for string_lower, score, _ in candidates:
            if string_lower == query_norm:
                continue
            for cui in _string2cuis[string_lower]:
                results.append((cui, string_lower, score / 100.0))

    # Fuzzy matching for normal queries
    if not short_mode and fuzzy and len(results) < max_results:
        scorer = fuzz.token_set_ratio if len(query_norm) > 6 else fuzz.ratio
        candidates = process.extract(
            query_norm,
            _all_strings_lower,
            scorer=scorer,
            limit=max_results * 20,
            score_cutoff=70,
        )
        for string_lower, score, _ in candidates:
            if string_lower == query_norm:
                continue
            for cui in _string2cuis[string_lower]:
                results.append((cui, string_lower, score / 100.0))

    # Deduplicate by CUI, keeping the highest scoring matched string per CUI
    best_by_cui = {}
    for cui, matched_string, score in results:
        if (cui not in best_by_cui) or (score > best_by_cui[cui][0]):
            best_by_cui[cui] = (score, matched_string)

    deduped = [(cui, ms, sc) for cui, (sc, ms) in best_by_cui.items()]
    deduped.sort(key=lambda x: x[2], reverse=True)
    return tuple(deduped[:max_results])


def find_string_candidates(
    mention: str, max_candidates: int = 25
) -> List[Dict[str, any]]:

    results = search_by_string(mention, max_results=max_candidates * 2, fuzzy=True)

    def _is_noise_term(term: Optional[str]) -> bool:
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
        loinc_tokens = [":Find:", ":PrThr:", ":Qn", ":Ord", ":Nom", ":Temp:", "Pt:"]
        if any(tok in t for tok in loinc_tokens):
            return True
        return False

    candidates = []
    seen_cuis = set()
    for cui, matched_string, score in results:
        if cui in seen_cuis:
            continue
        pref_term = get_cui_preferred_term(cui)
        if _is_noise_term(pref_term):
            continue
        candidates.append(
            {
                "cui": cui,
                "preferred_term": pref_term,
                "similarity_score": min(max(score, 0.0), 1.0),
            }
        )
        seen_cuis.add(cui)

    # Keep only top N after filtering
    candidates.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    return candidates[:max_candidates]


def get_candidates(
    mention: str, entity_types: Optional[List[str]] = None, max_candidates: int = 5
) -> List[Dict[str, any]]:

    # Lazy import to avoid circular dependency
    from .semantic_tools import (
        get_cui_semantic_types,
        filter_by_semantic_type,
        get_st21_type_mapping,
    )
    from .relation_tools import expand_candidates_with_neighbors

    # Get initial candidates using string matching
    candidates = find_string_candidates(mention, max_candidates=max_candidates * 3)

    # Enrich with semantic type information
    for candidate in candidates:
        cui = candidate["cui"]
        semantic_types = list(get_cui_semantic_types(cui))
        candidate["semantic_types"] = semantic_types

        # Add ST21pv type mapping if requested
        if entity_types:
            candidate["st21_types"] = get_st21_type_mapping(cui)

    # Filter by semantic type if provided
    if entity_types and entity_types[0] != "UNKNOWN":
        filtered = filter_by_semantic_type(candidates, target_types=entity_types)

        # If filtering is too restrictive, expand with neighbors
        if len(filtered) < 3 and candidates:
            # Use top candidates for neighbor expansion with type filtering
            expanded = expand_candidates_with_neighbors(
                candidates[:3], expansion_factor=1, target_types=entity_types
            )
            # Combine filtered and expanded, deduplicate
            cui_seen = {c["cui"] for c in filtered}
            for c in expanded:
                if c["cui"] not in cui_seen:
                    filtered.append(c)
                    cui_seen.add(c["cui"])
            candidates = filtered
        else:
            candidates = filtered

    # Ensure scores are normalized and sort
    for candidate in candidates:
        candidate["similarity_score"] = min(
            max(candidate.get("similarity_score", 0.0), 0.0), 1.0
        )

    candidates.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    return candidates[:max_candidates]


def main():
    parser = argparse.ArgumentParser(prog="lexical_tools", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("-n", "--max-results", type=int, default=10)
    p_search.add_argument("--no-fuzzy", action="store_true")

    p_strings = sub.add_parser("strings")
    p_strings.add_argument("cui")

    p_pref = sub.add_parser("pref")
    p_pref.add_argument("cui")

    p_candidates = sub.add_parser("candidates")
    p_candidates.add_argument("mention")
    p_candidates.add_argument("-n", "--max-candidates", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "search":
        out = search_by_string(
            args.query, max_results=args.max_results, fuzzy=not args.no_fuzzy
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "strings":
        out = get_cui_strings(args.cui)
        print(json.dumps(out, ensure_ascii=False))
        return

    if args.cmd == "pref":
        out = get_cui_preferred_term(args.cui)
        print(json.dumps(out, ensure_ascii=False))
        return

    if args.cmd == "candidates":
        out = find_string_candidates(args.mention, max_candidates=args.max_candidates)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()

    """
    Lexical:
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/lexical_tools.py search "heart attack" -n 10
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/lexical_tools.py strings C0011849
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/lexical_tools.py pref C0011849
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/lexical_tools.py candidates "myocardial infarction" -n 5

Semantic:
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/semantic_tools.py types C0011849
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/semantic_tools.py st21map C0011849
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/semantic_tools.py filter -t DISO CHEM -c /path/to/candidates.json
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/semantic_tools.py boost -t DISO -c /path/to/candidates.json --factor 1.4

Relations:

python3 /Users/lukablaskovic/Github/PhD/OGR/tools/relation_tools.py neighbors C0011849
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/relation_tools.py related C0011849 --depth 2 -n 10
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/relation_tools.py common C0011849 C0027051 C0032285
python3 /Users/lukablaskovic/Github/PhD/OGR/tools/relation_tools.py connect C0011849 C0027051 --depth 3

    """
