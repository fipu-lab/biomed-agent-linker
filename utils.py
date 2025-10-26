#!/usr/bin/env python3
"""
Utility functions for the biomedical entity linking system.
"""

import os
import json
import re
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from logging_setup import get_logger

# Global thread lock for file I/O
_file_lock = threading.Lock()

# Global dictionary to track output files for each approach
_progressive_files = {}


def get_timestamped_filename(base_name: str, extension: str = "json") -> str:
    """Generate a timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def save_progressive_results(
    approach: str, predictions: List[Dict], metrics: Dict = None, batch_num: int = None
):
    """Save progressive results for an approach with thread safety."""
    with _file_lock:
        output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
        os.makedirs(output_dir, exist_ok=True)

        # Use the same file for each approach throughout the evaluation
        if approach not in _progressive_files:
            # Create timestamped filename only once per approach
            filename = get_timestamped_filename(f"{approach}_progressive", "json")
            _progressive_files[approach] = {
                "filename": filename,
                "initial_timestamp": datetime.now().isoformat(),
            }

        filename = _progressive_files[approach]["filename"]
        output_file = os.path.join(output_dir, filename)
        initial_timestamp = _progressive_files[approach]["initial_timestamp"]

        results = {
            "approach": approach,
            "config": None,  # Will be set by caller if needed
            "predictions": predictions,
            "metrics": metrics,
            "initial_timestamp": initial_timestamp,
            "last_updated": datetime.now().isoformat(),
            "progress": {
                "completed_predictions": len(predictions),
                "total_batches_saved": batch_num or 1,
                "last_pmid": predictions[-1].get("pmid") if predictions else None,
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return output_file


def load_existing_predictions(approach: str) -> Tuple[List[Dict], str]:
    """Load existing predictions for resuming evaluation."""
    output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
    if not os.path.exists(output_dir):
        return [], None

    # Check if we already have a tracked file for this approach
    if approach in _progressive_files:
        filename = _progressive_files[approach]["filename"]
        output_file = os.path.join(output_dir, filename)
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                predictions = data.get("predictions", [])
                last_pmid = data.get("progress", {}).get("last_pmid")

                logger = get_logger(__name__)
                logger.info(
                    f"Loaded {len(predictions)} existing predictions from {filename}"
                )
                if last_pmid:
                    logger.info(f"Resume point: PMID {last_pmid}")
                return predictions, last_pmid
            except Exception as e:
                logger = get_logger(__name__)
                logger.warning(f"Failed to load existing predictions: {e}")

    # Find the most recent progressive file for this approach (for backwards compatibility)
    pattern = f"{approach}_progressive"
    files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith(pattern) and f.endswith(".json")
    ]

    if not files:
        return [], None

    # Sort by modification time, get the most recent
    files.sort(
        key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True
    )
    latest_file = os.path.join(output_dir, files[0])

    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        predictions = data.get("predictions", [])
        last_pmid = data.get("progress", {}).get("last_pmid")

        logger = get_logger(__name__)
        logger.info(f"Loaded {len(predictions)} existing predictions from {files[0]}")
        if last_pmid:
            logger.info(f"Resume point: PMID {last_pmid}")

        # Update global tracker for consistency
        _progressive_files[approach] = {
            "filename": files[0],
            "initial_timestamp": data.get(
                "initial_timestamp", datetime.now().isoformat()
            ),
        }

        return predictions, last_pmid
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to load existing predictions: {e}")
        return [], None


def should_skip_entity(
    entity: Dict, last_pmid: str, resume_from_pmid: str = None
) -> bool:
    """Check if entity should be skipped based on resume logic."""
    if not last_pmid or resume_from_pmid is None:
        return False

    current_pmid = entity.get("pmid")
    return current_pmid != last_pmid and current_pmid != resume_from_pmid


def load_prompt(prompt_file: str) -> str:
    """Load prompt template from file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_file)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def format_candidates_for_prompt(candidates: List[Dict]) -> str:
    """Format candidates list for LLM prompt."""
    if not candidates:
        return "No candidates found."

    formatted = []
    for candidate in candidates:
        cui = candidate.get("cui", "")
        pref_term = candidate.get("preferred_term", "Unknown")
        formatted.append(f"- {cui}: {pref_term}")

    return "\n".join(formatted)


def format_template_with_placeholders(
    template: str,
    mention: str,
    candidates: List[Dict],
    context: str = "",
    document: str = "",
) -> str:
    """Format template by replacing placeholders."""
    # Format candidates list
    candidates_text = format_candidates_for_prompt(candidates)

    # Replace placeholders
    formatted_template = template.replace("[[mention]]", mention)
    formatted_template = formatted_template.replace("[[top_10_list]]", candidates_text)
    formatted_template = formatted_template.replace("[[context]]", context)
    formatted_template = formatted_template.replace("[[document]]", document)

    return formatted_template


def safe_parse_json(text: str) -> Dict[str, Any]:
    """Parse model output into JSON robustly, normalizing common issues."""
    if text is None:
        return {
            "predicted_cui": None,
            "reasoning": "empty_response",
        }

    # Strip code fences if present
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    # Prefer the largest JSON object in the text
    m = re.search(r"\{[\s\S]*\}", stripped)
    raw = m.group(0) if m else stripped

    # Normalize quotes and literals
    raw = raw.replace(""", '"').replace(""", '"')
    # Remove trailing commas before object/array end
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    # Python literals -> JSON
    raw = raw.replace("None", "null").replace("True", "true").replace("False", "false")

    try:
        obj = json.loads(raw)
    except Exception as e:
        # Try to fix incomplete JSON by closing incomplete string/object
        fixed_raw = raw

        # If string is incomplete (ends with unmatched quote), try to close it
        if re.search(r'"[^"]*$', fixed_raw):
            # Close incomplete string that's cut off
            fixed_raw = fixed_raw + '"'

        open_braces = fixed_raw.count("{") - fixed_raw.count("}")
        open_brackets = fixed_raw.count("[") - fixed_raw.count("]")

        if open_braces > 0 or open_brackets > 0:
            # Close incomplete strings first
            if fixed_raw.endswith('"'):
                pass  # Already quoted
            elif re.search(r'"[^"]*$', fixed_raw):
                # Incomplete string - close it
                fixed_raw += '"'

            # Add comma if we need to close in the middle of an object/array
            if not fixed_raw.rstrip().endswith((",", "{", "[")):
                fixed_raw += ","

            fixed_raw += "]" * open_brackets
            fixed_raw += "}" * open_braces

            try:
                obj = json.loads(fixed_raw)
            except Exception:
                return {
                    "predicted_cui": None,
                    "reasoning": "parse_error",
                    "original_response": text[:500],  # Show more context
                    "parse_error": str(e),
                }
        else:
            return {
                "predicted_cui": None,
                "reasoning": "parse_error",
                "original_response": text[:500],  # Show more context
                "parse_error": str(e),
            }

    # Check for new format first
    top_1_predicted = obj.get("predicted_cui")
    top_5 = obj.get("alternative_cuis", [])

    if top_1_predicted is not None:
        # New format
        if isinstance(top_1_predicted, str):
            obj["predicted_cui"] = top_1_predicted.strip()
        else:
            obj["predicted_cui"] = (
                None if top_1_predicted is None else str(top_1_predicted).strip()
            )

        # Convert top-5 to alternative_cuis (excluding the top-1)
        if isinstance(top_5, list):
            top_5_clean = [str(c).strip() for c in top_5 if isinstance(c, (str, int))]
            # Remove the top-1 from alternatives if it's in the list
            alternative_cuis = [c for c in top_5_clean if c != obj["predicted_cui"]]
            obj["alternative_cuis"] = alternative_cuis[:4]  # Keep max 4 alternatives
        else:
            obj["alternative_cuis"] = []
    else:
        # Old format - backward compatibility
        cui = obj.get("predicted_cui")
        if isinstance(cui, str):
            obj["predicted_cui"] = cui.strip()
        else:
            obj["predicted_cui"] = None if cui is None else str(cui).strip()

        alts = obj.get("alternative_cuis", [])
        if isinstance(alts, list):
            obj["alternative_cuis"] = [
                str(c).strip() for c in alts if isinstance(c, (str, int))
            ]
        else:
            obj["alternative_cuis"] = []

    return obj


def validate_prediction_in_candidates(
    predicted_cui: str, candidates: List[Dict]
) -> bool:
    """Check if predicted CUI is in candidates list."""
    candidate_cuis = [c.get("cui") for c in candidates]
    return predicted_cui in candidate_cuis


def check_predictions_in_candidates(
    predicted_cui: str, alternative_cuis: List[str], candidates: List[Dict]
) -> Dict[str, Any]:
    """Check if predictions are in candidate list and return detailed validation info."""
    candidate_cuis = [c.get("cui") for c in candidates]

    return {
        "top_1_in_candidates": (
            predicted_cui in candidate_cuis if predicted_cui else False
        ),
        "any_top_5_in_candidates": any(
            cui in candidate_cuis for cui in ([predicted_cui] + alternative_cuis) if cui
        ),
        "all_predictions": [predicted_cui] + alternative_cuis,
        "predictions_in_candidates": [
            cui
            for cui in ([predicted_cui] + alternative_cuis)
            if cui and cui in candidate_cuis
        ],
        "predictions_outside_candidates": [
            cui
            for cui in ([predicted_cui] + alternative_cuis)
            if cui and cui not in candidate_cuis
        ],
    }


def load_sapbert_candidates_from_json(json_file_path: str) -> Dict[str, List[Dict]]:
    """Load SapBERT candidates from JSON file and return as dictionary keyed by mention."""
    logger = get_logger(__name__)

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        candidates_by_mention = {}
        for entry in data.get("candidates", []):
            mention = entry["mention"]
            candidates_by_mention[mention] = entry["sapbert_candidates"]

        logger.info(
            f"Loaded candidates for {len(candidates_by_mention)} mentions from {json_file_path}"
        )
        return candidates_by_mention

    except Exception as e:
        logger.error(f"Failed to load candidates from {json_file_path}: {e}")
        return {}


def get_sapbert_candidates_for_llm(
    mention: str, sapbert_system, max_candidates: int = 10
) -> List[Dict]:
    """Get SapBERT top-k candidates without scores for LLM to choose from."""
    # Get SapBERT candidates with scores
    sapbert_results = sapbert_system.sapbert_knn(mention, k=max_candidates)

    # Extract CUIs without scores
    candidate_cuis = [cui for cui, score in sapbert_results]

    # Get preferred terms for candidates
    from tools import get_cui_preferred_term

    candidates = []
    for cui in candidate_cuis:
        try:
            pref_term = get_cui_preferred_term(cui)
            candidates.append(
                {"cui": cui, "preferred_term": pref_term, "source": "sapbert"}
            )
        except Exception as e:
            # Skip candidates we can't get preferred terms for
            continue

    # Shuffle candidates to remove ranking information
    import random

    random.shuffle(candidates)

    return candidates
