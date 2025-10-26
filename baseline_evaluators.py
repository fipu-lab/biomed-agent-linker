import time
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
from logging_setup import get_logger
from utils import (
    load_existing_predictions,
    should_skip_entity,
    save_progressive_results,
)


def evaluate_baseline(
    test_entities: List[Dict], baseline_name: str, baseline_generator, config: Dict
) -> Tuple[Dict, List[Dict]]:
    """Evaluate a baseline system on test entities."""
    logger = get_logger(__name__)

    # Load existing predictions if resume is enabled
    if config.get("progressive_save"):
        existing_predictions, last_pmid = load_existing_predictions(baseline_name)
        predictions = existing_predictions.copy()
    else:
        predictions = []
        last_pmid = None

    correct_at_k = defaultdict(int)
    total_time = 0
    batch_count = 0

    logger.info(f"Evaluating {baseline_name} on {len(test_entities)} entities...")
    if last_pmid:
        logger.info(f"Resuming from PMID: {last_pmid}")

    # Progress bar for baseline evaluation
    pbar = tqdm(
        enumerate(test_entities),
        total=len(test_entities),
        desc=f"{baseline_name.upper()}",
        unit="entities",
    )

    for i, entity in pbar:
        mention = entity["mention"]
        gold_cui = entity["gold_cui"]

        # Time the prediction
        start_time = time.time()

        # Get candidates based on baseline type
        if baseline_name == "quickumls":
            candidates = baseline_generator.quickumls_candidates(
                mention, topN=config["topN"], normalize=config["normalize_text"]
            )
        elif baseline_name == "sapbert":
            candidates = baseline_generator.sapbert_knn(mention, k=config["topN"])
        elif baseline_name == "biobert":
            candidates = baseline_generator.biobert_knn(mention, k=config["topN"])
        elif baseline_name == "scispacy":
            candidates = baseline_generator.scispacy_knn(mention, k=config["topN"])
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        end_time = time.time()

        query_time = (end_time - start_time) * 1000  # Convert to ms
        total_time += query_time

        # Extract predicted CUIs
        predicted_cuis = [cui for cui, score in candidates]

        # Check accuracy at different k values
        for k in config["top_k_values"]:
            if gold_cui in predicted_cuis[:k]:
                correct_at_k[k] += 1

        # Check if we should skip this entity for resume
        if should_skip_entity(entity, last_pmid, config.get("resume_from_pmid")):
            continue

        # Store prediction
        prediction = {
            "mention": mention,
            "gold_cui": gold_cui,
            "predicted_cuis": predicted_cuis[: max(config["top_k_values"])],
            "candidates": candidates[: max(config["top_k_values"])],
            "query_time_ms": query_time,
            "types": entity["types"],
            "pmid": entity.get("pmid"),
        }
        predictions.append(prediction)

        # Update progress bar
        top_predicted = predicted_cuis[0] if predicted_cuis else "None"
        is_correct = "Y" if gold_cui in predicted_cuis[:1] else "N"
        pbar.set_postfix(
            {
                "Acc@1": f"{correct_at_k[1]/(len(predictions) or 1)*100:.1f}%",
                "Last": f"{mention[:15]}..." if len(mention) > 15 else mention,
                "Correct": is_correct,
            }
        )

        # Progressive saving
        if (
            config.get("progressive_save")
            and len(predictions) % config["save_batch_size"] == 0
        ):
            batch_count += 1
            save_progressive_results(baseline_name, predictions, batch_num=batch_count)

        # Detailed logging (reduced frequency)
        if config["verbose"] and i % 100 == 0:
            doc_info = f"PMID: {entity.get('pmid', 'unknown')}"
            logger.info(f"  [{i+1}/{len(test_entities)}] {doc_info}")
            logger.info(f"    Mention: '{mention}'")
            logger.info(f"    Gold CUI: {gold_cui}")
            logger.info(f"    Predicted: {top_predicted} {is_correct}")
            logger.info(f"    Text: {entity.get('context', 'No context')[:100]}...")

    pbar.close()

    # Calculate metrics
    total_entities = len(test_entities)
    metrics = {}

    for k in config["top_k_values"]:
        accuracy = correct_at_k[k] / total_entities if total_entities > 0 else 0
        metrics[f"top_{k}_accuracy"] = accuracy

    # Timing metrics
    avg_time_ms = total_time / total_entities if total_entities > 0 else 0
    queries_per_sec = 1000 / avg_time_ms if avg_time_ms > 0 else 0
    metrics["avg_time_ms"] = avg_time_ms
    metrics["queries_per_sec"] = queries_per_sec
    metrics["total_time_sec"] = total_time / 1000

    return metrics, predictions
