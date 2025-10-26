#!/usr/bin/env python3
"""
Results display and saving functions for the biomedical entity linking system.
"""

import os
import json
from datetime import datetime
from typing import Dict, List
from logging_setup import get_logger
from utils import get_timestamped_filename


def save_comprehensive_results(all_results: Dict, all_predictions: Dict, config: Dict):
    """Save comprehensive results to JSON file with timestamp."""
    output_file = os.path.join(
        os.path.dirname(__file__),
        "evaluation_results",
        get_timestamped_filename("comprehensive_evaluation", "json"),
    )

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = {
        "config": config,
        "approaches": {},
        "timestamp": datetime.now().isoformat(),
    }

    for approach in all_results.keys():
        results["approaches"][approach] = {
            "metrics": all_results[approach],
            "predictions": all_predictions.get(approach, []),
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger = get_logger(__name__)
    logger.info(f" Comprehensive results saved to: {output_file}")
    return output_file


def print_comprehensive_comparison(
    all_results: Dict, test_entity_count: int, config: Dict
):
    """Print comprehensive comparison results for all approaches."""
    logger = get_logger(__name__)

    logger.info("\n" + "=" * 100)
    logger.info(" COMPREHENSIVE ENTITY LINKING EVALUATION RESULTS")
    logger.info("=" * 100)

    # Dataset info
    logger.info(f"Dataset: MedMentions ST21pv (test set)")
    logger.info(f"Test entities: {test_entity_count}")
    # LLM models are dynamic; listed during run

    # Accuracy comparison table (dynamic columns)
    logger.info("\n ACCURACY COMPARISON:")
    logger.info("-" * 100)
    approaches = sorted(all_results.keys())
    header = ["Metric"] + approaches + ["Best"]
    logger.info(" ".join([f"{col:<18}" for col in header]))
    logger.info("-" * 100)

    for k in config["top_k_values"]:
        row_vals = []
        best_name = "N/A"
        best_val = -1
        for name in approaches:
            acc = all_results[name].get(f"top_{k}_accuracy", 0) * 100
            if acc > best_val:
                best_val = acc
                best_name = name
            row_vals.append(acc)
        logger.info(
            f"{'Top-' + str(k):<18} "
            + " ".join([f"{v:<18.2f}" for v in row_vals])
            + f" {best_name:<18}"
        )

    # Performance comparison (dynamic columns)
    logger.info("\n PERFORMANCE COMPARISON:")
    logger.info("-" * 100)
    logger.info(
        " ".join([f"{col:<18}" for col in ["Metric"] + approaches + ["Winner"]])
    )
    logger.info("-" * 100)

    # Avg time
    times = {name: all_results[name].get("avg_time_ms", 0) for name in approaches}
    fastest = min(times, key=times.get) if times else "N/A"
    logger.info(
        f"{'Avg time (ms)':<18} "
        + " ".join([f"{times[a]:<18.2f}" for a in approaches])
        + f" {fastest:<18}"
    )

    # QPS
    qps_map = {name: all_results[name].get("queries_per_sec", 0) for name in approaches}
    fastest_qps = max(qps_map, key=qps_map.get) if qps_map else "N/A"
    logger.info(
        f"{'Queries/sec':<18} "
        + " ".join([f"{qps_map[a]:<18.1f}" for a in approaches])
        + f" {fastest_qps:<18}"
    )

    # Summary
    logger.info("\n SUMMARY ANALYSIS:")
    logger.info("-" * 60)
    top5_map = {
        name: all_results[name].get("top_5_accuracy", 0) * 100 for name in approaches
    }
    if top5_map:
        top5_best = max(top5_map, key=top5_map.get)
        logger.info(f"Best Top-5 Accuracy: {top5_best} ({top5_map[top5_best]:.2f}%)")

    logger.info("=" * 100)
