#!/usr/bin/env python3
"""
Main script for comprehensive evaluation of biomedical entity linking approaches.

This script evaluates:
1. Baseline systems (QuickUMLS, SapBERT, BioBERT, SciSpacy)
2. LLM-based approaches with various configurations
"""

import sys
import traceback
from logging_setup import setup_logging, get_logger
from llm import models
from data_loader import load_test_data, generate_sapbert_candidates_json
from utils import load_sapbert_candidates_from_json
from baseline_evaluators import evaluate_baseline
from llm_evaluators import (
    evaluate_llm_with_tools_with_document,
    evaluate_llm_with_tools_no_document,
    evaluate_llm_no_tools_with_document,
    evaluate_llm_no_tools_no_document,
)
from results_handler import print_comprehensive_comparison, save_comprehensive_results
from system_initializer import initialize_systems
from candidate_gen import get_sapbert_system

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Dataset settings
    "test_sample_size": 500,
    "min_entity_length": 3,  # Minimum entity mention length
    # Baseline settings (QuickUMLS, SapBERT, BioBERT)
    "quickumls_index_path": "/Users/lukablaskovic/Github/PhD/db/UMLS",
    "threshold": 0.8,
    "similarity_name": "jaccard",
    "window": 5,
    "min_match_length": 3,
    "topN": 10,  # Number of candidates to retrieve
    "sapbert_candidates_for_llm": 10,  # Number of SapBERT candidates for LLM to choose from
    "sapbert_model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "sapbert_cache_dir": "candidate_embeddings_cache",
    "biobert_model_name": "dmis-lab/biobert-base-cased-v1.1",
    "biobert_cache_dir": "candidate_embeddings_cache",
    # LLM settings
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": 2500,
    "max_tool_candidates": 5,
    "max_concurrent_requests": 2,
    # Evaluation settings
    "top_k_values": [1, 3, 5],
    "normalize_text": True,
    "approaches_to_run": [
        "llm_with_tools_no_document",  # LLM with tools, no document
        "llm_with_tools_with_document",  # LLM with tools, with document
    ],
    "verbose": True,
    "save_predictions": True,
    "progressive_save": True,
    "save_batch_size": 5,  # Save every 5 predictions
    "resume_from_pmid": None,  # PMID to resume from (None = start from beginning)
}

# =============================================================================
# MAIN EVALUATION
# =============================================================================


def main():
    """Main evaluation function."""
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting comprehensive evaluation of all entity linking approaches...")
    logger.info(f"Approaches: {', '.join(CONFIG['approaches_to_run'])}")

    try:
        # Load test data
        test_entities = load_test_data(
            CONFIG["test_sample_size"], CONFIG["min_entity_length"]
        )

        # Initialize SapBERT system for candidate generation
        logger.info("Initializing SapBERT for candidate generation...")
        sapbert_system = get_sapbert_system()
        if not sapbert_system or not sapbert_system.is_initialized:
            raise ValueError(
                "Failed to initialize SapBERT system for candidate generation"
            )
        logger.info("SapBERT initialized successfully for candidate generation")

        # Generate SapBERT candidates for all test entities
        logger.info(" Generating SapBERT top 10 candidates for all test entities...")
        candidates_file, candidates_data = generate_sapbert_candidates_json(
            test_entities, sapbert_system
        )

        # Load candidates into memory for LLM evaluation
        candidates_by_mention = load_sapbert_candidates_from_json(candidates_file)

        # Initialize baseline systems if needed
        systems = initialize_systems(CONFIG)

        # Run evaluations
        all_results = {}
        all_predictions = {}

        # Evaluate baseline systems
        baseline_approaches = ["quickumls", "sapbert", "biobert", "scispacy"]
        for baseline_name in baseline_approaches:
            if (
                baseline_name in CONFIG["approaches_to_run"]
                and baseline_name in systems
            ):
                logger.info(f"\nEvaluating {baseline_name.upper()}...")
                metrics, predictions = evaluate_baseline(
                    test_entities, baseline_name, systems[baseline_name], CONFIG
                )
                all_results[baseline_name] = metrics
                all_predictions[baseline_name] = predictions
                logger.info(f"{baseline_name.upper()} evaluation completed")

        # Evaluate LLMs for all registered models with 4 different configurations
        # Iterate through llm.models registry
        for model_key, cfg in models.items():
            provider_model_name = getattr(cfg, "model", str(cfg))
            supports_function_calling = getattr(
                cfg.meta, "supports_function_calling", True
            )

            # 1. LLM no tools, no document
            if "llm_no_tools_no_document" in CONFIG["approaches_to_run"]:
                logger.info(
                    f"\nEvaluating LLM NO TOOLS, NO DOCUMENT for {model_key} -> {provider_model_name}..."
                )
                metrics, predictions = evaluate_llm_no_tools_no_document(
                    test_entities,
                    model_key,
                    provider_model_name,
                    candidates_by_mention,
                    CONFIG,
                )
                approach_name = f"llm_no_tools_no_document__{model_key}"
                all_results[approach_name] = metrics
                all_predictions[approach_name] = predictions
                logger.info(f"LLM NO TOOLS, NO DOCUMENT completed for {model_key}")

            # 2. LLM with tools, no document
            if (
                "llm_with_tools_no_document" in CONFIG["approaches_to_run"]
                and supports_function_calling
            ):
                logger.info(
                    f"\nEvaluating LLM WITH TOOLS, NO DOCUMENT for {model_key} -> {provider_model_name}..."
                )
                metrics, predictions = evaluate_llm_with_tools_no_document(
                    test_entities,
                    model_key,
                    provider_model_name,
                    candidates_by_mention,
                    CONFIG,
                )
                approach_name = f"llm_with_tools_no_document__{model_key}"
                all_results[approach_name] = metrics
                all_predictions[approach_name] = predictions
                logger.info(f"LLM WITH TOOLS, NO DOCUMENT completed for {model_key}")
            elif (
                "llm_with_tools_no_document" in CONFIG["approaches_to_run"]
                and not supports_function_calling
            ):
                logger.info(
                    f"Skipping LLM WITH TOOLS, NO DOCUMENT for {model_key} -> {provider_model_name} (function calling not supported)"
                )

            # 3. LLM no tools, with document
            if "llm_no_tools_with_document" in CONFIG["approaches_to_run"]:
                logger.info(
                    f"\nEvaluating LLM NO TOOLS, WITH DOCUMENT for {model_key} -> {provider_model_name}..."
                )
                metrics, predictions = evaluate_llm_no_tools_with_document(
                    test_entities,
                    model_key,
                    provider_model_name,
                    candidates_by_mention,
                    CONFIG,
                )
                approach_name = f"llm_no_tools_with_document__{model_key}"
                all_results[approach_name] = metrics
                all_predictions[approach_name] = predictions
                logger.info(f"LLM NO TOOLS, WITH DOCUMENT completed for {model_key}")

            # 4. LLM with tools, with document
            if (
                "llm_with_tools_with_document" in CONFIG["approaches_to_run"]
                and supports_function_calling
            ):
                logger.info(
                    f"\nEvaluating LLM WITH TOOLS, WITH DOCUMENT for {model_key} -> {provider_model_name}..."
                )
                metrics, predictions = evaluate_llm_with_tools_with_document(
                    test_entities,
                    model_key,
                    provider_model_name,
                    candidates_by_mention,
                    CONFIG,
                )
                approach_name = f"llm_with_tools_with_document__{model_key}"
                all_results[approach_name] = metrics
                all_predictions[approach_name] = predictions
                logger.info(f"LLM WITH TOOLS, WITH DOCUMENT completed for {model_key}")
            elif (
                "llm_with_tools_with_document" in CONFIG["approaches_to_run"]
                and not supports_function_calling
            ):
                logger.info(
                    f"Skipping LLM WITH TOOLS, WITH DOCUMENT for {model_key} -> {provider_model_name} (function calling not supported)"
                )

        # Print comprehensive comparison
        print_comprehensive_comparison(all_results, len(test_entities), CONFIG)

        # Save results
        if CONFIG["save_predictions"]:
            save_comprehensive_results(all_results, all_predictions, CONFIG)

        logger.info("Comprehensive evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
