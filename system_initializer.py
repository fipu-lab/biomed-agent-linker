#!/usr/bin/env python3
"""
System initialization functions for the biomedical entity linking system.
"""

from typing import Dict
from logging_setup import get_logger
from build_quickumls import QuickUMLSCandidateGenerator
from candidate_gen import get_sapbert_system, get_biobert_system


def initialize_systems(config: Dict) -> Dict:
    """Initialize all baseline systems."""
    logger = get_logger(__name__)
    systems = {}

    # Initialize QuickUMLS (Baseline 1)
    if "quickumls" in config["approaches_to_run"]:
        logger.info("Initializing QuickUMLS...")
        try:
            quickumls_generator = QuickUMLSCandidateGenerator(
                index_path=config["quickumls_index_path"],
                threshold=config["threshold"],
                similarity_name=config["similarity_name"],
                window=config["window"],
                min_match_length=config["min_match_length"],
            )
            systems["quickumls"] = quickumls_generator
            logger.info("QuickUMLS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QuickUMLS: {e}")
            config["approaches_to_run"].remove("quickumls")

    # Initialize SapBERT (Baseline 2)
    if "sapbert" in config["approaches_to_run"]:
        logger.info("Initializing SapBERT...")
        try:
            sapbert_system = get_sapbert_system()
            if sapbert_system and sapbert_system.is_initialized:
                systems["sapbert"] = sapbert_system
                logger.info("SapBERT initialized successfully")
                stats = sapbert_system.get_stats()
                logger.info(f"Pool size: {stats['pool_size']} terms")
            else:
                raise ValueError("SapBERT system not properly initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SapBERT: {e}")
            config["approaches_to_run"].remove("sapbert")

    # Initialize BioBERT (Baseline 3)
    if "biobert" in config["approaches_to_run"]:
        logger.info("Initializing BioBERT...")
        try:
            biobert_system = get_biobert_system()
            if biobert_system and biobert_system.is_initialized:
                systems["biobert"] = biobert_system
                logger.info("BioBERT initialized successfully")
                stats = biobert_system.get_stats()
                logger.info(f"Pool size: {stats['pool_size']} terms")
            else:
                raise ValueError("BioBERT system not properly initialized")
        except Exception as e:
            logger.error(f"Failed to initialize BioBERT: {e}")
            config["approaches_to_run"].remove("biobert")

    # Initialize SciSpacy UMLS Linker (Baseline 4)
    if "scispacy" in config["approaches_to_run"]:
        logger.info("Initializing SciSpacy UMLS Linker...")
        try:
            # Note: scispacy_linker.py module not implemented
            logger.warning(
                "SciSpacy initialization skipped - scispacy_linker module not found"
            )
            config["approaches_to_run"].remove("scispacy")
        except Exception as e:
            logger.error(f"Failed to initialize SciSpacy: {e}")
            config["approaches_to_run"].remove("scispacy")

    return systems
