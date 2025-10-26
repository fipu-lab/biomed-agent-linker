#!/usr/bin/env python3
"""Download MedMentions ST21pv dataset from HuggingFace."""

from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Downloading MedMentions ST21pv dataset from HuggingFace...")
dataset = load_dataset("zameji/medmentions-st21pv")
logger.info(f"Train: {dataset['train']}")
logger.info(f"Validation: {dataset['validation']}")
logger.info(f"Test: {dataset['test']}")

logger.info("Saving dataset to disk...")
dataset.save_to_disk("medmentions_st21pv")
logger.info("Dataset saved successfully!")
