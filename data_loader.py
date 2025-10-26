import os
import json
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from logging_setup import get_logger
from tools import get_cui_preferred_term


def load_test_data(sample_size=100, min_entity_length=3):
    logger = get_logger(__name__)

    logger.info("Loading MedMentions ST21pv dataset from artifacts...")

    artifacts_path = os.path.join(
        os.path.dirname(__file__), "artifacts", "medmentions_docs.json"
    )

    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")

    with open(artifacts_path, "r", encoding="utf-8") as f:
        docs_data = json.load(f)

    if not docs_data:
        raise ValueError("No MedMentions documents loaded from artifacts!")

    test_split_idx = int(0.8 * len(docs_data))
    test_docs = docs_data[test_split_idx:]

    logger.info(f"Total documents: {len(docs_data)}")
    logger.info(f"Test documents: {len(test_docs)}")

    test_entities = []
    for doc in test_docs:
        for span in doc["spans"]:
            start, end, mention, cui, types = span
            if cui and len(mention.strip()) >= min_entity_length:
                context_start = max(0, start - 100)
                context_end = min(len(doc["text"]), end + 100)
                context = doc["text"][context_start:context_end]

                test_entities.append(
                    {
                        "pmid": doc["pmid"],
                        "mention": mention.strip(),
                        "gold_cui": cui,
                        "types": types,
                        "start": start,
                        "end": end,
                        "context": context,
                        "document": doc["text"],
                    }
                )

    if sample_size and sample_size < len(test_entities):
        import random

        random.seed(42)
        test_entities = random.sample(test_entities, sample_size)

    logger.info(f"Test entities: {len(test_entities)}")
    return test_entities


def generate_sapbert_candidates_json(
    test_entities, sapbert_system, output_filename="sapbert_top10_candidates.json"
):
    """Generate SapBERT top 10 candidates for all test entities and save to JSON file."""
    logger = get_logger(__name__)

    logger.info(
        f"Generating SapBERT top 10 candidates for {len(test_entities)} entities..."
    )

    candidates_data = []

    # Progress bar for candidate generation
    pbar = tqdm(
        enumerate(test_entities),
        total=len(test_entities),
        desc="SapBERT Candidates",
        unit="entities",
    )

    for i, entity in pbar:
        mention = entity["mention"]
        gold_cui = entity["gold_cui"]

        try:
            sapbert_results = sapbert_system.sapbert_knn(mention, k=10)

            formatted_candidates = []
            for cui, score in sapbert_results:
                try:
                    pref_term = get_cui_preferred_term(cui)
                    formatted_candidates.append(
                        {"cui": cui, "preferred_term": pref_term, "score": float(score)}
                    )
                except Exception as e:
                    logger.warning(f"Failed to get preferred term for CUI {cui}: {e}")
                    continue

            # Store candidate data
            candidate_entry = {
                "pmid": entity.get("pmid"),
                "mention": mention,
                "gold_cui": gold_cui,
                "types": entity["types"],
                "start": entity["start"],
                "end": entity["end"],
                "context": entity["context"],
                "document": entity.get("document", ""),
                "sapbert_candidates": formatted_candidates,
            }
            candidates_data.append(candidate_entry)

            # Update progress bar
            pbar.set_postfix(
                {
                    "Last": f"{mention[:15]}..." if len(mention) > 15 else mention,
                    "Candidates": len(formatted_candidates),
                }
            )

        except Exception as e:
            logger.error(f"Failed to get SapBERT candidates for '{mention}': {e}")
            # Add entry with empty candidates
            candidate_entry = {
                "pmid": entity.get("pmid"),
                "mention": mention,
                "gold_cui": gold_cui,
                "types": entity["types"],
                "start": entity["start"],
                "end": entity["end"],
                "context": entity["context"],
                "document": entity.get("document", ""),
                "sapbert_candidates": [],
                "error": str(e),
            }
            candidates_data.append(candidate_entry)

    pbar.close()

    output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, output_filename)

    final_data = {
        "metadata": {
            "total_entities": len(test_entities),
            "generated_at": datetime.now().isoformat(),
            "sapbert_model": (
                sapbert_system.model_name
                if hasattr(sapbert_system, "model_name")
                else "unknown"
            ),
            "candidates_per_entity": 10,
        },
        "candidates": candidates_data,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    logger.info(f"SapBERT candidates saved to: {output_file}")
    return output_file, candidates_data
