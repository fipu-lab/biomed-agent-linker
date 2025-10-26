import os
import gzip
from dataclasses import dataclass
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MedMentionsDocument:

    pmid: str
    text: str
    spans: List[
        Tuple[int, int, str, str, List[str]]
    ]  # (start, end, mention, gold_cui, types)


def process_pubtator_chunk(chunk_data):
    chunk_start, chunk_size, file_path = chunk_data
    docs = []
    cur = {"pmid": None, "text": "", "spans": []}

    with open(file_path, encoding="utf-8") as f:
        for _ in range(chunk_start):
            f.readline()

        lines_processed = 0
        for line in f:
            if lines_processed >= chunk_size:
                break
            lines_processed += 1

            line = line.rstrip("\n")
            if not line:
                if cur["pmid"]:
                    docs.append(
                        MedMentionsDocument(
                            pmid=cur["pmid"], text=cur["text"], spans=cur["spans"]
                        )
                    )
                    cur = {"pmid": None, "text": "", "spans": []}
                continue

            if "|t|" in line or "|a|" in line:
                pmid, tag, val = line.split("|", 2)
                if cur["pmid"] and cur["pmid"] != pmid:
                    docs.append(
                        MedMentionsDocument(
                            pmid=cur["pmid"], text=cur["text"], spans=cur["spans"]
                        )
                    )
                    cur = {"pmid": pmid, "text": "", "spans": []}
                cur["pmid"] = pmid
                cur["text"] += (" " if cur["text"] else "") + val
            else:
                # Entity line: PMID<TAB>start<TAB>end<TAB>text<TAB>TYPE<TAB>CUI
                try:
                    parts = line.split("\t")
                    if len(parts) >= 6:
                        pmid, s, e, m, types, cui = parts[:6]
                        # Extract actual CUI from UMLS:C format
                        if cui.startswith("UMLS:"):
                            cui = cui[5:]  # Remove 'UMLS:' prefix
                        cur["spans"].append((int(s), int(e), m, cui, types.split("|")))
                except (ValueError, IndexError):
                    # Skip malformed lines
                    continue

    # Don't forget the last document
    if cur["pmid"]:
        docs.append(
            MedMentionsDocument(pmid=cur["pmid"], text=cur["text"], spans=cur["spans"])
        )

    return docs


def load_pubtator_st21(
    file_path: str, use_multiprocessing: bool = True
) -> List[MedMentionsDocument]:

    logger.info(f"Loading MedMentions ST21pv from: {file_path}")

    # Determine if file is gzipped
    is_gzipped = file_path.endswith(".gz")
    open_func = gzip.open if is_gzipped else open
    mode = "rt" if is_gzipped else "r"

    # Disable multiprocessing for gzipped files (complexity)
    if is_gzipped:
        use_multiprocessing = False

    if not use_multiprocessing:
        # Single-threaded version (professor's original logic)
        docs = []
        cur = {"pmid": None, "text": "", "spans": []}

        with open_func(file_path, mode=mode, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    if cur["pmid"]:
                        docs.append(
                            MedMentionsDocument(
                                pmid=cur["pmid"], text=cur["text"], spans=cur["spans"]
                            )
                        )
                        cur = {"pmid": None, "text": "", "spans": []}
                    continue

                if "|t|" in line or "|a|" in line:
                    pmid, tag, val = line.split("|", 2)
                    if cur["pmid"] and cur["pmid"] != pmid:
                        docs.append(
                            MedMentionsDocument(
                                pmid=cur["pmid"], text=cur["text"], spans=cur["spans"]
                            )
                        )
                        cur = {"pmid": pmid, "text": "", "spans": []}
                    cur["pmid"] = pmid
                    cur["text"] += (" " if cur["text"] else "") + val
                else:
                    # Entity line
                    try:
                        parts = line.split("\t")
                        if len(parts) >= 6:
                            pmid, s, e, m, types, cui = parts[:6]
                            # Extract actual CUI from UMLS:C format
                            if cui.startswith("UMLS:"):
                                cui = cui[5:]  # Remove 'UMLS:' prefix
                            cur["spans"].append(
                                (int(s), int(e), m, cui, types.split("|"))
                            )
                    except (ValueError, IndexError):
                        continue

        if cur["pmid"]:
            docs.append(
                MedMentionsDocument(
                    pmid=cur["pmid"], text=cur["text"], spans=cur["spans"]
                )
            )

        logger.info(f"Loaded {len(docs)} documents (single-threaded)")
        return docs

    # Multi-threaded version
    # Count total lines for chunking
    with open(file_path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Determine chunk size for multiprocessing
    num_cores = cpu_count()
    chunk_size = max(1000, total_lines // (num_cores * 2))
    chunks = []

    for start in range(0, total_lines, chunk_size):
        size = min(chunk_size, total_lines - start)
        chunks.append((start, size, file_path))

    logger.info(
        f"Processing {total_lines} lines using {num_cores} cores in {len(chunks)} chunks..."
    )

    # Process chunks in parallel
    all_docs = []
    with Pool(num_cores) as pool:
        chunk_results = pool.map(process_pubtator_chunk, chunks)
        for result in chunk_results:
            all_docs.extend(result)

    logger.info(f"Loaded {len(all_docs)} documents (multi-threaded)")
    return all_docs


def load_medmentions_hf(
    dataset_path: str, split: str = "train"
) -> List[MedMentionsDocument]:

    try
        from datasets import load_from_disk
    except ImportError:
        raise ImportError(
            "datasets library required for HuggingFace format. Install with: pip install datasets"
        )

    logger.info(f"Loading MedMentions ST21pv from HuggingFace dataset: {dataset_path}")

    # Load the dataset
    dataset = load_from_disk(dataset_path)
    data = dataset[split]

    docs = []
    for idx, example in enumerate(data):
        # Convert HF format to our document format
        spans = []
        text = example.get("text", "")

        if "annotations" in example:
            for ann_idx, annotation in enumerate(example["annotations"]):
                # Extract span information from simplified format
                start = annotation.get("start", 0)
                end = annotation.get("end", 0)

                # Extract mention text from positions
                mention = (
                    text[start:end] if start < len(text) and end <= len(text) else ""
                )

                # Missing data - use placeholders
                cui = f"UNKNOWN_CUI_{ann_idx}"  # No CUI in this dataset
                types = ["UNKNOWN_TYPE"]  # No types in this dataset

                spans.append((start, end, mention, cui, types))

        # Use document index as PMID since real PMIDs are not available
        pmid = f"DOC_{split}_{idx}"

        docs.append(MedMentionsDocument(pmid=pmid, text=text, spans=spans))

    logger.info(f"Loaded {len(docs)} documents from HuggingFace dataset ({split} split)")
    logger.info(f"  Total spans: {sum(len(doc.spans) for doc in docs)}")
    return docs


def load_medmentions(
    file_path: Optional[str] = None, dataset_format: str = "auto"
) -> List[MedMentionsDocument]:
    if file_path is None:
        cwd = os.getcwd()
        file_path = os.path.join(
            cwd,
            "datasets",
            "medmentions_St21pv",
            "corpus_pubtator.txt.gz",
        )

    if dataset_format == "auto":
        # Auto-detect format
        if os.path.isdir(file_path) and os.path.exists(
            os.path.join(file_path, "dataset_dict.json")
        ):
            dataset_format = "huggingface"
        elif os.path.isfile(file_path) and (
            file_path.endswith(".txt") or file_path.endswith(".txt.gz")
        ):
            dataset_format = "pubtator"
        else:
            # Default to HuggingFace if directory exists
            if os.path.isdir(file_path):
                dataset_format = "huggingface"
            else:
                raise ValueError(f"Cannot auto-detect dataset format for: {file_path}")

    if dataset_format == "pubtator":
        return load_pubtator_st21(file_path)
    elif dataset_format == "huggingface":
        # Load all splits and combine
        all_docs = []
        for split in ["train", "validation", "test"]:
            try:
                docs = load_medmentions_hf(file_path, split)
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {split} split")
            except Exception as e:
                logger.warning(f"Could not load {split} split: {e}")
        return all_docs
    else:
        raise ValueError(f"Unknown dataset format: {dataset_format}")


def save_medmentions_artifacts(docs: List[MedMentionsDocument], output_dir: str = None):
    """Save MedMentions data as JSON artifacts"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")

    os.makedirs(output_dir, exist_ok=True)

    # Convert to serializable format
    docs_data = []
    for doc in docs:
        docs_data.append({"pmid": doc.pmid, "text": doc.text, "spans": doc.spans})

    # Save artifacts
    artifacts = {
        "medmentions_docs.json": docs_data,
        "medmentions_stats.json": {
            "total_documents": len(docs),
            "total_entities": sum(len(doc.spans) for doc in docs),
            "unique_pmids": len(set(doc.pmid for doc in docs)),
            "avg_entities_per_doc": (
                sum(len(doc.spans) for doc in docs) / len(docs) if docs else 0
            ),
        },
    }

    for filename, data in artifacts.items():
        filepath = os.path.join(output_dir, filename)
        logger.info(f"Saving {filename}...")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {filepath}")

    return output_dir


if __name__ == "__main__":
    # Test the loader
    logger.info("Testing MedMentions ST21pv loader...")

    try:
        # Try loading from default location
        docs = load_medmentions()

        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Total documents: {len(docs)}")
        logger.info(f"  Total entities: {sum(len(doc.spans) for doc in docs)}")
        logger.info(f"  Unique PMIDs: {len(set(doc.pmid for doc in docs))}")

        if docs:
            logger.info(f"\nSample Document:")
            sample = docs[0]
            logger.info(f"  PMID: {sample.pmid}")
            logger.info(f"  Text: {sample.text[:200]}...")
            logger.info(f"  Entities: {len(sample.spans)}")
            for i, span in enumerate(sample.spans[:3]):
                logger.info(f"    {i+1}. {span}")

            # Save artifacts
            artifacts_dir = save_medmentions_artifacts(docs)
            logger.info(f"\nArtifacts saved to: {artifacts_dir}")

    except Exception as e:
        logger.error(f"Error loading MedMentions: {e}")
        logger.error("Make sure the dataset is available in the expected location.")
