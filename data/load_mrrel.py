import pandas as pd
import csv
import os
from multiprocessing import Pool, cpu_count
from collections import defaultdict

REL_COLS = [
    "CUI1",
    "AUI1",
    "STYPE1",
    "REL",
    "CUI2",
    "AUI2",
    "STYPE2",
    "RELA",
    "RUI",
    "SRUI",
    "SAB",
    "SL",
    "RG",
    "DIR",
    "SUPPRESS",
    "CVF",
]


def process_mrrel_chunk(chunk_data):
    chunk_start, chunk_size, file_path, allowed = chunk_data
    neighbors = defaultdict(set)

    with open(file_path, encoding="utf-8") as f:
        # Skip to start position
        for _ in range(chunk_start):
            f.readline()

        # Process chunk
        for i in range(chunk_size):
            line = f.readline()
            if not line:
                break

            r = line.strip().split("|")
            if not r or len(r) < 5:
                continue

            rel = r[3]  # REL column
            c1, c2 = r[0], r[4]  # CUI1, CUI2 columns

            # Filter by allowed relation types and non-empty CUIs
            if rel in allowed and c1 and c2:
                neighbors[c1].add(c2)
                neighbors[c2].add(c1)

    return neighbors


def load_mrrel(umls_path=None, allowed={"PAR", "CHD", "RB", "RO"}):
    if umls_path is None:
        # Default path
        cwd = os.getcwd()
        root_dir = os.path.dirname(cwd)  # Get parent directory (PhD folder)
        umls_path = os.path.join(root_dir, "datasets", "UMLS_raw", "META", "MRREL.RRF")

    print(f"Loading MRREL from: {umls_path}")
    print(f"Allowed relation types: {allowed}")

    # Count total lines for chunking
    with open(umls_path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Determine chunk size for multiprocessing
    num_cores = cpu_count()
    chunk_size = max(1000, total_lines // (num_cores * 2))
    chunks = []

    for start in range(0, total_lines, chunk_size):
        size = min(chunk_size, total_lines - start)
        chunks.append((start, size, umls_path, allowed))

    print(
        f"Processing {total_lines} lines using {num_cores} cores in {len(chunks)} chunks..."
    )

    # Process chunks in parallel
    all_neighbors = defaultdict(set)
    with Pool(num_cores) as pool:
        chunk_results = pool.map(process_mrrel_chunk, chunks)

        # Merge results from all chunks
        for chunk_neighbors in chunk_results:
            for cui, neighbors in chunk_neighbors.items():
                all_neighbors[cui].update(neighbors)

    # Convert sets to lists for JSON serialization
    cui2neighbors = {k: list(v) for k, v in all_neighbors.items()}

    print(f"Built neighbor mapping: {len(cui2neighbors)} CUIs with neighbors")

    return cui2neighbors
