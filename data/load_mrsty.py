import pandas as pd
import csv
import os
from multiprocessing import Pool, cpu_count

STY_COLS = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]


def process_mrsty_chunk(chunk_data):
    chunk_start, chunk_size, file_path = chunk_data
    rows = []

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
            if not r or len(r) < 3:
                continue

            # CUI, TUI, STY (as per professor's specification)
            rows.append((r[0], r[1], r[3]))

    return rows


def load_mrsty(umls_path=None):
    if umls_path is None:
        # Default path
        cwd = os.getcwd()
        root_dir = os.path.dirname(cwd)  # Get parent directory (PhD folder)
        umls_path = os.path.join(root_dir, "datasets", "UMLS_raw", "META", "MRSTY.RRF")

    print(f"Loading MRSTY from: {umls_path}")

    # Count total lines for chunking
    with open(umls_path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Determine chunk size for multiprocessing
    num_cores = cpu_count()
    chunk_size = max(1000, total_lines // (num_cores * 2))
    chunks = []

    for start in range(0, total_lines, chunk_size):
        size = min(chunk_size, total_lines - start)
        chunks.append((start, size, umls_path))

    print(
        f"Processing {total_lines} lines using {num_cores} cores in {len(chunks)} chunks..."
    )

    # Process chunks in parallel
    all_rows = []
    with Pool(num_cores) as pool:
        chunk_results = pool.map(process_mrsty_chunk, chunks)
        for result in chunk_results:
            all_rows.extend(result)

    # Create DataFrame and remove duplicates
    df = pd.DataFrame(all_rows, columns=["CUI", "TUI", "STY"]).drop_duplicates()
    print(f"Loaded {len(df)} unique MRSTY records")

    # Create CUI -> set of TUIs mapping (as per professor's specification)
    cui2types = df.groupby("CUI")["TUI"].apply(set).to_dict()

    print(f"Built semantic type mapping: {len(cui2types)} CUIs with types")

    return cui2types
