import pandas as pd
import csv
import os
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import json

MCOLS = [
    "CUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "AUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
]


def process_mrconso_chunk(chunk_data):
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
            if not r or len(r) < len(MCOLS):
                continue

            rec = {c: r[i] for i, c in enumerate(MCOLS)}

            # Apply filters: LAT='ENG', SUPPRESS≠'Y'
            if rec["LAT"] != "ENG" or rec["SUPPRESS"] == "Y":
                continue

            rows.append((rec["CUI"], rec["TTY"], rec["ISPREF"], rec["STR"].strip()))

    return rows


def load_mrconso(umls_path=None):
    if umls_path is None:
        # Default path
        cwd = os.getcwd()
        root_dir = os.path.dirname(cwd)  # Get parent directory (PhD folder)
        umls_path = os.path.join(
            root_dir, "datasets", "UMLS_raw", "META", "MRCONSO.RRF"
        )

    print(f"Loading MRCONSO from: {umls_path}")

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
        chunk_results = pool.map(process_mrconso_chunk, chunks)
        for result in chunk_results:
            all_rows.extend(result)

    # Create DataFrame and remove duplicates
    df = pd.DataFrame(
        all_rows, columns=["CUI", "TTY", "ISPREF", "STR"]
    ).drop_duplicates()
    print(f"Loaded {len(df)} unique MRCONSO records after filtering")

    return build_cui_lex(df)


def build_cui_lex(df):
    # Create CUI -> all strings mapping
    cui2strings = df.groupby("CUI")["STR"].apply(list).to_dict()

    def pick_pref(g):
        # Prefer PT/PN first
        ptpn = g[(g.TTY == "PT") | (g.TTY == "PN")]
        if len(ptpn):
            return ptpn["STR"].iloc[0]

        # Then ISPREF=Y
        pref = g[g.ISPREF == "Y"]
        if len(pref):
            return pref["STR"].iloc[0]

        # Fallback to first string
        return g["STR"].iloc[0]

    # Create CUI -> preferred term mapping
    pref = df.groupby("CUI").apply(pick_pref)
    cui2pref = pref.to_dict()

    print(
        f"Built lexicons: {len(cui2strings)} CUIs with strings, {len(cui2pref)} with preferred terms"
    )

    return cui2strings, cui2pref
