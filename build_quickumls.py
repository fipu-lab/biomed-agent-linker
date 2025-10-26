#!/usr/bin/env python3

import os
import sys
import time
import re
import subprocess
import threading
import json
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_setup import setup_logging, get_logger

logger = get_logger(__name__)

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available - install with: pip install tqdm")


def get_file_sizes(source_path):
    meta_path = os.path.join(source_path, "META")
    sizes = {}

    for filename in ["MRCONSO.RRF", "MRSTY.RRF"]:
        filepath = os.path.join(meta_path, filename)
        if os.path.exists(filepath):
            sizes[filename] = os.path.getsize(filepath)

    return sizes


def monitor_directory_progress(output_dir, total_size_mb, progress_callback=None):
    if not TQDM_AVAILABLE:
        return

    pbar = tqdm(
        total=total_size_mb,
        unit="MB",
        desc="Building QuickUMLS index",
        ncols=100,
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}MB [{elapsed}<{remaining}, {rate_fmt}]",
    )

    start_time = time.time()
    last_size = 0

    while True:
        try:
            if os.path.exists(output_dir):
                total_size = 0
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(filepath)
                        except (OSError, FileNotFoundError):
                            pass

                current_size_mb = total_size / (1024 * 1024)

                # Update progress bar
                progress = min(current_size_mb, total_size_mb)
                pbar.n = progress
                pbar.refresh()

                if progress_callback:
                    progress_callback(progress, total_size_mb)

                if current_size_mb >= total_size_mb * 0.9:  # 90% threshold
                    break

                last_size = current_size_mb

            time.sleep(2)

        except Exception as e:
            time.sleep(5)

    pbar.close()


def run_subprocess_with_progress(cmd, output_dir, estimated_size_mb, timeout=7200):
    logger = get_logger(__name__)

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Estimated index size: {estimated_size_mb:.1f} MB")
    logger.info("This may take 1-3 hours depending on your system...")

    progress_thread = None
    if TQDM_AVAILABLE and estimated_size_mb > 0:
        progress_thread = threading.Thread(
            target=monitor_directory_progress,
            args=(output_dir, estimated_size_mb),
            daemon=True,
        )
        progress_thread.start()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                line = output.strip()
                output_lines.append(line)
                if any(
                    keyword in line.lower()
                    for keyword in [
                        "progress",
                        "building",
                        "creating",
                        "processing",
                        "done",
                        "error",
                        "warning",
                    ]
                ):
                    logger.info(f"QuickUMLS: {line}")

        return_code = process.poll()
        full_output = "\n".join(output_lines)

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output=full_output)

        return full_output

    except subprocess.TimeoutExpired:
        process.kill()
        raise

    finally:
        if progress_thread and progress_thread.is_alive():
            time.sleep(2)


def preprocess_umls_parallel(source_path, use_preprocessing=True):
    """Optionally preprocess UMLS data using our optimized loaders for faster index building."""
    if not use_preprocessing:
        return None

    logger = get_logger(__name__)
    logger.info("Preprocessing UMLS data with optimized loaders...")

    try:
        # Import our existing optimized loaders
        from data.load_mrconso import load_mrconso
        from data.load_mrsty import load_mrsty

        start_time = time.time()

        # Load and process data in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            mrconso_future = executor.submit(load_mrconso, source_path)
            mrsty_future = executor.submit(load_mrsty, source_path)

            # Get results
            cui2strings, cui2pref = mrconso_future.result()
            cui2types = mrsty_future.result()

        preprocessing_time = time.time() - start_time
        logger.info(f"UMLS preprocessing completed in {preprocessing_time:.2f} seconds")
        logger.info(f"   Loaded {len(cui2strings)} CUIs with strings")
        logger.info(f"   Loaded {len(cui2types)} CUIs with semantic types")

        return {
            "cui2strings": cui2strings,
            "cui2pref": cui2pref,
            "cui2types": cui2types,
            "preprocessing_time": preprocessing_time,
        }

    except Exception as e:
        logger.warning(f"UMLS preprocessing failed: {e}")
        logger.info("Continuing with standard QuickUMLS build...")
        return None


def normalize_text(text):
    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_quickumls_index(
    umls_source_path=None,
    index_output_path=None,
    force_rebuild=False,
    use_parallel_preprocessing=True,
    show_progress=True,
):
    logger = get_logger(__name__)

    if umls_source_path is None:
        cwd = os.getcwd()
        root_dir = os.path.dirname(cwd)
        umls_source_path = os.path.join(root_dir, "datasets", "UMLS_raw")

    if index_output_path is None:
        index_output_path = os.path.join(
            os.path.dirname(__file__), "data", "quickumls_index"
        )

    logger.info(f"Building QuickUMLS index with enhanced progress monitoring...")
    logger.info(f"   Source: {umls_source_path}")
    logger.info(f"   Output: {index_output_path}")
    logger.info(f"   Parallel preprocessing: {use_parallel_preprocessing}")
    logger.info(f"   Progress monitoring: {show_progress}")

    if not os.path.exists(umls_source_path):
        raise FileNotFoundError(f"UMLS source path not found: {umls_source_path}")

    meta_path = os.path.join(umls_source_path, "META")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"META directory not found in UMLS source: {meta_path}")

    required_files = ["MRCONSO.RRF", "MRSTY.RRF"]
    for file in required_files:
        file_path = os.path.join(meta_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required UMLS file not found: {file_path}")

    if os.path.exists(index_output_path) and not force_rebuild:
        logger.info(f"QuickUMLS index already exists at: {index_output_path}")
        logger.info("   Use force_rebuild=True to rebuild")
        return index_output_path

    file_sizes = get_file_sizes(umls_source_path)
    total_source_size_mb = sum(file_sizes.values()) / (1024 * 1024)
    estimated_index_size_mb = total_source_size_mb * 2.5

    logger.info(f"Source files: {total_source_size_mb:.1f} MB")
    logger.info(f"Estimated index size: {estimated_index_size_mb:.1f} MB")

    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)

    preprocessing_result = None
    if use_parallel_preprocessing:
        preprocessing_result = preprocess_umls_parallel(umls_source_path)

    logger.info("Starting QuickUMLS index build...")
    logger.info("⏰ Estimated time: 1-3 hours (depending on system and source size)")
    start_time = time.time()

    try:
        cmd = [
            sys.executable,
            "-m",
            "quickumls.install",
            umls_source_path,
            index_output_path,
        ]

        if show_progress and TQDM_AVAILABLE:
            output = run_subprocess_with_progress(
                cmd,
                index_output_path,
                estimated_index_size_mb,
                timeout=7200,
            )
        else:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

            if result.returncode != 0:
                logger.error(
                    f"QuickUMLS build failed with return code {result.returncode}"
                )
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"QuickUMLS index build failed: {result.stderr}")

            output = result.stdout

        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info(f"QuickUMLS index built successfully!")
        logger.info(
            f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        )
        logger.info(f"Index saved to: {index_output_path}")

        if os.path.exists(index_output_path):
            actual_size = 0
            for root, dirs, files in os.walk(index_output_path):
                for file in files:
                    try:
                        actual_size += os.path.getsize(os.path.join(root, file))
                    except (OSError, FileNotFoundError):
                        pass
            actual_size_mb = actual_size / (1024 * 1024)
            logger.info(f"Final index size: {actual_size_mb:.1f} MB")

        if preprocessing_result:
            logger.info(
                f"Preprocessing saved ~{preprocessing_result['preprocessing_time']:.1f}s of warmup time"
            )

        return index_output_path

    except subprocess.TimeoutExpired:
        logger.error("QuickUMLS build timed out after 2 hours")
        logger.error("Try running with smaller UMLS subset or on a faster system")
        raise
    except Exception as e:
        logger.error(f"QuickUMLS build failed: {e}")
        logger.error("Check UMLS source files and available disk space")
        raise


class QuickUMLSCandidateGenerator:
    """
    QuickUMLS-based candidate generator with professor's configuration.
    """

    def __init__(
        self,
        index_path,
        threshold=0.8,
        similarity_name="jaccard",
        window=5,
        min_match_length=3,
        accept_overlapping=True,
    ):
        """
        Initialize QuickUMLS matcher with professor's specifications.

        Args:
            index_path: Path to QuickUMLS index
            threshold: Similarity threshold (0.8, try 0.7-0.9 later)
            similarity_name: Similarity metric ("jaccard")
            window: Maximum tokens to consider (5)
            min_match_length: Minimum match length (3)
            accept_overlapping: Accept overlapping matches (True)
        """
        self.logger = get_logger(__name__)
        self.index_path = index_path
        self.threshold = threshold
        self.similarity_name = similarity_name
        self.window = window
        self.min_match_length = min_match_length
        self.accept_overlapping = accept_overlapping

        self._matcher = None
        self._initialize_matcher()

    def _initialize_matcher(self):
        try:
            from quickumls import QuickUMLS

            self.logger.info(f"Initializing QuickUMLS matcher...")
            self.logger.info(f"   Index: {self.index_path}")
            self.logger.info(f"   Threshold: {self.threshold}")
            self.logger.info(f"   Similarity: {self.similarity_name}")
            self.logger.info(f"   Window: {self.window}")

            self._matcher = QuickUMLS(
                self.index_path,
                threshold=self.threshold,
                similarity_name=self.similarity_name,
                window=self.window,
                min_match_length=self.min_match_length,
                verbose=False,
            )

            self.logger.info("QuickUMLS matcher initialized successfully")

        except ImportError:
            raise ImportError(
                "quickumls package not found. Install with: pip install quickumls"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize QuickUMLS: {e}")
            raise

    def quickumls_candidates(self, text_span, topN=25, normalize=True):

        if not text_span or not text_span.strip():
            return []

        if normalize:
            text_span = normalize_text(text_span)

        if not text_span:
            return []

        try:
            matches = self._matcher.match(
                text_span, best_match=True, ignore_syntax=False
            )

            # Flatten, take CUIs + their scores
            cands = []
            for m in matches:
                for v in m:
                    cands.append((v["cui"], v["similarity"]))

            # De-duplicate by CUI keeping max similarity
            best = {}
            for cui, sim in cands:
                best[cui] = max(best.get(cui, 0.0), sim)

            return sorted(best.items(), key=lambda x: -x[1])[:topN]

        except Exception as e:
            self.logger.error(f"Error in quickumls_candidates: {e}")
            return []

    def test_matching(self, test_texts=None):
        """Test the matcher with sample medical texts."""
        if test_texts is None:
            test_texts = [
                "myocardial infarction",
                "diabetes mellitus",
                "hypertension",
                "covid-19",
                "What is the prevalence of myocardial infarction in the last year?",
                "The patient has diabetes and high blood pressure.",
            ]

        self.logger.info(f"Testing QuickUMLS matching with {len(test_texts)} texts...")

        for i, text in enumerate(test_texts, 1):
            self.logger.info(f"\n{i}. Input: '{text}'")

            # Get candidates
            candidates = self.quickumls_candidates(text, topN=5)

            if candidates:
                self.logger.info(f"   Found {len(candidates)} candidates:")
                for j, (cui, score) in enumerate(candidates, 1):
                    self.logger.info(f"     {j}. {cui} (similarity: {score:.3f})")
            else:
                self.logger.info("   No candidates found")

        self.logger.info("\nQuickUMLS testing completed")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build QuickUMLS index with enhanced monitoring"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if index exists",
    )
    parser.add_argument(
        "--no-preprocessing", action="store_true", help="Skip parallel preprocessing"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress monitoring"
    )
    parser.add_argument("--source-path", type=str, help="Custom UMLS source path")
    parser.add_argument("--index-path", type=str, help="Custom index output path")

    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    logger.info("QuickUMLS Index Builder & Candidate Generator")
    logger.info("Professor's Configuration: jaccard, threshold=0.8, window=5")

    if args.no_preprocessing:
        logger.info("Parallel preprocessing disabled")
    if args.no_progress:
        logger.info("Progress monitoring disabled")

    try:
        logger.info("\nStep 1: Building QuickUMLS Index with Enhanced Features")
        index_path = build_quickumls_index(
            umls_source_path=args.source_path,
            index_output_path=args.index_path,
            force_rebuild=args.force_rebuild,
            use_parallel_preprocessing=not args.no_preprocessing,
            show_progress=not args.no_progress,
        )

        logger.info("\nStep 2: Initializing Candidate Generator")
        generator = QuickUMLSCandidateGenerator(
            index_path=index_path,
            threshold=0.8,  # Professor's config
            similarity_name="jaccard",  # Professor's config
            window=5,  # Professor's config
            min_match_length=3,
            accept_overlapping=True,  # Professor's config
        )

        logger.info("\nStep 3: Testing System")
        generator.test_matching()

        logger.info("\nQuickUMLS system ready!")
        logger.info(f"Index location: {index_path}")
        logger.info("\nUsage Example:")
        logger.info("   generator = QuickUMLSCandidateGenerator('/path/to/index')")
        logger.info(
            "   candidates = generator.quickumls_candidates('diabetes mellitus')"
        )

        return generator

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
