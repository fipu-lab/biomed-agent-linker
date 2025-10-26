import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from candidate_gen import get_sapbert_system
from utils import get_sapbert_candidates_for_llm

print("=== TESTING CANDIDATE GENERATION ===")

try:
    print("Initializing SapBERT system...")
    sapbert_system = get_sapbert_system()

    if not sapbert_system or not sapbert_system.is_initialized:
        print("ERROR: SapBERT system not initialized!")
        exit(1)

    print(f"SapBERT system initialized: {sapbert_system.is_initialized}")

    mention = "heart attack"
    print(f"\nTesting candidate generation for: '{mention}'")

    raw_results = sapbert_system.sapbert_knn(mention, k=10)
    print(f"Raw SapBERT results: {len(raw_results)} candidates")
    for i, (cui, score) in enumerate(raw_results[:5]):
        print(f"  {i+1}. {cui}: {score:.3f}")

    candidates = get_sapbert_candidates_for_llm(
        mention, sapbert_system, max_candidates=10
    )
    print(f"\nLLM candidates: {len(candidates)} candidates")
    for i, candidate in enumerate(candidates[:5]):
        print(f"  {i+1}. {candidate}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
