#!/usr/bin/env python3

import os
import re
import json
import logging
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from logging_setup import get_logger

logger = get_logger(__name__)

try:
    from quickumls import QuickUMLS

    QUICKUMLS_AVAILABLE = True
except ImportError:
    QUICKUMLS_AVAILABLE = False
    logger.warning("QuickUMLS not available - install with: pip install quickumls")


class CLSPoolingEncoder:
    def __init__(
        self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 128,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **kwargs,
    ):
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []
        rng = range(0, len(texts), max(1, batch_size))
        for i in rng:
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=25,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = self.model(**enc)
                cls = outputs.last_hidden_state[:, 0, :]
            if convert_to_numpy:
                all_embeddings.append(cls.detach().cpu().numpy())
            else:
                all_embeddings.append(cls)

        if convert_to_numpy:
            return np.concatenate(all_embeddings, axis=0)
        return torch.cat(all_embeddings, dim=0)


def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'["\',;!?]', " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_quickumls_matcher(
    index_path=None,
    threshold=0.7,
    similarity_name="jaccard",
    window=5,
    min_match_length=3,
):
    if index_path is None:
        env_path = os.environ.get("OGR_QUICKUMLS_INDEX")
        if env_path and os.path.exists(env_path):
            index_path = env_path
        else:
            possible_paths = ["/Users/lukablaskovic/Github/PhD/db/UMLS"]
            index_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    index_path = path
                    break
            if index_path is None:
                raise FileNotFoundError(
                    f"QuickUMLS index not found. Tried env OGR_QUICKUMLS_INDEX and: {possible_paths}"
                )

    return QuickUMLS(
        index_path,
        threshold=threshold,
        similarity_name=similarity_name,
        window=window,
        min_match_length=min_match_length,
    )


_qu = None
_sapbert_system = None
_biobert_system = None


def get_matcher():
    global _qu
    if _qu is None and QUICKUMLS_AVAILABLE:
        _qu = get_quickumls_matcher()
    return _qu


class BioBERTCandidateSystem:

    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1"):
        self.model_name = model_name
        self.biobert = None
        self.index = None
        self.terms = []
        self.term_idx_to_cui = {}
        self.cui_to_term = {}
        self.is_initialized = False

    def _load_biobert(self):
        logger.info(f"Loading BioBERT model: {self.model_name}")
        self.biobert = CLSPoolingEncoder(self.model_name)
        logger.info("BioBERT model loaded")

    def _load_cui_preferred_terms(self, artifacts_dir=None):
        if artifacts_dir is None:
            artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        cui2pref_path = os.path.join(artifacts_dir, "cui2pref.json")
        if not os.path.exists(cui2pref_path):
            raise FileNotFoundError(
                f"CUI preferred terms file not found: {cui2pref_path}"
            )
        logger.info(f"Loading CUI preferred terms from: {cui2pref_path}")
        with open(cui2pref_path, "r", encoding="utf-8") as f:
            cui2pref = json.load(f)
        logger.info(f"Loaded {len(cui2pref)} CUI preferred terms")
        return cui2pref

    def build_candidate_pool(
        self,
        target_cuis: Optional[Set[str]] = None,
        max_pool_size: int = 10000,
        artifacts_dir=None,
    ):
        cui2pref = self._load_cui_preferred_terms(artifacts_dir)

        if target_cuis is None:
            logger.info(
                "No target CUIs provided. Creating demo pool with medical-related CUIs..."
            )
            medical_keywords = {
                "disease",
                "syndrome",
                "disorder",
                "infection",
                "cancer",
                "tumor",
                "diabetes",
                "hypertension",
                "cardiac",
                "heart",
                "lung",
                "kidney",
                "blood",
                "brain",
                "nerve",
                "muscle",
                "bone",
                "skin",
                "liver",
                "pain",
                "fever",
                "inflammation",
                "therapy",
                "treatment",
                "drug",
                "medication",
                "antibiotic",
                "virus",
                "bacteria",
                "patient",
                "medical",
                "clinical",
                "hospital",
                "diagnosis",
                "symptom",
            }

            target_cuis = set()
            for cui, term in cui2pref.items():
                if any(keyword in term.lower() for keyword in medical_keywords):
                    target_cuis.add(cui)
                    if len(target_cuis) >= max_pool_size:
                        break

            logger.info(f"Selected {len(target_cuis)} medical CUIs for demo pool")

        filtered_terms = []
        filtered_cuis = []

        for cui in target_cuis:
            if cui in cui2pref:
                term = cui2pref[cui]
                if term and term.strip():
                    filtered_terms.append(term.strip())
                    filtered_cuis.append(cui)

        self.terms = filtered_terms
        self.cui_to_term = {
            cui: term for cui, term in zip(filtered_cuis, filtered_terms)
        }
        self.term_idx_to_cui = {i: cui for i, cui in enumerate(filtered_cuis)}

        logger.info(
            f"Built candidate pool with {len(self.terms)} unique preferred terms"
        )

    def build_faiss_index(self, batch_size=128, cache_dir="candidate_embeddings_cache"):
        if not self.terms:
            raise ValueError(
                "No candidate pool built. Call build_candidate_pool() first."
            )
        if self.biobert is None:
            self._load_biobert()

        os.makedirs(cache_dir, exist_ok=True)
        import hashlib

        terms_str = "\n".join(sorted(self.terms))
        terms_hash = hashlib.md5(terms_str.encode()).hexdigest()[:12]
        cache_file = os.path.join(cache_dir, f"biobert_embeddings_{terms_hash}.pkl")
        if os.path.exists(cache_file):
            try:
                logger.info(f"Loading BioBERT embeddings from cache: {cache_file}")
                import pickle

                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    emb = cache_data["embeddings"]
                    cached_terms = cache_data["terms"]

                if sorted(cached_terms) == sorted(self.terms):
                    logger.info(f"Loaded {len(self.terms)} embeddings from cache")
                    if cached_terms != self.terms:
                        logger.info(
                            "Reordering embeddings to match current term order..."
                        )
                        term_to_idx = {term: i for i, term in enumerate(cached_terms)}
                        import numpy as np

                        reordered_emb = np.zeros_like(emb)
                        for i, term in enumerate(self.terms):
                            if term in term_to_idx:
                                reordered_emb[i] = emb[term_to_idx[term]]
                        emb = reordered_emb
                else:
                    logger.info("Cache terms mismatch, recomputing...")
                    raise ValueError("Cache mismatch")
            except Exception as e:
                logger.warning(f"Cache load failed ({e}), recomputing embeddings...")
                emb = None
        else:
            emb = None

        if emb is None:
            logger.info(
                f"Encoding {len(self.terms)} terms with BioBERT (batch_size={batch_size})..."
            )
            emb = self.biobert.encode(
                self.terms,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

            try:
                import pickle

                cache_data = {
                    "embeddings": emb,
                    "terms": self.terms,
                    "model_name": self.model_name,
                }
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Saved embeddings to cache: {cache_file}")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

        logger.info(f"Created embeddings with shape: {emb.shape}")

        import numpy as _np

        emb = _np.asarray(emb, dtype=_np.float32)
        emb = _np.ascontiguousarray(emb)

        norms = _np.linalg.norm(emb, axis=1, keepdims=True)
        norms = _np.clip(norms, 1e-8, None)
        self._emb_matrix = emb / norms
        self.index = None
        self.is_initialized = True
        logger.info(f"NumPy cosine kNN ready with {self._emb_matrix.shape[0]} vectors")

    def biobert_knn(self, text_span: str, k: int = 25) -> List[Tuple[str, float]]:
        if not self.is_initialized:
            raise ValueError(
                "BioBERT system not initialized. Call build_candidate_pool() and build_faiss_index() first."
            )

        if not text_span or not text_span.strip():
            return []

        try:
            import numpy as _np

            q = self.biobert.encode([text_span.strip()], convert_to_numpy=True)

            q = _np.asarray(q, dtype=_np.float32)
            q_norm = _np.linalg.norm(q, axis=1, keepdims=True)
            q_norm = _np.clip(q_norm, 1e-8, None)
            qn = q / q_norm
            sims = qn @ self._emb_matrix.T
            sims = sims[0]
            top_k = min(k, sims.shape[0])
            idx = _np.argpartition(-sims, top_k - 1)[:top_k]
            idx = idx[_np.argsort(-sims[idx])]

            results = [(self.term_idx_to_cui[i], float(sims[i])) for i in idx]
            return results

        except Exception as e:
            logger.error(f"Error in biobert_knn: {e}")
            return []

    def get_stats(self) -> Dict:
        return {
            "is_initialized": self.is_initialized,
            "model_name": self.model_name,
            "pool_size": len(self.terms),
            "index_size": self.index.ntotal if self.index else 0,
            "biobert_loaded": self.biobert is not None,
        }


class SapBERTCandidateSystem:

    def __init__(self, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        self.model_name = model_name
        self.sapbert = None
        self.index = None
        self.terms = []
        self.term_idx_to_cui = {}
        self.cui_to_term = {}
        self.is_initialized = False

    def _load_sapbert(self):
        logger.info(f"Loading SapBERT model: {self.model_name}")
        self.sapbert = CLSPoolingEncoder(self.model_name)
        logger.info("SapBERT model loaded")

    def _load_cui_preferred_terms(self, artifacts_dir=None):
        if artifacts_dir is None:
            artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        cui2pref_path = os.path.join(artifacts_dir, "cui2pref.json")
        if not os.path.exists(cui2pref_path):
            raise FileNotFoundError(
                f"CUI preferred terms file not found: {cui2pref_path}"
            )
        logger.info(f"Loading CUI preferred terms from: {cui2pref_path}")
        with open(cui2pref_path, "r", encoding="utf-8") as f:
            cui2pref = json.load(f)
        logger.info(f"Loaded {len(cui2pref)} CUI preferred terms")
        return cui2pref

    def build_candidate_pool(
        self,
        target_cuis: Optional[Set[str]] = None,
        max_pool_size: int = 10000,
        artifacts_dir=None,
    ):
        cui2pref = self._load_cui_preferred_terms(artifacts_dir)

        if target_cuis is None:
            logger.info(
                "No target CUIs provided. Creating demo pool with medical-related CUIs..."
            )
            medical_keywords = {
                "disease",
                "syndrome",
                "disorder",
                "infection",
                "cancer",
                "tumor",
                "diabetes",
                "hypertension",
                "cardiac",
                "heart",
                "lung",
                "kidney",
                "blood",
                "brain",
                "nerve",
                "muscle",
                "bone",
                "skin",
                "liver",
                "pain",
                "fever",
                "inflammation",
                "therapy",
                "treatment",
                "drug",
                "medication",
                "antibiotic",
                "virus",
                "bacteria",
                "patient",
                "medical",
                "clinical",
                "hospital",
                "diagnosis",
                "symptom",
            }

            target_cuis = set()
            for cui, term in cui2pref.items():
                if any(keyword in term.lower() for keyword in medical_keywords):
                    target_cuis.add(cui)
                    if len(target_cuis) >= max_pool_size:
                        break

            logger.info(f"Selected {len(target_cuis)} medical CUIs for demo pool")

        filtered_terms = []
        filtered_cuis = []

        for cui in target_cuis:
            if cui in cui2pref:
                term = cui2pref[cui]
                if term and term.strip():
                    filtered_terms.append(term.strip())
                    filtered_cuis.append(cui)

        self.terms = filtered_terms
        self.cui_to_term = {
            cui: term for cui, term in zip(filtered_cuis, filtered_terms)
        }
        self.term_idx_to_cui = {i: cui for i, cui in enumerate(filtered_cuis)}

        logger.info(
            f"Built candidate pool with {len(self.terms)} unique preferred terms"
        )

    def build_faiss_index(self, batch_size=128, cache_dir="candidate_embeddings_cache"):

        if not self.terms:
            raise ValueError(
                "No candidate pool built. Call build_candidate_pool() first."
            )

        if self.sapbert is None:
            self._load_sapbert()

        os.makedirs(cache_dir, exist_ok=True)
        import hashlib

        terms_str = "\n".join(sorted(self.terms))
        terms_hash = hashlib.md5(terms_str.encode()).hexdigest()[:12]
        cache_file = os.path.join(cache_dir, f"sapbert_embeddings_{terms_hash}.pkl")
        if os.path.exists(cache_file):
            try:
                logger.info(f" Loading SapBERT embeddings from cache: {cache_file}")
                import pickle

                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    emb = cache_data["embeddings"]
                    cached_terms = cache_data["terms"]
                if sorted(cached_terms) == sorted(self.terms):
                    logger.info(f" Loaded {len(self.terms)} embeddings from cache")
                    if cached_terms != self.terms:
                        logger.info(
                            " Reordering embeddings to match current term order..."
                        )
                        term_to_idx = {term: i for i, term in enumerate(cached_terms)}
                        import numpy as np

                        reordered_emb = np.zeros_like(emb)
                        for i, term in enumerate(self.terms):
                            if term in term_to_idx:
                                reordered_emb[i] = emb[term_to_idx[term]]
                        emb = reordered_emb
                else:
                    logger.info("  Cache terms mismatch, recomputing...")
                    raise ValueError("Cache mismatch")
            except Exception as e:
                logger.info(f"  Cache load failed ({e}), recomputing embeddings...")
                emb = None
        else:
            emb = None
        if emb is None:
            logger.info(
                f"Encoding {len(self.terms)} terms with SapBERT (batch_size={batch_size})..."
            )
            emb = self.sapbert.encode(
                self.terms,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            try:
                import pickle

                cache_data = {
                    "embeddings": emb,
                    "terms": self.terms,
                    "model_name": self.model_name,
                }
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f" Saved embeddings to cache: {cache_file}")
            except Exception as e:
                logger.info(f"  Failed to save cache: {e}")

        logger.info(f"Created embeddings with shape: {emb.shape}")

        import numpy as _np

        emb = _np.asarray(emb, dtype=_np.float32)
        emb = _np.ascontiguousarray(emb)

        norms = _np.linalg.norm(emb, axis=1, keepdims=True)
        norms = _np.clip(norms, 1e-8, None)
        self._emb_matrix = emb / norms
        self.index = None
        self.is_initialized = True
        logger.info(f"NumPy cosine kNN ready with {self._emb_matrix.shape[0]} vectors")

    def sapbert_knn(self, text_span: str, k: int = 25) -> List[Tuple[str, float]]:
        if not self.is_initialized:
            raise ValueError(
                "SapBERT system not initialized. Call build_candidate_pool() and build_faiss_index() first."
            )

        if not text_span or not text_span.strip():
            return []

        try:
            import numpy as _np

            q = self.sapbert.encode([text_span.strip()], convert_to_numpy=True)

            q = _np.asarray(q, dtype=_np.float32)
            q_norm = _np.linalg.norm(q, axis=1, keepdims=True)
            q_norm = _np.clip(q_norm, 1e-8, None)
            qn = q / q_norm
            sims = qn @ self._emb_matrix.T
            sims = sims[0]
            top_k = min(k, sims.shape[0])
            idx = _np.argpartition(-sims, top_k - 1)[:top_k]
            idx = idx[_np.argsort(-sims[idx])]

            results = [(self.term_idx_to_cui[i], float(sims[i])) for i in idx]
            return results

        except Exception as e:
            logger.error(f"Error in sapbert_knn: {e}")
            return []

    def get_stats(self) -> Dict:
        return {
            "is_initialized": self.is_initialized,
            "model_name": self.model_name,
            "pool_size": len(self.terms),
            "index_size": self.index.ntotal if self.index else 0,
            "sapbert_loaded": self.sapbert is not None,
        }


def get_biobert_system():
    global _biobert_system
    if _biobert_system is None:
        _biobert_system = BioBERTCandidateSystem()
        try:
            artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
            docs_path = os.path.join(artifacts_dir, "medmentions_docs.json")
            target_cuis = None
            if os.path.exists(docs_path):
                with open(docs_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                split_idx = int(0.8 * len(docs)) if docs else 0
                train_dev_docs = docs[:split_idx] if split_idx > 0 else docs
                target_cuis = {
                    cui
                    for d in train_dev_docs
                    for (s, e, m, cui, types) in d.get("spans", [])
                    if cui
                }
                logger.info(
                    f"Building BioBERT pool from MedMentions train+dev: {len(target_cuis)} CUIs"
                )
            else:
                logger.info(
                    f"MedMentions artifacts not found at {docs_path}. Falling back to heuristic pool."
                )
            _biobert_system.build_candidate_pool(
                target_cuis=target_cuis,
                max_pool_size=20000,
                artifacts_dir=artifacts_dir,
            )
            _biobert_system.build_faiss_index()
        except Exception as e:
            logger.warning(f" Could not initialize BioBERT system: {e}")
            _biobert_system = None
    return _biobert_system


def get_sapbert_system():
    global _sapbert_system
    if _sapbert_system is None:
        _sapbert_system = SapBERTCandidateSystem()
        try:
            artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
            docs_path = os.path.join(artifacts_dir, "medmentions_docs.json")
            target_cuis = None
            if os.path.exists(docs_path):
                with open(docs_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                split_idx = int(0.8 * len(docs)) if docs else 0
                train_dev_docs = docs[:split_idx] if split_idx > 0 else docs
                target_cuis = {
                    cui
                    for d in train_dev_docs
                    for (s, e, m, cui, types) in d.get("spans", [])
                    if cui
                }
                logger.info(
                    f"Building SapBERT pool from MedMentions train+dev: {len(target_cuis)} CUIs"
                )
            else:
                logger.info(
                    f"MedMentions artifacts not found at {docs_path}. Falling back to heuristic pool."
                )
            _sapbert_system.build_candidate_pool(
                target_cuis=target_cuis,
                max_pool_size=20000,
                artifacts_dir=artifacts_dir,
            )
            _sapbert_system.build_faiss_index()
        except Exception as e:
            logger.warning(f" Could not initialize SapBERT system: {e}")
            _sapbert_system = None
    return _sapbert_system


def quickumls_candidates(text_span, topN=25, normalize=True, use_sapbert_fallback=True):
    if not text_span or not text_span.strip():
        return []

    original_text = text_span
    if normalize:
        text_span = normalize_text(text_span)

    if not text_span:
        return []

    quickumls_results = []

    if QUICKUMLS_AVAILABLE:
        try:
            qu = get_matcher()
            if qu is not None:
                matches = qu.match(text_span, best_match=False, ignore_syntax=False)
                cands = []
                for m in matches:
                    for v in m:
                        cands.append((v["cui"], v["similarity"]))
                best = {}
                for cui, sim in cands:
                    best[cui] = max(best.get(cui, 0.0), sim)
                quickumls_results = sorted(best.items(), key=lambda x: -x[1])[:topN]
        except Exception as e:
            logger.warning(f" QuickUMLS error: {e}")

    if not quickumls_results and use_sapbert_fallback:
        try:
            sapbert_system = get_sapbert_system()
            if sapbert_system and sapbert_system.is_initialized:
                logger.info(
                    f"QuickUMLS found no candidates for '{original_text[:50]}...', using SapBERT fallback"
                )
                sapbert_results = sapbert_system.sapbert_knn(
                    original_text.strip(), k=topN
                )
                return sapbert_results
            else:
                logger.warning("SapBERT fallback not available")
        except Exception as e:
            logger.warning(f" SapBERT fallback error: {e}")

    return quickumls_results


def test_candidates():
    test_texts = [
        "myocardial infarction",
        "diabetes mellitus",
        "hypertension",
        "covid-19",
        "pneumonia",
        "acute kidney failure",
        "What is the prevalence of myocardial infarction in the last year?",
        "The patient has diabetes and high blood pressure.",
        "rare genetic disorder",
        "unknown medical condition",
    ]

    logger.info("Testing QuickUMLS + SapBERT fallback candidate generation...")
    logger.info(f"QuickUMLS config: jaccard similarity, threshold=0.8, window=5")
    logger.info(f"SapBERT fallback: cosine similarity, k-NN search")

    sapbert_system = get_sapbert_system()
    if sapbert_system:
        stats = sapbert_system.get_stats()
        logger.info(f"\nSapBERT System Status:")
        logger.info(f"   Initialized: {stats['is_initialized']}")
        logger.info(f"   Pool size: {stats['pool_size']}")
        logger.info(f"   Model: {stats['model_name']}")
    else:
        logger.info("\nSapBERT system not available")

    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n{i}. Input: '{text}'")
        candidates = quickumls_candidates(text, topN=5, use_sapbert_fallback=True)

        if candidates:
            logger.info(f"   Found {len(candidates)} candidates:")
            for j, (cui, score) in enumerate(candidates, 1):
                logger.info(f"     {j}. {cui} (similarity: {score:.3f})")
        else:
            logger.info("   No candidates found")

        if sapbert_system and sapbert_system.is_initialized:
            sapbert_candidates = sapbert_system.sapbert_knn(text, k=3)
            if sapbert_candidates:
                logger.info(f"   SapBERT direct (top 3):")
                for j, (cui, score) in enumerate(sapbert_candidates[:3], 1):
                    term = sapbert_system.cui_to_term.get(cui, "Unknown term")
                    logger.info(
                        f"     {j}. {cui} -> '{term[:50]}...' (sim: {score:.3f})"
                    )

    logger.info("\nTesting completed")


def initialize_sapbert_system(
    target_cuis: Optional[Set[str]] = None,
    max_pool_size: int = 10000,
    artifacts_dir: Optional[str] = None,
    force_rebuild: bool = False,
) -> Optional[SapBERTCandidateSystem]:
    global _sapbert_system

    if _sapbert_system and _sapbert_system.is_initialized and not force_rebuild:
        logger.info("SapBERT system already initialized")
        return _sapbert_system

    try:
        logger.info("Initializing SapBERT candidate system...")
        _sapbert_system = SapBERTCandidateSystem()
        _sapbert_system.build_candidate_pool(
            target_cuis=target_cuis,
            max_pool_size=max_pool_size,
            artifacts_dir=artifacts_dir,
        )
        _sapbert_system.build_faiss_index()

        stats = _sapbert_system.get_stats()
        logger.info(f"SapBERT system initialized successfully!")
        logger.info(f"   Pool size: {stats['pool_size']} terms")
        logger.info(f"   Index size: {stats['index_size']} vectors")

        return _sapbert_system

    except Exception as e:
        logger.error(f"Failed to initialize SapBERT system: {e}")
        _sapbert_system = None
        return None


if __name__ == "__main__":
    test_candidates()
