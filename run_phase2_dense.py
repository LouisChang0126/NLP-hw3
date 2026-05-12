"""
INLP HW3 - Phase 2: 稠密檢索 (Dense Retrieval)
使用 BGE-M3 或其他 dense embedding model
"""
import json
import os
import numpy as np
from typing import Dict, List

from tqdm import tqdm
import torch

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict,
)
from evaluation import recall_at_k, error_analysis
from submission_utils import (
    generate_submission, save_metrics, append_leaderboard, save_rankings,
)

config.print_config()


def load_embed_model():
    """載入 embedding model"""
    model_name = config.PHASE_2_EMBED_MODEL
    print(f"📦 Loading embedding model: {model_name}")
    
    if "bge" in model_name.lower():
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(model_name, use_fp16=True)
        return model, "bge-m3"
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model, "sentence-transformer"


def encode_texts(model, texts: List[str], model_type: str, batch_size: int = 64) -> np.ndarray:
    """將文字列表編碼成 dense embedding"""
    if model_type == "bge-m3":
        embeddings = model.encode(texts, batch_size=batch_size)["dense_vecs"]
        return np.array(embeddings)
    else:
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.array(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """計算 cosine similarity, a: (d,), b: (n, d) -> (n,)"""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def dense_retrieve_per_question(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidates: List[dict],
    top_k: int = 5,
) -> List[str]:
    """對單題做 dense retrieval"""
    if len(candidates) == 0:
        return []
    
    scores = cosine_similarity(query_emb, candidate_embs)
    sorted_indices = np.argsort(scores)[::-1]
    
    return [candidates[i]["quote_id"] for i in sorted_indices[:top_k]]


def run_dense_retrieval(
    data: List[dict],
    model,
    model_type: str,
    top_k: int = 100,
) -> Dict[int, List[str]]:
    """對整個 dataset 做 dense per-question 檢索"""
    results = {}
    
    # 批次收集所有 query 和 candidate 文字
    all_queries = []
    all_candidates_per_q = []
    
    for sample in data:
        candidates = build_candidates(sample)
        all_queries.append(sample["question"])
        all_candidates_per_q.append(candidates)
    
    # 編碼所有 query
    print("  Encoding queries...")
    query_embs = encode_texts(model, all_queries, model_type)
    
    # 編碼所有 candidate (批次)
    print("  Encoding candidates...")
    all_cand_texts = []
    cand_boundaries = [0]
    for candidates in all_candidates_per_q:
        texts = [c["text_for_retrieval"] for c in candidates]
        all_cand_texts.extend(texts)
        cand_boundaries.append(len(all_cand_texts))
    
    if all_cand_texts:
        all_cand_embs = encode_texts(model, all_cand_texts, model_type)
    else:
        all_cand_embs = np.array([])
    
    # Per-question 檢索
    for i, sample in enumerate(tqdm(data, desc="Dense retrieval")):
        q_id = sample["q_id"]
        q_emb = query_embs[i]
        
        start = cand_boundaries[i]
        end = cand_boundaries[i + 1]
        cand_embs = all_cand_embs[start:end]
        candidates = all_candidates_per_q[i]
        
        ranked = dense_retrieve_per_question(q_emb, cand_embs, candidates, top_k)
        results[q_id] = ranked
    
    return results


def main():
    # ── 載入資料 ──────────────────────────────────────
    train_data = load_train()
    test_data = load_test()
    train_subset, dev_subset = split_train_dev(train_data)
    
    print(f"📊 Train: {len(train_subset)}, Dev: {len(dev_subset)}, Test: {len(test_data)}")
    
    # ── 載入模型 ──────────────────────────────────────
    model, model_type = load_embed_model()
    
    # ── 輸出目錄 ──────────────────────────────────────
    output_dir = config.get_output_dir("phase_2", config.PHASE_2_EMBED_MODEL)
    
    # ── Dev 評估 ──────────────────────────────────────
    print("\n🔍 Running Dense Retrieval on dev set...")
    dev_preds = run_dense_retrieval(dev_subset, model, model_type, top_k=config.DENSE_TOP_K)
    dev_golds = get_gold_quotes_dict(dev_subset)
    
    dev_recall = recall_at_k(dev_preds, dev_golds, k=5)
    print(f"\n📈 Dev Recall@5 = {dev_recall:.4f}")
    
    ea = error_analysis(dev_preds, dev_golds, k=5)
    print(f"  Perfect: {ea['summary']['perfect_count']}")
    print(f"  Partial: {ea['summary']['partial_count']}")
    print(f"  Zero:    {ea['summary']['zero_count']}")
    
    # ── Test 預測 ─────────────────────────────────────
    print("\n🔍 Running Dense Retrieval on test set...")
    test_preds_large = run_dense_retrieval(test_data, model, model_type, top_k=config.DENSE_TOP_K_LARGE)
    test_preds_top5 = {q: v[:5] for q, v in test_preds_large.items()}
    
    # ── 儲存結果 ──────────────────────────────────────
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds_top5, test_data, submission_path)
    
    rankings_path = os.path.join(output_dir, "rankings_top100.json")
    save_rankings(test_preds_large, rankings_path)
    
    dev_preds_large = run_dense_retrieval(dev_subset, model, model_type, top_k=config.DENSE_TOP_K_LARGE)
    dev_rankings_path = os.path.join(output_dir, "dev_rankings_top100.json")
    save_rankings(dev_preds_large, dev_rankings_path)
    
    metrics = {
        "phase": "phase_2",
        "model": config.PHASE_2_EMBED_MODEL,
        "dev_recall_at_5": dev_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)
    
    append_leaderboard("Phase 2", config.PHASE_2_EMBED_MODEL, dev_recall)
    
    print(f"\n✅ Phase 2 完成! 輸出目錄: {output_dir}")


if __name__ == "__main__":
    main()
