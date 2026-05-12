"""
INLP HW3 - Phase 1: Simple Baseline (BM25 稀疏檢索)
"""
import json
import os
import re
from typing import Dict, List

from tqdm import tqdm
from rank_bm25 import BM25Okapi

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict,
)
from evaluation import recall_at_k, recall_at_k_per_sample, error_analysis
from submission_utils import (
    generate_submission, save_metrics, append_leaderboard, save_rankings,
)

config.print_config()


def tokenize(text: str) -> List[str]:
    """簡單分詞：小寫 + 用非字母數字切分"""
    return re.findall(r'\w+', text.lower())


def bm25_retrieve_per_question(
    question: str,
    candidates: List[dict],
    top_k: int = 5,
) -> List[str]:
    """
    對單題的候選池做 BM25 檢索。
    
    Args:
        question: 問題文字
        candidates: [{quote_id, modality, text_for_retrieval}]
        top_k: 回傳前幾名
        
    Returns:
        [quote_id, ...] 已按分數排序
    """
    if not candidates:
        return []
    
    # 建立 corpus
    corpus = [tokenize(c["text_for_retrieval"]) for c in candidates]
    
    # 避免空 corpus
    if all(len(doc) == 0 for doc in corpus):
        return [c["quote_id"] for c in candidates[:top_k]]
    
    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(question)
    
    if not query_tokens:
        return [c["quote_id"] for c in candidates[:top_k]]
    
    scores = bm25.get_scores(query_tokens)
    
    # 排序
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [c["quote_id"] for c, _ in scored[:top_k]]


def run_bm25_retrieval(data: List[dict], top_k: int = 100) -> Dict[int, List[str]]:
    """對整個 dataset 做 BM25 per-question 檢索"""
    results = {}
    for sample in tqdm(data, desc="BM25 retrieval"):
        q_id = sample["q_id"]
        question = sample["question"]
        candidates = build_candidates(sample)
        ranked = bm25_retrieve_per_question(question, candidates, top_k=top_k)
        results[q_id] = ranked
    return results


def main():
    # ── 載入資料 ──────────────────────────────────────
    train_data = load_train()
    test_data = load_test()
    train_subset, dev_subset = split_train_dev(train_data)
    
    print(f"📊 Train: {len(train_subset)}, Dev: {len(dev_subset)}, Test: {len(test_data)}")
    
    # ── 輸出目錄 ──────────────────────────────────────
    output_dir = config.get_output_dir("phase_1", "bm25")
    
    # ── Dev 評估 ──────────────────────────────────────
    print("\n🔍 Running BM25 on dev set...")
    dev_preds = run_bm25_retrieval(dev_subset, top_k=config.BM25_TOP_K)
    dev_golds = get_gold_quotes_dict(dev_subset)
    
    dev_recall = recall_at_k(dev_preds, dev_golds, k=5)
    print(f"\n📈 Dev Recall@5 = {dev_recall:.4f}")
    
    # Error analysis
    ea = error_analysis(dev_preds, dev_golds, k=5)
    print(f"  Perfect: {ea['summary']['perfect_count']}")
    print(f"  Partial: {ea['summary']['partial_count']}")
    print(f"  Zero:    {ea['summary']['zero_count']}")
    
    # ── Test 預測 ─────────────────────────────────────
    print("\n🔍 Running BM25 on test set...")
    test_preds_large = run_bm25_retrieval(test_data, top_k=config.BM25_TOP_K_LARGE)
    test_preds_top5 = {q: v[:5] for q, v in test_preds_large.items()}
    
    # ── 儲存結果 ──────────────────────────────────────
    # Submission
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds_top5, test_data, submission_path)
    
    # Rankings (供後續 RRF 使用)
    rankings_path = os.path.join(output_dir, "rankings_top100.json")
    save_rankings(test_preds_large, rankings_path)
    
    # Dev rankings (供後續 RRF 使用)
    dev_preds_large = run_bm25_retrieval(dev_subset, top_k=config.BM25_TOP_K_LARGE)
    dev_rankings_path = os.path.join(output_dir, "dev_rankings_top100.json")
    save_rankings(dev_preds_large, dev_rankings_path)
    
    # Metrics
    metrics = {
        "phase": "phase_1",
        "model": "BM25 (rank_bm25)",
        "dev_recall_at_5": dev_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)
    
    # Leaderboard
    append_leaderboard("Phase 1", "BM25", dev_recall, note="baseline")
    
    print(f"\n✅ Phase 1 完成! 輸出目錄: {output_dir}")


if __name__ == "__main__":
    main()
