"""
INLP HW3 - Phase 3: 混合檢索 (Hybrid Search: BM25 + Dense via RRF)
"""
import json
import os
import glob
from typing import Dict, List

from tqdm import tqdm

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict,
)
from evaluation import recall_at_k, error_analysis
from submission_utils import (
    generate_submission, save_metrics, append_leaderboard,
    save_rankings, load_rankings,
)

config.print_config()


def reciprocal_rank_fusion(
    rankings_list: List[Dict[int, List[str]]],
    k: int = 60,
    weights: List[float] = None,
    top_k: int = 100,
) -> Dict[int, List[str]]:
    """
    Reciprocal Rank Fusion (RRF)
    
    Args:
        rankings_list: [ranking_dict_1, ranking_dict_2, ...]
                       每個 ranking_dict = {q_id: [quote_id 依分數排序]}
        k: RRF 平滑常數
        weights: 各 ranking 的權重，預設均等
        top_k: 輸出前幾名
        
    Returns:
        {q_id: [fused_quote_id 排序]}
    """
    if weights is None:
        weights = [1.0] * len(rankings_list)
    
    # 收集所有 q_id
    all_qids = set()
    for rankings in rankings_list:
        all_qids.update(rankings.keys())
    
    fused = {}
    for q_id in all_qids:
        scores = {}
        
        for rank_idx, rankings in enumerate(rankings_list):
            ranked_list = rankings.get(q_id, [])
            w = weights[rank_idx]
            
            for rank, quote_id in enumerate(ranked_list, start=1):
                if quote_id not in scores:
                    scores[quote_id] = 0.0
                scores[quote_id] += w / (k + rank)
        
        # 排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused[q_id] = [qid for qid, _ in sorted_items[:top_k]]
    
    return fused


def find_latest_rankings(phase_dir: str, filename: str = "rankings_top100.json") -> str:
    """在 phase 目錄中找最新的 rankings 檔"""
    phase_path = os.path.join(config.OUTPUT_DIR, phase_dir)
    if not os.path.exists(phase_path):
        return None
    
    candidates = []
    for subdir in os.listdir(phase_path):
        filepath = os.path.join(phase_path, subdir, filename)
        if os.path.exists(filepath):
            candidates.append(filepath)
    
    if not candidates:
        return None
    
    # 取最新修改的
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    # ── 載入資料 ──────────────────────────────────────
    train_data = load_train()
    test_data = load_test()
    train_subset, dev_subset = split_train_dev(train_data)
    
    print(f"📊 Train: {len(train_subset)}, Dev: {len(dev_subset)}, Test: {len(test_data)}")
    
    # ── 尋找先前 Phase 的 rankings ───────────────────
    bm25_test_path = find_latest_rankings("phase_1", "rankings_top100.json")
    dense_test_path = find_latest_rankings("phase_2", "rankings_top100.json")
    bm25_dev_path = find_latest_rankings("phase_1", "dev_rankings_top100.json")
    dense_dev_path = find_latest_rankings("phase_2", "dev_rankings_top100.json")
    
    if not bm25_test_path or not dense_test_path:
        print("❌ 需要先跑 Phase 1 (BM25) 和 Phase 2 (Dense) 才能做 Hybrid!")
        print(f"  BM25 test rankings: {bm25_test_path}")
        print(f"  Dense test rankings: {dense_test_path}")
        return
    
    print(f"📂 BM25 rankings: {bm25_test_path}")
    print(f"📂 Dense rankings: {dense_test_path}")
    
    # ── 載入 rankings ────────────────────────────────
    bm25_test_rankings = load_rankings(bm25_test_path)
    dense_test_rankings = load_rankings(dense_test_path)
    
    # ── 輸出目錄 ──────────────────────────────────────
    output_dir = config.get_output_dir("phase_3", "hybrid_rrf")
    
    # ── Dev 評估 ──────────────────────────────────────
    if bm25_dev_path and dense_dev_path:
        print("\n🔍 Running RRF on dev set...")
        bm25_dev_rankings = load_rankings(bm25_dev_path)
        dense_dev_rankings = load_rankings(dense_dev_path)
        
        dev_fused = reciprocal_rank_fusion(
            [bm25_dev_rankings, dense_dev_rankings],
            k=config.RRF_K,
            weights=[1.0, 1.0],
            top_k=100,
        )
        
        dev_preds = {q: v[:5] for q, v in dev_fused.items()}
        dev_golds = get_gold_quotes_dict(dev_subset)
        
        dev_recall = recall_at_k(dev_preds, dev_golds, k=5)
        print(f"\n📈 Dev Recall@5 (Hybrid RRF) = {dev_recall:.4f}")
        
        ea = error_analysis(dev_preds, dev_golds, k=5)
        print(f"  Perfect: {ea['summary']['perfect_count']}")
        print(f"  Partial: {ea['summary']['partial_count']}")
        print(f"  Zero:    {ea['summary']['zero_count']}")
    else:
        print("⚠️ Dev rankings 不齊全，跳過 dev 評估")
        dev_recall = -1.0
        ea = None
    
    # ── Test 融合 ─────────────────────────────────────
    print("\n🔍 Running RRF on test set...")
    test_fused = reciprocal_rank_fusion(
        [bm25_test_rankings, dense_test_rankings],
        k=config.RRF_K,
        weights=[1.0, 1.0],
        top_k=100,
    )
    
    test_preds_top5 = {q: v[:5] for q, v in test_fused.items()}
    
    # ── 儲存結果 ──────────────────────────────────────
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds_top5, test_data, submission_path)
    
    rankings_path = os.path.join(output_dir, "rankings_top100.json")
    save_rankings(test_fused, rankings_path)
    
    if bm25_dev_path and dense_dev_path:
        dev_rankings_path = os.path.join(output_dir, "dev_rankings_top100.json")
        save_rankings(dev_fused, dev_rankings_path)
    
    metrics = {
        "phase": "phase_3",
        "model": "Hybrid RRF (BM25 + Dense)",
        "rrf_k": config.RRF_K,
        "weights": [1.0, 1.0],
        "bm25_source": bm25_test_path,
        "dense_source": dense_test_path,
        "dev_recall_at_5": dev_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
    }
    if ea:
        metrics["error_analysis_summary"] = ea["summary"]
    save_metrics(metrics, output_dir)
    
    if dev_recall > 0:
        append_leaderboard("Phase 3", "Hybrid RRF", dev_recall,
                           note=f"BM25+Dense, k={config.RRF_K}")
    
    print(f"\n✅ Phase 3 完成! 輸出目錄: {output_dir}")


if __name__ == "__main__":
    main()
