"""
INLP HW3 - Phase 3b: BGE-M3 三路向量 + BM25 四路 RRF 融合 + Image-Score Boost

改進方向（vs Phase 3）:
  1. BGE-M3 同時輸出 dense + sparse (lexical_weights) + colbert (multi-vec)
     三路訊號全部加進 RRF，比僅用 dense 多兩路強檢索訊號
  2. 對 image-modality 候選的最終分數加 boost，校正 modality bias
     （Gold 60% image, 預測只有 41% image）
"""
import json
import os
import time
from typing import Dict, List

from tqdm import tqdm
import numpy as np
import torch

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
from run_phase3_hybrid import find_latest_rankings


# ── 可調超參數 ────────────────────────────────────────
RRF_K = 60
# 4-way 權重 (BM25, dense, lexical, colbert)
RRF_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
# 額外給 image candidate 的 RRF 分數補貼
IMAGE_BOOST = 0.010   # tune 範圍 0.005 ~ 0.03


def encode_with_bgem3(model, texts: List[str], batch_size: int = 32):
    """單次呼叫拿到 dense / sparse / colbert 三種向量"""
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    return out["dense_vecs"], out["lexical_weights"], out["colbert_vecs"]


def cosine_sim(a, b):
    """a: (d,), b: (n,d)"""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b @ a


def run_bgem3_retrieval(model, data: List[dict], top_k: int = 100):
    """
    對整個 dataset 做 BGE-M3 三路檢索。
    回傳: (dense_rank, lexical_rank, colbert_rank, modality_map)
        modality_map: {q_id: {quote_id: 'text'|'image'}}
    """
    all_queries = [s["question"] for s in data]
    all_cands = [build_candidates(s) for s in data]
    cand_texts = []
    cand_bounds = [0]
    for cands in all_cands:
        cand_texts.extend([c["text_for_retrieval"] for c in cands])
        cand_bounds.append(len(cand_texts))

    print(f"  Encoding {len(all_queries)} queries...")
    q_dense, q_lex, q_col = encode_with_bgem3(model, all_queries, batch_size=64)
    print(f"  Encoding {len(cand_texts)} candidates...")
    c_dense, c_lex, c_col = encode_with_bgem3(model, cand_texts, batch_size=64)

    dense_rank: Dict[int, List[str]] = {}
    lex_rank: Dict[int, List[str]] = {}
    col_rank: Dict[int, List[str]] = {}
    modality_map: Dict[int, Dict[str, str]] = {}

    for i, sample in enumerate(tqdm(data, desc="Per-question scoring")):
        q_id = sample["q_id"]
        cands = all_cands[i]
        if not cands:
            dense_rank[q_id] = []
            lex_rank[q_id] = []
            col_rank[q_id] = []
            modality_map[q_id] = {}
            continue

        s, e = cand_bounds[i], cand_bounds[i + 1]
        cd, cl, cc = c_dense[s:e], c_lex[s:e], c_col[s:e]

        # Dense
        d_scores = cosine_sim(q_dense[i], cd)

        # Lexical (sparse)
        l_scores = np.array([
            model.compute_lexical_matching_score(q_lex[i], cl_j) for cl_j in cl
        ])

        # ColBERT
        c_scores = np.array([
            float(model.colbert_score(q_col[i], cc_j)) for cc_j in cc
        ])

        # 三路各自排序
        cand_ids = [c["quote_id"] for c in cands]

        def to_rank(scores):
            idx = np.argsort(-scores)
            return [cand_ids[j] for j in idx[:top_k]]

        dense_rank[q_id] = to_rank(d_scores)
        lex_rank[q_id] = to_rank(l_scores)
        col_rank[q_id] = to_rank(c_scores)
        modality_map[q_id] = {c["quote_id"]: c["modality"] for c in cands}

    return dense_rank, lex_rank, col_rank, modality_map


def rrf_with_image_boost(
    rankings_list: List[Dict[int, List[str]]],
    modality_map: Dict[int, Dict[str, str]],
    k: int = 60,
    weights: List[float] = None,
    image_boost: float = 0.0,
    top_k: int = 100,
) -> Dict[int, List[str]]:
    """
    RRF + 對 image candidate 在最終分數加 boost。
    """
    if weights is None:
        weights = [1.0] * len(rankings_list)

    all_qids = set()
    for r in rankings_list:
        all_qids.update(r.keys())

    fused = {}
    for q_id in all_qids:
        scores: Dict[str, float] = {}
        for w, rankings in zip(weights, rankings_list):
            for rank, qid in enumerate(rankings.get(q_id, []), start=1):
                scores[qid] = scores.get(qid, 0.0) + w / (k + rank)

        if image_boost != 0.0:
            mod = modality_map.get(q_id, {})
            for qid in scores:
                if mod.get(qid) == "image":
                    scores[qid] += image_boost

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused[q_id] = [qid for qid, _ in sorted_items[:top_k]]
    return fused


def modality_split_stats(preds: Dict[int, List[str]], modality_map):
    """統計 top-5 中 text / image 各占多少"""
    t, im = 0, 0
    for q_id, lst in preds.items():
        mod = modality_map.get(q_id, {})
        for qid in lst[:5]:
            if mod.get(qid) == "image":
                im += 1
            else:
                t += 1
    total = max(t + im, 1)
    return {"text": t, "image": im, "text_pct": 100 * t / total, "image_pct": 100 * im / total}


def main():
    # ── 載入資料 ──────────────────────────────────────
    train_data = load_train()
    test_data = load_test()
    train_subset, dev_subset = split_train_dev(train_data)
    print(f"📊 Train: {len(train_subset)}, Dev: {len(dev_subset)}, Test: {len(test_data)}")

    # ── 載入 BM25 既有 rankings ────────────────────────
    bm25_test_path = find_latest_rankings("phase_1", "rankings_top100.json")
    bm25_dev_path = find_latest_rankings("phase_1", "dev_rankings_top100.json")
    if not bm25_test_path or not bm25_dev_path:
        print("❌ 需要 Phase 1 (BM25) 的 rankings")
        return
    bm25_test = load_rankings(bm25_test_path)
    bm25_dev = load_rankings(bm25_dev_path)
    print(f"📂 BM25 rankings: {bm25_test_path}")

    # ── 載入 BGE-M3 ───────────────────────────────────
    from FlagEmbedding import BGEM3FlagModel
    print("📦 Loading BAAI/bge-m3 (dense + sparse + colbert)...")
    bgem3 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    output_dir = config.get_output_dir("phase_3b", "bgem3_multi_imgboost")

    # ── DEV ──────────────────────────────────────────
    print("\n🔍 BGE-M3 三路檢索 on dev set...")
    t0 = time.time()
    d_dev, l_dev, c_dev, mod_dev = run_bgem3_retrieval(bgem3, dev_subset, top_k=100)
    print(f"  Dev encode+score: {time.time()-t0:.1f}s")

    dev_golds = get_gold_quotes_dict(dev_subset)

    # 比較不同設定的 dev recall
    rs4 = [bm25_dev, d_dev, l_dev, c_dev]
    w_eq = [1.0, 1.0, 1.0, 1.0]
    settings = [
        ("BM25 only", [bm25_dev], [1.0], 0.0),
        ("BM25+Dense (Phase 3 baseline)", [bm25_dev, d_dev], [1.0, 1.0], 0.0),
        ("BM25+Dense+Lex+ColBERT", rs4, w_eq, 0.0),
        ("4-way + boost=0.001", rs4, w_eq, 0.001),
        ("4-way + boost=0.002", rs4, w_eq, 0.002),
        ("4-way + boost=0.003", rs4, w_eq, 0.003),
        ("4-way + boost=0.004", rs4, w_eq, 0.004),
        ("4-way + boost=0.005", rs4, w_eq, 0.005),
        ("4-way + boost=0.006", rs4, w_eq, 0.006),
        ("4-way + boost=0.007", rs4, w_eq, 0.007),
        ("4-way + boost=0.008", rs4, w_eq, 0.008),
        # 加重 ColBERT
        ("colbert*1.3 + boost=0.004", rs4, [1.0, 1.0, 1.0, 1.3], 0.004),
        ("colbert*1.3 + boost=0.005", rs4, [1.0, 1.0, 1.0, 1.3], 0.005),
        ("colbert*1.5 + boost=0.005", rs4, [1.0, 1.0, 1.0, 1.5], 0.005),
        # 降低 BM25 (BM25 大量重複 text-favoring)
        ("BM25*0.7 + boost=0.005", rs4, [0.7, 1.0, 1.0, 1.0], 0.005),
        ("BM25*0.5 + boost=0.005", rs4, [0.5, 1.0, 1.0, 1.0], 0.005),
    ]

    print("\n📊 Dev sweep:")
    best_recall = -1.0
    best_setting = None
    best_preds = None
    sweep_results = []
    for name, rs, ws, boost in settings:
        fused = rrf_with_image_boost(rs, mod_dev, k=RRF_K, weights=ws,
                                     image_boost=boost, top_k=100)
        preds5 = {q: v[:5] for q, v in fused.items()}
        r = recall_at_k(preds5, dev_golds, k=5)
        stats = modality_split_stats(preds5, mod_dev)
        sweep_results.append({
            "name": name, "weights": ws, "image_boost": boost,
            "dev_recall_at_5": r,
            "text_pct": stats["text_pct"], "image_pct": stats["image_pct"],
        })
        marker = ""
        if r > best_recall:
            best_recall = r
            best_setting = (name, rs, ws, boost)
            best_preds = preds5
            marker = " ← best"
        print(f"  {name:<45s}  recall={r:.4f}  text/img={stats['text_pct']:.1f}/{stats['image_pct']:.1f}%{marker}")

    print(f"\n📈 Best Dev Recall@5 = {best_recall:.4f}  setting={best_setting[0]}")

    # ── TEST (用 best setting) ────────────────────────
    print("\n🔍 BGE-M3 三路檢索 on test set...")
    t0 = time.time()
    d_test, l_test, c_test, mod_test = run_bgem3_retrieval(bgem3, test_data, top_k=100)
    print(f"  Test encode+score: {time.time()-t0:.1f}s")

    _, _, best_ws, best_boost = best_setting
    test_fused = rrf_with_image_boost(
        [bm25_test, d_test, l_test, c_test],
        mod_test,
        k=RRF_K,
        weights=best_ws,
        image_boost=best_boost,
        top_k=100,
    )
    test_preds_top5 = {q: v[:5] for q, v in test_fused.items()}

    # ── 儲存 ──────────────────────────────────────────
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds_top5, test_data, submission_path)

    rankings_path = os.path.join(output_dir, "rankings_top100.json")
    save_rankings(test_fused, rankings_path)

    # 也存 dev rankings 供 Phase 4 reranker 使用
    dev_fused = rrf_with_image_boost(
        [bm25_dev, d_dev, l_dev, c_dev],
        mod_dev,
        k=RRF_K, weights=best_ws, image_boost=best_boost, top_k=100,
    )
    dev_rankings_path = os.path.join(output_dir, "dev_rankings_top100.json")
    save_rankings(dev_fused, dev_rankings_path)

    ea = error_analysis(best_preds, dev_golds, k=5)
    metrics = {
        "phase": "phase_3b",
        "model": "BGE-M3 multi-vector + BM25, 4-way RRF + image boost",
        "rrf_k": RRF_K,
        "best_setting": best_setting[0],
        "best_weights": best_ws,
        "best_image_boost": best_boost,
        "dev_recall_at_5": best_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "sweep_results": sweep_results,
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)

    append_leaderboard(
        "Phase 3b",
        "BGE-M3(dense+sparse+colbert)+BM25 RRF+img_boost",
        best_recall,
        note=f"{best_setting[0]}",
    )

    print(f"\n✅ Phase 3b 完成! 輸出目錄: {output_dir}")
    print(f"📤 Submission: {submission_path}")


if __name__ == "__main__":
    main()
