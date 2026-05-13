"""
INLP HW3 - Phase 3d: Stratified Dev + 完整 boost sweep (不使用 VLM caption)

目的:
  1. 隔離 P2 (BGE-M3 sparse + colbert) 的真實邊際貢獻
  2. 建立 stratified dev (按 test domain 比例) — 比 random 200 更能預測 LB
  3. 比較 random-dev 與 stratified-dev 找出 best config 的差異

無 VLM caption — 用原 img_description (Phase 3c LB 證實 VLM concat 無增益)
"""
import json
import os
import random
import time
from typing import Dict, List

from tqdm import tqdm
import numpy as np

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict, load_jsonl,
)
from evaluation import recall_at_k, error_analysis
from submission_utils import (
    generate_submission, save_metrics, append_leaderboard,
    save_rankings, load_rankings,
)
from run_phase3_hybrid import find_latest_rankings
from run_phase3b_bgem3_multi import (
    encode_with_bgem3, cosine_sim, rrf_with_image_boost,
    modality_split_stats,
)


STRAT_DEV_IDS_PATH = os.path.join(config.SPLITS_DIR, "stratified_dev_ids.json")


def build_stratified_dev_ids(train_data, test_data, target_size=200, seed=42):
    """
    從 train 抽 target_size 筆做 stratified dev，比例符合 test 內各 domain 的占比 ——
    僅限 train 也存在的 domain (Financial, Academic, Research, News).
    """
    from collections import Counter
    test_dom = Counter(s["domain"] for s in test_data)
    train_dom = Counter(s["domain"] for s in train_data)
    shared = [d for d in test_dom if d in train_dom]
    test_total = sum(test_dom[d] for d in shared)
    quotas = {d: max(1, round(target_size * test_dom[d] / test_total)) for d in shared}
    # 調整總數
    diff = target_size - sum(quotas.values())
    # 把 diff 加到最大組
    biggest = max(quotas, key=lambda d: quotas[d])
    quotas[biggest] += diff

    rng = random.Random(seed)
    by_dom: Dict[str, List[dict]] = {d: [] for d in shared}
    for s in train_data:
        if s["domain"] in by_dom:
            by_dom[s["domain"]].append(s)
    dev_ids = []
    for d in shared:
        pool = by_dom[d]
        rng.shuffle(pool)
        dev_ids.extend([s["q_id"] for s in pool[:quotas[d]]])
    return dev_ids, quotas


def bgem3_full_retrieve(model, data):
    """跑 BGE-M3 三路檢索, 回傳 (dense, lex, colbert, modality)"""
    all_queries = [s["question"] for s in data]
    all_cands = [build_candidates(s) for s in data]
    cand_texts = []
    bounds = [0]
    for cands in all_cands:
        cand_texts.extend([c["text_for_retrieval"] for c in cands])
        bounds.append(len(cand_texts))
    print(f"  Encoding {len(all_queries)} queries...")
    q_d, q_l, q_c = encode_with_bgem3(model, all_queries)
    print(f"  Encoding {len(cand_texts)} candidates...")
    c_d, c_l, c_c = encode_with_bgem3(model, cand_texts)
    dense, lex, col, mod = {}, {}, {}, {}
    for i, sample in enumerate(tqdm(data, desc="scoring")):
        q_id = sample["q_id"]
        cands = all_cands[i]
        if not cands:
            dense[q_id] = []
            lex[q_id] = []
            col[q_id] = []
            mod[q_id] = {}
            continue
        s, e = bounds[i], bounds[i + 1]
        cd, cl, cc = c_d[s:e], c_l[s:e], c_c[s:e]
        d_s = cosine_sim(q_d[i], cd)
        l_s = np.array([model.compute_lexical_matching_score(q_l[i], j) for j in cl])
        c_s = np.array([float(model.colbert_score(q_c[i], j)) for j in cc])
        cand_ids = [c["quote_id"] for c in cands]

        def to_rank(scores):
            idx = np.argsort(-scores)
            return [cand_ids[k] for k in idx[:100]]

        dense[q_id] = to_rank(d_s)
        lex[q_id] = to_rank(l_s)
        col[q_id] = to_rank(c_s)
        mod[q_id] = {c["quote_id"]: c["modality"] for c in cands}
    return dense, lex, col, mod


def eval_on_subset(rankings_list, weights, image_boost, mod_map, subset_ids, golds):
    """在指定 subset_ids 上算 recall@5"""
    fused_full = rrf_with_image_boost(rankings_list, mod_map, k=60, weights=weights,
                                      image_boost=image_boost, top_k=100)
    preds = {q: v[:5] for q, v in fused_full.items() if q in subset_ids}
    sub_golds = {q: g for q, g in golds.items() if q in subset_ids}
    return recall_at_k(preds, sub_golds, k=5), fused_full


def main():
    train_data = load_train()
    test_data = load_test()
    random_train_subset, random_dev_subset = split_train_dev(train_data)
    print(f"📊 Train: {len(train_data)}, Test: {len(test_data)}")

    # ── 建 stratified dev ─────────────────────────────
    strat_dev_ids, quotas = build_stratified_dev_ids(train_data, test_data, target_size=200, seed=42)
    print(f"\n📋 Stratified dev quotas (size={len(strat_dev_ids)}):")
    for d, q in quotas.items():
        print(f"  {d}: {q}")
    os.makedirs(config.SPLITS_DIR, exist_ok=True)
    with open(STRAT_DEV_IDS_PATH, "w") as f:
        json.dump(strat_dev_ids, f)
    print(f"💾 Saved: {STRAT_DEV_IDS_PATH}")

    random_dev_ids = set(s["q_id"] for s in random_dev_subset)
    strat_dev_id_set = set(strat_dev_ids)
    overlap = random_dev_ids & strat_dev_id_set
    print(f"\n  Overlap between random-dev & stratified-dev: {len(overlap)}/{len(strat_dev_id_set)}")

    # ── 載入 BGE-M3 ───────────────────────────────────
    from FlagEmbedding import BGEM3FlagModel
    print("\n📦 Loading BAAI/bge-m3...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    # ── 在 FULL TRAIN 上跑 BGE-M3 三路 ─────────────────
    print(f"\n🔍 BGE-M3 retrieval on full train ({len(train_data)} samples)...")
    t0 = time.time()
    d_train, l_train, c_train, mod_train = bgem3_full_retrieve(model, train_data)
    print(f"  Train encode+score: {time.time()-t0:.1f}s")

    # ── 載入 BM25 train rankings (Phase 1 已經有 dev_subset 的 100, 需要全 train) ──
    # 既有 Phase 1 只算了 dev_subset 的 BM25。要在 full train 上做。
    # 為了不重跑 BM25，直接借用 Phase 1 的 logic
    print("\n🔍 Running BM25 on full train...")
    from run_phase1_bm25 import run_bm25_retrieval
    bm25_train = run_bm25_retrieval(train_data, top_k=100)

    train_golds = get_gold_quotes_dict(train_data)

    # ── Sweep ────────────────────────────────────────
    # Experiment 1: 各路訊號的邊際貢獻 (在 stratified dev 上看)
    # Experiment 2: boost sweep on both devs
    rs_bm25 = [bm25_train]
    rs_bd = [bm25_train, d_train]
    rs_bds = [bm25_train, d_train, l_train]
    rs_4way = [bm25_train, d_train, l_train, c_train]

    settings = []
    # 1. 各路訊號 (boost=0)
    settings.append(("BM25 only",                  rs_bm25, [1.0],            0.0))
    settings.append(("BM25+Dense",                 rs_bd,   [1, 1],           0.0))
    settings.append(("BM25+Dense+Sparse",          rs_bds,  [1, 1, 1],        0.0))
    settings.append(("BM25+Dense+Sparse+ColBERT",  rs_4way, [1, 1, 1, 1],     0.0))
    # 2. boost sweep on 4-way
    for b in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]:
        settings.append((f"4-way boost={b:.3f}", rs_4way, [1, 1, 1, 1], b))
    # 3. boost sweep on BM25+Dense (隔離 multi-vec 的真實貢獻)
    for b in [0.002, 0.004, 0.006]:
        settings.append((f"BM25+Dense boost={b:.3f}", rs_bd, [1, 1], b))

    rows = []
    print("\n📊 Sweeping configs on both devs:")
    print(f"  {'setting':<38s}  {'random':>7s} {'strat':>7s}  text/img(strat)")
    for name, rs, ws, boost in settings:
        r_rand, _ = eval_on_subset(rs, ws, boost, mod_train, random_dev_ids, train_golds)
        r_strat, _ = eval_on_subset(rs, ws, boost, mod_train, strat_dev_id_set, train_golds)
        # modality stats on stratified dev
        fused = rrf_with_image_boost(rs, mod_train, k=60, weights=ws,
                                     image_boost=boost, top_k=100)
        sub = {q: v[:5] for q, v in fused.items() if q in strat_dev_id_set}
        stats = modality_split_stats(sub, mod_train)
        rows.append({
            "name": name, "weights": ws, "image_boost": boost,
            "random_dev_recall": r_rand, "strat_dev_recall": r_strat,
            "strat_text_pct": stats["text_pct"], "strat_image_pct": stats["image_pct"],
        })
        print(f"  {name:<38s}  {r_rand:.4f}  {r_strat:.4f}  {stats['text_pct']:.1f}/{stats['image_pct']:.1f}%")

    # ── 選 stratified dev best 跑 test ───────────────
    best_row = max(rows, key=lambda r: r["strat_dev_recall"])
    print(f"\n📈 Best on stratified dev: {best_row['name']}  recall={best_row['strat_dev_recall']:.4f}")
    print(f"   weights={best_row['weights']}, boost={best_row['image_boost']}")
    best_rs = (rs_bm25 if "BM25 only" in best_row["name"] else
               rs_bd if "BM25+Dense" in best_row["name"] and "Sparse" not in best_row["name"] else
               rs_bds if "BM25+Dense+Sparse" in best_row["name"] and "ColBERT" not in best_row["name"] else
               rs_4way)

    # ── 在 test 上跑相同 config ───────────────────────
    print("\n🔍 BGE-M3 retrieval on test...")
    t0 = time.time()
    d_test, l_test, c_test, mod_test = bgem3_full_retrieve(model, test_data)
    print(f"  Test encode+score: {time.time()-t0:.1f}s")
    print("🔍 Running BM25 on test...")
    bm25_test = run_bm25_retrieval(test_data, top_k=100)

    # 對應 4-way / 3-way / 2-way 等
    if best_rs is rs_bm25:
        test_rs = [bm25_test]
    elif best_rs is rs_bd:
        test_rs = [bm25_test, d_test]
    elif best_rs is rs_bds:
        test_rs = [bm25_test, d_test, l_test]
    else:
        test_rs = [bm25_test, d_test, l_test, c_test]
    test_fused = rrf_with_image_boost(test_rs, mod_test, k=60,
                                      weights=best_row["weights"],
                                      image_boost=best_row["image_boost"], top_k=100)
    test_preds5 = {q: v[:5] for q, v in test_fused.items()}

    output_dir = config.get_output_dir("phase_3d", "stratified_sweep")
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds5, test_data, submission_path)
    save_rankings(test_fused, os.path.join(output_dir, "rankings_top100.json"))

    metrics = {
        "phase": "phase_3d",
        "stratified_dev_quotas": quotas,
        "stratified_dev_ids_count": len(strat_dev_ids),
        "best_setting": best_row["name"],
        "best_weights": best_row["weights"],
        "best_image_boost": best_row["image_boost"],
        "random_dev_recall": best_row["random_dev_recall"],
        "strat_dev_recall": best_row["strat_dev_recall"],
        "all_sweep": rows,
    }
    save_metrics(metrics, output_dir)

    append_leaderboard(
        "Phase 3d",
        "BGE-M3 4-way RRF + img_boost (stratified-dev tuned)",
        best_row["strat_dev_recall"],
        note=f"{best_row['name']} (strat dev only); random_dev={best_row['random_dev_recall']:.4f}",
    )

    print(f"\n✅ Phase 3d 完成! 輸出目錄: {output_dir}")
    print(f"📤 Submission: {submission_path}")


if __name__ == "__main__":
    main()
