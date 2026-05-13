"""
INLP HW3 - Phase 5b: 多 submission ensemble via RRF

對手上 LB ≥ 0.72342 的 4 份 ranking 做 RRF 融合：
  1. Phase 3b RRF (4-way + boost=0.004)
  2. Phase 3c RRF (4-way + VLM-concat + boost=0.003)
  3. Phase 4b 最終排名 (rerank + 2.0*RRF(3b) + boost=0.30)
  4. Phase 4c 最終排名 (rerank + 3.0*RRF(3c) + boost=0.40)
"""
import json
import os
from typing import Dict, List

import numpy as np

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict,
)
from evaluation import recall_at_k, error_analysis
from submission_utils import (
    generate_submission, save_metrics, append_leaderboard, load_rankings,
)
from run_phase3_hybrid import find_latest_rankings


P4B_DIR = "/home/louis/NLP-hw3/outputs/phase_4b/rerank_v2gemma_fusion_05122231"
P4C_DIR = "/home/louis/NLP-hw3/outputs/phase_4c/fuse_prior3c_05130859"
STRAT_DEV_PATH = os.path.join(config.SPLITS_DIR, "stratified_dev_ids.json")


def load_rerank_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def rrf_rank_to_score(rrf_rankings, k=60):
    out = {}
    for q_id, lst in rrf_rankings.items():
        out[q_id] = {qid: 1.0 / (k + r) for r, qid in enumerate(lst, start=1)}
    return out


def zscore(values):
    v = np.array(values, dtype=np.float64)
    sd = v.std() + 1e-10
    return (v - v.mean()) / sd


def fuse_to_ranking(rerank_scores, rrf_scores, modality, alpha, image_boost, top_k=100):
    """完全照 Phase 4b/4c 的融合邏輯，回傳 top-100 ranking"""
    out = {}
    for q_id, sd in rerank_scores.items():
        ids = list(sd.keys())
        if not ids:
            out[q_id] = []
            continue
        rr = [sd[i] for i in ids]
        rf = [rrf_scores.get(q_id, {}).get(i, 0.0) for i in ids]
        z_rr = zscore(rr)
        z_rf = zscore(rf) if any(x > 0 for x in rf) else np.zeros(len(rf))
        final = z_rr + alpha * z_rf
        if image_boost != 0.0:
            mod = modality.get(q_id, {})
            for j, qid in enumerate(ids):
                if mod.get(qid) == "image":
                    final[j] += image_boost
        idx = np.argsort(-final)
        out[q_id] = [ids[j] for j in idx[:top_k]]
    return out


def build_modality(data):
    out = {}
    for s in data:
        cands = build_candidates(s)
        out[s["q_id"]] = {c["quote_id"]: c["modality"] for c in cands}
    return out


def rrf_ensemble(rankings_list: List[Dict[int, List[str]]], weights, modality,
                 image_boost=0.0, k=60, top_k=5):
    """RRF 融合多個 rankings + 可選 image boost"""
    all_qids = set()
    for r in rankings_list:
        all_qids.update(r.keys())
    out = {}
    for q_id in all_qids:
        scores = {}
        for w, r in zip(weights, rankings_list):
            for rank, qid in enumerate(r.get(q_id, []), start=1):
                scores[qid] = scores.get(qid, 0.0) + w / (k + rank)
        if image_boost != 0.0:
            mod = modality.get(q_id, {})
            for qid in scores:
                if mod.get(qid) == "image":
                    scores[qid] += image_boost
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out[q_id] = [qid for qid, _ in sorted_items[:top_k]]
    return out


def modality_split_stats(preds, modality):
    t, im = 0, 0
    for q_id, lst in preds.items():
        mod = modality.get(q_id, {})
        for qid in lst[:5]:
            if mod.get(qid) == "image":
                im += 1
            else:
                t += 1
    total = max(t + im, 1)
    return {"text_pct": 100 * t / total, "image_pct": 100 * im / total}


def main():
    train_data = load_train()
    test_data = load_test()
    _, dev_subset = split_train_dev(train_data)
    random_dev_ids = set(s["q_id"] for s in dev_subset)
    with open(STRAT_DEV_PATH) as f:
        strat_dev_ids = set(json.load(f))

    train_golds = get_gold_quotes_dict(train_data)
    mod_train = build_modality(train_data)
    mod_test = build_modality(test_data)

    # ── 載入 Phase 3b/3c 的 dev/test rankings ────────
    p3b_test = load_rankings(find_latest_rankings("phase_3b", "rankings_top100.json"))
    p3b_dev = load_rankings(find_latest_rankings("phase_3b", "dev_rankings_top100.json"))
    p3c_test = load_rankings(find_latest_rankings("phase_3c", "rankings_top100.json"))
    p3c_dev = load_rankings(find_latest_rankings("phase_3c", "dev_rankings_top100.json"))
    print(f"📂 P3b test={len(p3b_test)} dev={len(p3b_dev)}")
    print(f"📂 P3c test={len(p3c_test)} dev={len(p3c_dev)}")

    # ── 重建 Phase 4b/4c 的 final ranking (用各自的 LB-best 設定) ──
    rerank_dev = load_rerank_scores(os.path.join(P4B_DIR, "rerank_scores_dev.json"))
    rerank_test = load_rerank_scores(os.path.join(P4B_DIR, "rerank_scores_test.json"))

    p4b_test = fuse_to_ranking(rerank_test, rrf_rank_to_score(p3b_test), mod_test,
                               alpha=2.0, image_boost=0.30, top_k=100)
    p4b_dev = fuse_to_ranking(rerank_dev, rrf_rank_to_score(p3b_dev),
                              {q: mod_train[q] for q in p3b_dev},
                              alpha=2.0, image_boost=0.30, top_k=100)
    p4c_test = fuse_to_ranking(rerank_test, rrf_rank_to_score(p3c_test), mod_test,
                               alpha=3.0, image_boost=0.40, top_k=100)
    p4c_dev = fuse_to_ranking(rerank_dev, rrf_rank_to_score(p3c_dev),
                              {q: mod_train[q] for q in p3c_dev},
                              alpha=3.0, image_boost=0.40, top_k=100)

    # ── Sweep ensemble ───────────────────────────────
    # 4 ranking sources
    dev_rankings = {"3b": p3b_dev, "3c": p3c_dev, "4b": p4b_dev, "4c": p4c_dev}
    test_rankings = {"3b": p3b_test, "3c": p3c_test, "4b": p4b_test, "4c": p4c_test}

    # 嘗試幾種權重組合
    weight_configs = [
        ("uniform 4-way",       [1.0, 1.0, 1.0, 1.0]),
        ("uniform 3-way (3b+3c+4b)", [1.0, 1.0, 1.0, 0.0]),
        ("uniform 3-way (3b+4b+4c)", [1.0, 0.0, 1.0, 1.0]),
        ("4b emphasis",         [1.0, 1.0, 2.0, 1.0]),
        ("4b emphasis 2",       [0.5, 0.5, 2.0, 0.5]),
        ("3b+4b only",          [1.0, 0.0, 1.0, 0.0]),
        ("3c+4b only",          [0.0, 1.0, 1.0, 0.0]),
        ("4b alone (sanity)",   [0.0, 0.0, 1.0, 0.0]),
    ]
    boost_grid = [0.0, 0.001, 0.002, 0.003]

    print("\n📊 Ensemble dev sweep:")
    print(f"  {'weights':<32s}  {'boost':>6s}  {'random':>7s} {'strat':>7s}  t/i%")
    rows = []
    best_combined = -1.0
    best_cfg = None
    for wname, ws in weight_configs:
        for boost in boost_grid:
            preds_full = rrf_ensemble(list(dev_rankings.values()), ws, mod_train,
                                      image_boost=boost, top_k=5)
            preds_rand = {q: preds_full[q] for q in random_dev_ids if q in preds_full}
            preds_strat = {q: preds_full[q] for q in strat_dev_ids if q in preds_full}
            golds_rand = {q: train_golds[q] for q in random_dev_ids if q in train_golds}
            golds_strat = {q: train_golds[q] for q in strat_dev_ids if q in train_golds}
            r_rand = recall_at_k(preds_rand, golds_rand, k=5)
            r_strat = recall_at_k(preds_strat, golds_strat, k=5)
            stats = modality_split_stats(preds_full, mod_train)
            rows.append({"weights": wname, "ws": ws, "boost": boost,
                         "random_dev": r_rand, "strat_dev": r_strat,
                         "image_pct": stats["image_pct"]})
            # 用 average 當挑選依據（兩種 dev 平均）
            combined = (r_rand + r_strat) / 2
            marker = ""
            if combined > best_combined:
                best_combined = combined
                best_cfg = (wname, ws, boost)
                marker = " ← best avg"
            print(f"  {wname:<32s}  {boost:.3f}  {r_rand:.4f}  {r_strat:.4f}  {stats['image_pct']:.1f}%{marker}")

    wname, ws, boost = best_cfg
    print(f"\n📈 Best (avg of random+strat): {wname}, boost={boost}")

    # ── Test prediction with best config ─────────────
    test_preds = rrf_ensemble(list(test_rankings.values()), ws, mod_test,
                              image_boost=boost, top_k=5)

    output_dir = config.get_output_dir("phase_5b", "ensemble")
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds, test_data, submission_path)

    metrics = {
        "phase": "phase_5b",
        "best_weights": ws,
        "best_weight_name": wname,
        "best_image_boost": boost,
        "best_random_dev": rows[-1]["random_dev"],  # last row
        "best_strat_dev": rows[-1]["strat_dev"],
        "all_sweep": [{"weight_name": r["weights"], "weights": r["ws"], "boost": r["boost"],
                       "random_dev_recall": r["random_dev"], "strat_dev_recall": r["strat_dev"]}
                      for r in rows],
    }
    save_metrics(metrics, output_dir)

    append_leaderboard(
        "Phase 5b",
        f"Ensemble RRF of Phase 3b/3c/4b/4c rankings",
        max(r["strat_dev"] for r in rows),  # 用 strat 當主指標
        note=f"{wname}, boost={boost}",
    )

    print(f"\n✅ Phase 5b 完成! 輸出目錄: {output_dir}")
    print(f"📤 Submission: {submission_path}")


if __name__ == "__main__":
    main()
