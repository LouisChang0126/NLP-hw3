"""
INLP HW3 - Phase 4c: 用 Phase 4b 已存的 reranker scores + Phase 3c 的 RRF rankings 重新融合
                     (避免重跑 7-min 的 reranker)

預期：Phase 3c RRF 的 image=58% 比 Phase 3b 的 62% 更貼近 gold 60%；
      若 reranker 加上更貼近真相的 prior，LB 可能再上分。
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
    generate_submission, save_metrics, append_leaderboard,
    load_rankings,
)
from run_phase3_hybrid import find_latest_rankings


RERANK_DIR = "/home/louis/NLP-hw3/outputs/phase_4b/rerank_v2gemma_fusion_05122231"


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


def fuse(rerank_scores, rrf_scores, modality, alpha=2.0, image_boost=0.0, top_k=5):
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
    # ── data ──────────────────────────────────────────
    train_data = load_train()
    test_data = load_test()
    _, dev_subset = split_train_dev(train_data)

    # ── 載入 rerank scores ───────────────────────────
    rerank_dev = load_rerank_scores(os.path.join(RERANK_DIR, "rerank_scores_dev.json"))
    rerank_test = load_rerank_scores(os.path.join(RERANK_DIR, "rerank_scores_test.json"))
    print(f"📂 Rerank scores loaded: dev={len(rerank_dev)}, test={len(rerank_test)}")

    # ── 載入兩種 RRF prior: Phase 3b & Phase 3c ──────
    p3b_test_path = find_latest_rankings("phase_3b", "rankings_top100.json")
    p3b_dev_path = find_latest_rankings("phase_3b", "dev_rankings_top100.json")
    p3c_test_path = find_latest_rankings("phase_3c", "rankings_top100.json")
    p3c_dev_path = find_latest_rankings("phase_3c", "dev_rankings_top100.json")
    print(f"📂 Phase 3b prior: {p3b_test_path}")
    print(f"📂 Phase 3c prior: {p3c_test_path}")

    p3b_test_rrf = rrf_rank_to_score(load_rankings(p3b_test_path))
    p3b_dev_rrf = rrf_rank_to_score(load_rankings(p3b_dev_path))
    p3c_test_rrf = rrf_rank_to_score(load_rankings(p3c_test_path))
    p3c_dev_rrf = rrf_rank_to_score(load_rankings(p3c_dev_path))

    mod_dev = build_modality(dev_subset)
    mod_test = build_modality(test_data)

    dev_golds = get_gold_quotes_dict(dev_subset)

    # ── Sweep on dev ─────────────────────────────────
    settings = []
    for prior_name, dev_rrf in [("3b", p3b_dev_rrf), ("3c", p3c_dev_rrf)]:
        for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for boost in [0.0, 0.20, 0.30, 0.40, 0.50]:
                settings.append((prior_name, alpha, boost, dev_rrf))

    print(f"\n📊 Sweeping {len(settings)} configs on dev...")
    best_recall = -1.0
    best_setting = None
    sweep = []
    for prior, alpha, boost, dev_rrf in settings:
        preds = fuse(rerank_dev, dev_rrf, mod_dev, alpha=alpha, image_boost=boost, top_k=5)
        r = recall_at_k(preds, dev_golds, k=5)
        stats = modality_split_stats(preds, mod_dev)
        sweep.append({"prior": prior, "alpha": alpha, "boost": boost,
                      "dev_recall_at_5": r,
                      "text_pct": stats["text_pct"], "image_pct": stats["image_pct"]})
        if r > best_recall:
            best_recall = r
            best_setting = (prior, alpha, boost)

    # 印出 top-15 setting
    sweep.sort(key=lambda x: -x["dev_recall_at_5"])
    print("\nTop-15 dev settings:")
    for row in sweep[:15]:
        print(f"  prior={row['prior']}  alpha={row['alpha']:.1f}  boost={row['boost']:.2f}  "
              f"recall={row['dev_recall_at_5']:.4f}  text/img={row['text_pct']:.1f}/{row['image_pct']:.1f}%")

    print(f"\n📈 Best Dev Recall@5 = {best_recall:.4f}")
    print(f"   prior={best_setting[0]}  alpha={best_setting[1]}  boost={best_setting[2]}")

    # ── Test prediction ──────────────────────────────
    best_prior_name, best_alpha, best_boost = best_setting
    test_rrf = p3c_test_rrf if best_prior_name == "3c" else p3b_test_rrf
    test_preds = fuse(rerank_test, test_rrf, mod_test,
                      alpha=best_alpha, image_boost=best_boost, top_k=5)

    output_dir = config.get_output_dir("phase_4c", f"fuse_prior{best_prior_name}")
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds, test_data, submission_path)

    # ── dev best preds for error analysis ───────────
    best_dev_preds = fuse(rerank_dev, p3c_dev_rrf if best_prior_name == "3c" else p3b_dev_rrf,
                          mod_dev, alpha=best_alpha, image_boost=best_boost, top_k=5)
    ea = error_analysis(best_dev_preds, dev_golds, k=5)

    metrics = {
        "phase": "phase_4c",
        "reranker_source": RERANK_DIR,
        "best_prior": best_prior_name,
        "best_alpha": best_alpha,
        "best_image_boost_z": best_boost,
        "dev_recall_at_5": best_recall,
        "sweep_top": sweep[:30],
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)

    append_leaderboard(
        "Phase 4c",
        f"bge-reranker-v2-gemma + Phase 3{best_prior_name} RRF fusion",
        best_recall,
        note=f"prior=3{best_prior_name}, alpha={best_alpha}, boost={best_boost}",
    )

    print(f"\n✅ Phase 4c 完成! 輸出目錄: {output_dir}")
    print(f"📤 Submission: {submission_path}")


if __name__ == "__main__":
    main()
