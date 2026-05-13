"""
INLP HW3 - Phase 4b: 強 Reranker + Score Fusion (移除 HyDE)

改進方向（vs Phase 4 原版）:
  1. 移除 HyDE — 用 Qwen2.5-7B 對短問題生成 hypothetical answer 在 dev 上 -13 pt
  2. 換更強 reranker (BAAI/bge-reranker-v2-gemma, 2B decoder reranker)
     用 transformers 直接調用 (FlagEmbedding 與當前 transformers 版本不相容)
  3. 重排「全部候選」而非 Top-100 — 每題其實只有 ~15 候選
  4. Score Fusion:
       final = z(rerank_score) + alpha * z(phase3b_rrf_score) + image_boost(image)
     不是用 reranker 完全取代 RRF
"""
import json
import os
import time
from typing import Dict, List

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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


RERANK_MODEL = "BAAI/bge-reranker-v2-gemma"
RERANK_MAX_LEN = 512
RERANK_BATCH = 16
RERANK_PROMPT = (
    'Given a query A and a passage B, determine whether the passage contains an answer '
    'to the query by providing a prediction of either "Yes" or "No".'
)


# ══════════════════════════════════════════════════════
#  Reranker (transformers raw)
# ══════════════════════════════════════════════════════

def load_reranker():
    print(f"📦 Loading reranker: {RERANK_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        RERANK_MODEL, dtype=torch.float16
    ).to("cuda")
    model.eval()
    yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    return model, tokenizer, yes_id


def build_inputs(pairs, tokenizer, max_length=512):
    """按 bge-reranker-v2-gemma 官方範例組裝輸入 (A:/B:/prompt)"""
    sep = "\n"
    prompt_ids = tokenizer(RERANK_PROMPT, return_tensors=None,
                           add_special_tokens=False)["input_ids"]
    sep_ids = tokenizer(sep, return_tensors=None,
                        add_special_tokens=False)["input_ids"]
    items = []
    qmax = max_length * 3 // 4
    for q, p in pairs:
        q_ids = tokenizer(f"A: {q}", return_tensors=None,
                          add_special_tokens=False,
                          max_length=qmax, truncation=True)["input_ids"]
        p_ids = tokenizer(f"B: {p}", return_tensors=None,
                          add_special_tokens=False,
                          max_length=max_length, truncation=True)["input_ids"]
        ids = q_ids + sep_ids + p_ids + sep_ids + prompt_ids
        items.append({"input_ids": ids,
                      "attention_mask": [1] * len(ids)})
    return tokenizer.pad(items, padding=True, return_tensors="pt")


@torch.no_grad()
def score_pairs(pairs, model, tokenizer, yes_id, batch=RERANK_BATCH):
    """批次計算 reranker 分數 (yes-token logit)"""
    scores = np.zeros(len(pairs), dtype=np.float32)
    for i in tqdm(range(0, len(pairs), batch), desc="rerank"):
        chunk = pairs[i:i + batch]
        inputs = build_inputs(chunk, tokenizer, max_length=RERANK_MAX_LEN)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model(**inputs)
        logits = out.logits[:, -1, yes_id].float().cpu().numpy()
        scores[i:i + len(chunk)] = logits
    return scores


def rerank_all_candidates(
    data: List[dict],
    model, tokenizer, yes_id,
) -> Dict[int, Dict[str, float]]:
    pairs_global = []
    meta = []
    for sample in data:
        q_id = sample["q_id"]
        cands = build_candidates(sample)
        for c in cands:
            text = (c["text_for_retrieval"] or "")[:1200]
            if not text:
                continue
            pairs_global.append([sample["question"], text])
            meta.append((q_id, c["quote_id"]))

    print(f"  Rerank {len(pairs_global)} (q,cand) pairs...")
    t0 = time.time()
    scores = score_pairs(pairs_global, model, tokenizer, yes_id, batch=RERANK_BATCH)
    print(f"  Rerank wall: {time.time()-t0:.1f}s")

    out = {}
    for (q_id, qid), s in zip(meta, scores):
        out.setdefault(q_id, {})[qid] = float(s)
    return out


# ══════════════════════════════════════════════════════
#  Fusion
# ══════════════════════════════════════════════════════

def rrf_rank_to_score(rrf_rankings: Dict[int, List[str]], k: int = 60):
    out = {}
    for q_id, lst in rrf_rankings.items():
        out[q_id] = {}
        for rank, qid in enumerate(lst, start=1):
            out[q_id][qid] = 1.0 / (k + rank)
    return out


def zscore(values):
    v = np.array(values, dtype=np.float64)
    sd = v.std() + 1e-10
    return (v - v.mean()) / sd


def fuse_scores(
    rerank_scores, rrf_scores, modality,
    alpha=0.5, image_boost=0.0, top_k=5,
):
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
        sorted_idx = np.argsort(-final)
        out[q_id] = [ids[j] for j in sorted_idx[:top_k]]
    return out


def build_modality_map(data):
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


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════

def main():
    train_data = load_train()
    test_data = load_test()
    train_subset, dev_subset = split_train_dev(train_data)
    print(f"📊 Train: {len(train_subset)}, Dev: {len(dev_subset)}, Test: {len(test_data)}")

    # RRF prior from Phase 3b
    rrf_test_path = find_latest_rankings("phase_3b", "rankings_top100.json")
    rrf_dev_path = find_latest_rankings("phase_3b", "dev_rankings_top100.json")
    if not rrf_test_path:
        print("⚠️ Phase 3b 不存在，fallback Phase 3")
        rrf_test_path = find_latest_rankings("phase_3", "rankings_top100.json")
        rrf_dev_path = find_latest_rankings("phase_3", "dev_rankings_top100.json")
    print(f"📂 RRF rankings: {rrf_test_path}")

    rrf_test_rank = load_rankings(rrf_test_path)
    rrf_dev_rank = load_rankings(rrf_dev_path)
    rrf_test_score = rrf_rank_to_score(rrf_test_rank)
    rrf_dev_score = rrf_rank_to_score(rrf_dev_rank)

    mod_dev = build_modality_map(dev_subset)
    mod_test = build_modality_map(test_data)

    output_dir = config.get_output_dir("phase_4b", "rerank_v2gemma_fusion")

    model, tokenizer, yes_id = load_reranker()

    # ── Dev ────────────────────────────────────────────
    print("\n🔍 Rerank on dev set (全部候選)...")
    dev_rerank = rerank_all_candidates(dev_subset, model, tokenizer, yes_id)

    dev_golds = get_gold_quotes_dict(dev_subset)

    settings = [
        ("rerank only", 0.0, 0.0),
        ("rerank + boost=0.30", 0.0, 0.30),
        ("rerank + boost=0.50", 0.0, 0.50),
        ("rerank + 0.3*RRF", 0.3, 0.0),
        ("rerank + 0.5*RRF", 0.5, 0.0),
        ("rerank + 1.0*RRF", 1.0, 0.0),
        ("rerank + 0.5*RRF + boost=0.20", 0.5, 0.20),
        ("rerank + 0.5*RRF + boost=0.30", 0.5, 0.30),
        ("rerank + 0.5*RRF + boost=0.40", 0.5, 0.40),
        ("rerank + 1.0*RRF + boost=0.30", 1.0, 0.30),
        ("rerank + 1.0*RRF + boost=0.40", 1.0, 0.40),
        ("rerank + 2.0*RRF + boost=0.30", 2.0, 0.30),
    ]

    print("\n📊 Dev sweep:")
    best_recall = -1.0
    best_setting = None
    best_preds = None
    sweep_results = []
    for name, alpha, boost in settings:
        preds = fuse_scores(dev_rerank, rrf_dev_score, mod_dev,
                            alpha=alpha, image_boost=boost, top_k=5)
        r = recall_at_k(preds, dev_golds, k=5)
        stats = modality_split_stats(preds, mod_dev)
        sweep_results.append({
            "name": name, "alpha": alpha, "image_boost_z": boost,
            "dev_recall_at_5": r,
            "text_pct": stats["text_pct"], "image_pct": stats["image_pct"],
        })
        marker = ""
        if r > best_recall:
            best_recall = r
            best_setting = (name, alpha, boost)
            best_preds = preds
            marker = " ← best"
        print(f"  {name:<45s}  recall={r:.4f}  text/img={stats['text_pct']:.1f}/{stats['image_pct']:.1f}%{marker}")

    print(f"\n📈 Best Dev Recall@5 = {best_recall:.4f}  setting={best_setting[0]}")

    # ── Test ───────────────────────────────────────────
    print("\n🔍 Rerank on test set (全部候選)...")
    test_rerank = rerank_all_candidates(test_data, model, tokenizer, yes_id)

    _, best_alpha, best_boost = best_setting
    test_preds = fuse_scores(test_rerank, rrf_test_score, mod_test,
                             alpha=best_alpha, image_boost=best_boost, top_k=5)

    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds, test_data, submission_path)

    # 存原始 reranker 分數 (調參用)
    with open(os.path.join(output_dir, "rerank_scores_dev.json"), "w") as f:
        json.dump({str(k): v for k, v in dev_rerank.items()}, f)
    with open(os.path.join(output_dir, "rerank_scores_test.json"), "w") as f:
        json.dump({str(k): v for k, v in test_rerank.items()}, f)

    ea = error_analysis(best_preds, dev_golds, k=5)
    metrics = {
        "phase": "phase_4b",
        "reranker_model": RERANK_MODEL,
        "rrf_prior_source": rrf_test_path,
        "best_setting": best_setting[0],
        "best_alpha": best_alpha,
        "best_image_boost_z": best_boost,
        "dev_recall_at_5": best_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "sweep_results": sweep_results,
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)

    append_leaderboard(
        "Phase 4b",
        f"{RERANK_MODEL} + Phase3b RRF fusion + img_boost",
        best_recall,
        note=best_setting[0],
    )

    print(f"\n✅ Phase 4b 完成! 輸出目錄: {output_dir}")
    print(f"📤 Submission: {submission_path}")


if __name__ == "__main__":
    main()
