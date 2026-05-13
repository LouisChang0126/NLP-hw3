"""
INLP HW3 - Phase 3c: VLM 重生成 caption 取代 img_description,
                     其餘流程同 Phase 3b (4-way RRF + image boost)
"""
import json
import os
import time
from typing import Dict, List

from tqdm import tqdm
import numpy as np

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
from run_phase3b_bgem3_multi import (
    encode_with_bgem3, cosine_sim, rrf_with_image_boost,
    modality_split_stats, RRF_K,
)


VLM_CAPTIONS_FILE = os.path.join(config.DATA_DIR, "img_captions_vlm.json")


def load_vlm_captions() -> Dict[str, str]:
    if not os.path.exists(VLM_CAPTIONS_FILE):
        raise FileNotFoundError(
            f"VLM captions not found: {VLM_CAPTIONS_FILE}\n"
            "Run run_vlm_caption.py first."
        )
    with open(VLM_CAPTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_candidates_vlm(sample: dict, vlm_caps: Dict[str, str], strategy: str = "vlm") -> List[dict]:
    """
    像 build_candidates 但 image text_for_retrieval 由 strategy 決定：
      'vlm':   用 VLM caption；若無 fallback 到 img_description
      'concat': img_description + ' ' + VLM caption
      'orig':  原版 img_description（baseline 對照）
    """
    out = []
    for tq in sample.get("text_quotes", []):
        out.append({
            "quote_id": tq["quote_id"],
            "modality": "text",
            "text_for_retrieval": tq.get("text", ""),
        })
    for iq in sample.get("img_quotes", []):
        path = iq.get("img_path", "")
        orig_desc = iq.get("img_description", "") or ""
        vlm = vlm_caps.get(path, "") or ""
        if strategy == "vlm":
            text = vlm if vlm else orig_desc
        elif strategy == "concat":
            if vlm and orig_desc:
                text = f"{orig_desc} {vlm}"
            else:
                text = vlm or orig_desc
        else:
            text = orig_desc
        out.append({
            "quote_id": iq["quote_id"],
            "modality": "image",
            "text_for_retrieval": text,
            "img_path": path,
        })
    return out


def run_bgem3_retrieval_vlm(model, data, vlm_caps, strategy="vlm"):
    """同 phase 3b 但 candidate text 用 strategy 取得"""
    all_queries = [s["question"] for s in data]
    all_cands = [build_candidates_vlm(s, vlm_caps, strategy) for s in data]
    cand_texts = []
    cand_bounds = [0]
    for cands in all_cands:
        cand_texts.extend([c["text_for_retrieval"] for c in cands])
        cand_bounds.append(len(cand_texts))

    print(f"  Encoding {len(all_queries)} queries...")
    q_dense, q_lex, q_col = encode_with_bgem3(model, all_queries)
    print(f"  Encoding {len(cand_texts)} candidates (strategy={strategy})...")
    c_dense, c_lex, c_col = encode_with_bgem3(model, cand_texts)

    dense_rank, lex_rank, col_rank, mod_map = {}, {}, {}, {}
    for i, sample in enumerate(tqdm(data, desc=f"Score ({strategy})")):
        q_id = sample["q_id"]
        cands = all_cands[i]
        if not cands:
            dense_rank[q_id] = []
            lex_rank[q_id] = []
            col_rank[q_id] = []
            mod_map[q_id] = {}
            continue
        s, e = cand_bounds[i], cand_bounds[i + 1]
        cd, cl, cc = c_dense[s:e], c_lex[s:e], c_col[s:e]
        d_s = cosine_sim(q_dense[i], cd)
        l_s = np.array([model.compute_lexical_matching_score(q_lex[i], j) for j in cl])
        c_s = np.array([float(model.colbert_score(q_col[i], j)) for j in cc])
        cand_ids = [c["quote_id"] for c in cands]

        def to_rank(scores):
            idx = np.argsort(-scores)
            return [cand_ids[k] for k in idx[:100]]

        dense_rank[q_id] = to_rank(d_s)
        lex_rank[q_id] = to_rank(l_s)
        col_rank[q_id] = to_rank(c_s)
        mod_map[q_id] = {c["quote_id"]: c["modality"] for c in cands}
    return dense_rank, lex_rank, col_rank, mod_map


def coverage_stats(data, vlm_caps):
    have, miss = 0, 0
    for s in data:
        for iq in s.get("img_quotes", []):
            p = iq.get("img_path", "")
            if vlm_caps.get(p, ""):
                have += 1
            else:
                miss += 1
    return have, miss


def main():
    print("📂 Loading VLM captions...")
    vlm_caps = load_vlm_captions()
    print(f"  Total cached captions: {len(vlm_caps)}")
    print(f"  Non-empty: {sum(1 for v in vlm_caps.values() if v)}")

    train_data = load_train()
    test_data = load_test()
    train_subset, dev_subset = split_train_dev(train_data)
    print(f"📊 Train: {len(train_subset)}, Dev: {len(dev_subset)}, Test: {len(test_data)}")

    dev_have, dev_miss = coverage_stats(dev_subset, vlm_caps)
    test_have, test_miss = coverage_stats(test_data, vlm_caps)
    print(f"  Dev   img coverage: {dev_have}/{dev_have+dev_miss} ({100*dev_have/max(dev_have+dev_miss,1):.1f}%)")
    print(f"  Test  img coverage: {test_have}/{test_have+test_miss} ({100*test_have/max(test_have+test_miss,1):.1f}%)")

    # BM25 prior
    bm25_test_path = find_latest_rankings("phase_1", "rankings_top100.json")
    bm25_dev_path = find_latest_rankings("phase_1", "dev_rankings_top100.json")
    bm25_test = load_rankings(bm25_test_path)
    bm25_dev = load_rankings(bm25_dev_path)

    from FlagEmbedding import BGEM3FlagModel
    print("📦 Loading BAAI/bge-m3...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    output_dir = config.get_output_dir("phase_3c", "vlm_caption_4way_rrf")

    dev_golds = get_gold_quotes_dict(dev_subset)

    # 跑兩個 strategy: 'vlm' 全替換, 'concat' 拼接
    results_per_strategy = {}
    for strategy in ["vlm", "concat"]:
        print(f"\n=== Strategy: {strategy} ===")
        t0 = time.time()
        d_dev, l_dev, c_dev, mod_dev = run_bgem3_retrieval_vlm(model, dev_subset, vlm_caps, strategy)
        print(f"  Dev encode+score: {time.time()-t0:.1f}s")

        rs4 = [bm25_dev, d_dev, l_dev, c_dev]
        sweep = []
        best_r, best_boost, best_preds = -1.0, None, None
        for boost in [0.0, 0.002, 0.003, 0.004, 0.005, 0.006]:
            fused = rrf_with_image_boost(rs4, mod_dev, k=RRF_K, weights=[1, 1, 1, 1],
                                         image_boost=boost, top_k=100)
            preds5 = {q: v[:5] for q, v in fused.items()}
            r = recall_at_k(preds5, dev_golds, k=5)
            stats = modality_split_stats(preds5, mod_dev)
            sweep.append({"strategy": strategy, "boost": boost,
                          "dev_recall_at_5": r,
                          "text_pct": stats["text_pct"], "image_pct": stats["image_pct"]})
            print(f"  boost={boost:.3f}  recall={r:.4f}  text/img={stats['text_pct']:.1f}/{stats['image_pct']:.1f}%")
            if r > best_r:
                best_r, best_boost, best_preds = r, boost, preds5
        print(f"  → best boost={best_boost}  dev_recall={best_r:.4f}")
        results_per_strategy[strategy] = {
            "dev_recall_at_5": best_r,
            "best_boost": best_boost,
            "best_preds": best_preds,
            "sweep": sweep,
            "dev_rankings": (d_dev, l_dev, c_dev, mod_dev),
        }

    # 選 dev recall 最佳的 strategy 用在 test
    best_strategy = max(results_per_strategy, key=lambda s: results_per_strategy[s]["dev_recall_at_5"])
    best_info = results_per_strategy[best_strategy]
    print(f"\n📈 Best strategy: {best_strategy}  dev_recall={best_info['dev_recall_at_5']:.4f}  boost={best_info['best_boost']}")

    # Test
    print(f"\n🔍 Run on TEST with strategy={best_strategy}...")
    t0 = time.time()
    d_test, l_test, c_test, mod_test = run_bgem3_retrieval_vlm(model, test_data, vlm_caps, best_strategy)
    print(f"  Test encode+score: {time.time()-t0:.1f}s")

    test_fused = rrf_with_image_boost(
        [bm25_test, d_test, l_test, c_test],
        mod_test, k=RRF_K, weights=[1, 1, 1, 1],
        image_boost=best_info["best_boost"], top_k=100,
    )
    test_preds_top5 = {q: v[:5] for q, v in test_fused.items()}

    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds_top5, test_data, submission_path)

    rankings_path = os.path.join(output_dir, "rankings_top100.json")
    save_rankings(test_fused, rankings_path)

    # dev rankings 也存 (供 Phase 4 reranker 使用)
    d_dev, l_dev, c_dev, mod_dev = best_info["dev_rankings"]
    dev_fused = rrf_with_image_boost(
        [bm25_dev, d_dev, l_dev, c_dev],
        mod_dev, k=RRF_K, weights=[1, 1, 1, 1],
        image_boost=best_info["best_boost"], top_k=100,
    )
    save_rankings(dev_fused, os.path.join(output_dir, "dev_rankings_top100.json"))

    ea = error_analysis(best_info["best_preds"], dev_golds, k=5)
    metrics = {
        "phase": "phase_3c",
        "model": "BGE-M3 multi-vec + BM25 + VLM caption + img_boost",
        "vlm_caption_source": VLM_CAPTIONS_FILE,
        "vlm_caption_count": sum(1 for v in vlm_caps.values() if v),
        "best_strategy": best_strategy,
        "best_image_boost": best_info["best_boost"],
        "dev_recall_at_5": best_info["dev_recall_at_5"],
        "results_per_strategy": {
            s: {"dev_recall_at_5": v["dev_recall_at_5"], "best_boost": v["best_boost"], "sweep": v["sweep"]}
            for s, v in results_per_strategy.items()
        },
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)

    append_leaderboard(
        "Phase 3c",
        f"BGE-M3 4-way RRF + VLM caption ({best_strategy}) + img_boost",
        best_info["dev_recall_at_5"],
        note=f"strategy={best_strategy}, boost={best_info['best_boost']}",
    )

    print(f"\n✅ Phase 3c 完成! 輸出目錄: {output_dir}")
    print(f"📤 Submission: {submission_path}")


if __name__ == "__main__":
    main()
