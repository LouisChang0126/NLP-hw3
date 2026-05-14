"""
Evaluate Phase 6 local (Qwen3.6-27B Q6_K, CL=1500) on large_dev_ids.json (n=899).

Reuses existing 200-sample dev predictions from llm_picks_cache_cl1500.json
(safe — those 200 q_ids are a subset of large_dev and were the FIRST writes
to that cache, so they are genuine dev predictions, not test contamination).

The remaining 699 q_ids are computed fresh into a separate cache to avoid
the train/test q_id-collision contamination noted in MEMORY.md.

Run (GPU 1):
    CUDA_VISIBLE_DEVICES=1 python tools/eval_large_dev.py
"""
import json
import os
import sys
import time
from typing import Dict, List

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset import load_train, build_candidates, get_gold_quotes_dict
from evaluation import recall_at_k, error_analysis
from run_phase6_direct_llm import PROMPT_TEMPLATE, parse_ids, format_candidates
from run_phase6_local import build_qwen_prompt, download_gguf, load_llm

# ── 設定 ────────────────────────────────────────────────────────
os.environ.setdefault("CHAR_LIMIT_PER_CANDIDATE", "1500")
os.environ.setdefault("N_CTX", "8192")

CL = int(os.environ["CHAR_LIMIT_PER_CANDIDATE"])
LARGE_DEV_FILE = os.path.join(config.SPLITS_DIR, "large_dev_ids.json")
SMALL_DEV_CACHE = os.path.join(
    config.OUTPUT_DIR, "phase_6_local", f"llm_picks_cache_cl{CL}.json"
)
LARGE_DEV_CACHE = os.path.join(
    config.OUTPUT_DIR, "phase_6_local", f"llm_picks_cache_cl{CL}_largedev.json"
)

MAX_TOKENS = 64
TEMPERATURE = 0.0
SAVE_EVERY = 25


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_cache(cache, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def main():
    config.print_config()

    # ── 載入 large_dev ─────────────────────────────────────────
    large_dev_qids = set(load_json(LARGE_DEV_FILE))
    train = load_train()
    large_dev = [s for s in train if s["q_id"] in large_dev_qids]
    assert len(large_dev) == len(large_dev_qids), \
        f"split size mismatch: {len(large_dev)} vs {len(large_dev_qids)}"
    print(f"\n📊 large_dev: {len(large_dev)} samples")

    # ── 載入既有 small-dev cache, 取交集作為 seed ───────────────
    cache = {}
    if os.path.exists(SMALL_DEV_CACHE):
        small_cache = load_json(SMALL_DEV_CACHE)
        # small_dev (200 筆) 為 large_dev 子集 → 那 200 條 q_id 直接複用
        small_dev_ids = set(load_json(
            os.path.join(config.SPLITS_DIR, "dev_ids.json")
        ))
        reused = 0
        for qid in small_dev_ids:
            k = str(qid)
            if k in small_cache and len(small_cache[k]) == 5:
                cache[k] = small_cache[k]
                reused += 1
        print(f"🪄 reused {reused} predictions from {SMALL_DEV_CACHE}")

    # 若已存在 large-dev cache, 合併
    if os.path.exists(LARGE_DEV_CACHE):
        prev = load_json(LARGE_DEV_CACHE)
        for k, v in prev.items():
            if len(v) == 5:
                cache.setdefault(k, v)
        print(f"   + {len(prev)} from prior large-dev cache")

    save_cache(cache, LARGE_DEV_CACHE)

    todo = [s for s in large_dev
            if str(s["q_id"]) not in cache or len(cache[str(s["q_id"])]) != 5]
    print(f"📥 cached: {len(cache)},  ⏳ todo: {len(todo)}")

    # ── 載入 LLM (僅當有 todo 時) ──────────────────────────────
    if todo:
        gguf_path = download_gguf()
        llm = load_llm(gguf_path)
        pbar = tqdm(total=len(todo), desc="large_dev")
        t0 = time.time()
        last_save = 0
        for sample in todo:
            q_id_str = str(sample["q_id"])
            cands = build_candidates(sample)
            allowed = {c["quote_id"].lower() for c in cands}
            prompt = PROMPT_TEMPLATE.format(
                question=sample["question"],
                candidates=format_candidates(cands),
            )
            try:
                full_prompt = build_qwen_prompt(prompt)
                r = llm.create_completion(
                    prompt=full_prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stop=["<|im_end|>", "<|endoftext|>"],
                )
                raw = r["choices"][0]["text"] or ""
                picks = parse_ids(raw, allowed, top_k=5)
                if len(picks) < 5:
                    picks = [c["quote_id"] for c in cands[:5]]
                cache[q_id_str] = picks
            except Exception as e:
                tqdm.write(f"❌ q={q_id_str}: {type(e).__name__}: {str(e)[:120]}")
                cache[q_id_str] = [c["quote_id"] for c in cands[:5]]

            pbar.update(1)
            if (pbar.n - last_save) >= SAVE_EVERY:
                save_cache(cache, LARGE_DEV_CACHE)
                last_save = pbar.n
        pbar.close()
        save_cache(cache, LARGE_DEV_CACHE)
        wall = time.time() - t0
        print(f"  Inference wall: {wall:.1f}s  ({wall/max(len(todo),1):.2f}s / sample)")
    else:
        print("✅ All predictions cached, skipping inference.")

    # ── 評估 ───────────────────────────────────────────────────
    preds = {int(k): v for k, v in cache.items() if int(k) in large_dev_qids}
    golds = get_gold_quotes_dict(large_dev)
    recall = recall_at_k(preds, golds, k=5)
    ea = error_analysis(preds, golds, k=5)

    print()
    print("=" * 60)
    print(f"📈 large_dev Recall@5 (n={len(large_dev)}) = {recall:.4f}")
    print(f"   Perfect: {ea['summary']['perfect_count']}")
    print(f"   Partial: {ea['summary']['partial_count']}")
    print(f"   Zero:    {ea['summary']['zero_count']}")
    print("=" * 60)

    # modality split
    text_n, img_n = 0, 0
    for qid, lst in preds.items():
        for qid2 in lst[:5]:
            if qid2.lower().startswith("image"):
                img_n += 1
            else:
                text_n += 1
    print(f"   Top-5 modality: text {text_n} ({100*text_n/(text_n+img_n):.1f}%)"
          f" / img {img_n} ({100*img_n/(text_n+img_n):.1f}%)")

    # save metric file
    out_dir = os.path.join(config.OUTPUT_DIR, "phase_6_local",
                           f"large_dev_cl{CL}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "split": "large_dev_ids.json",
            "n_samples": len(large_dev),
            "char_limit": CL,
            "model": "unsloth/Qwen3.6-27B-Q6_K.gguf",
            "recall_at_5": recall,
            "error_analysis_summary": ea["summary"],
            "modality": {"text": text_n, "image": img_n},
        }, f, indent=2)
    print(f"💾 metrics → {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
