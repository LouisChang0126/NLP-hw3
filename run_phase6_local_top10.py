"""
INLP HW3 - Phase 6 hybrid: Phase 3b top-10 候選 → Qwen3.6-27B 挑 5

實驗動機: 純 15→5 LLM (Phase 6 local CL=1500) LB 0.81895.
        如果先用 Phase 3b 過濾掉 5 個低分候選, LLM 只看 top-10,
        雜訊變少, 是否能再上分?

設定 (跟 Phase 6 local CL=1500 對齊):
  - Model: Qwen3.6-27B Q6_K (unsloth)
  - char_limit per candidate: 1500
  - n_ctx: 8192
  - GPU 1 (CUDA_VISIBLE_DEVICES=1)
  - bypass thinking (空 <think></think> 預填)
  - prompt 與 parse_ids 完全與 Phase 6 一致, 只是候選池縮成 10
"""
import json
import os
import time
from typing import Dict, List

from tqdm import tqdm

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
from run_phase6_direct_llm import PROMPT_TEMPLATE, parse_ids, format_candidates
from run_phase6_local import (
    REPO_ID, GGUF_FILE, N_GPU_LAYERS, N_BATCH, MAX_TOKENS, TEMPERATURE, SAVE_EVERY,
    download_gguf, load_llm, build_qwen_prompt,
)


N_CTX = int(os.environ.get("N_CTX", 8192))
TOP_K_CANDIDATES = int(os.environ.get("TOP_K_CANDIDATES", 10))
CACHE_PATH = os.path.join(
    config.OUTPUT_DIR, "phase_6_local_top10",
    f"llm_picks_top{TOP_K_CANDIDATES}.json",
)


def filter_to_top_k(sample: dict, top_ids: List[str], k: int) -> List[dict]:
    """從原候選池取出排在 Phase 3b top-k 的候選, 保留原 build_candidates 順序"""
    cands = build_candidates(sample)
    keep = set(top_ids[:k])
    return [c for c in cands if c["quote_id"] in keep]


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_PATH)


def run_inference(samples, llm, p3b_rankings, desc):
    cache = load_cache()
    todo = [s for s in samples
            if str(s["q_id"]) not in cache or len(cache[str(s["q_id"])]) != 5]
    print(f"📥 cached: {len(cache)},  ⏳ todo: {len(todo)} ({desc})")
    if not todo:
        return {int(k): v for k, v in cache.items()
                if int(k) in {s["q_id"] for s in samples}}

    pbar = tqdm(total=len(todo), desc=desc)
    t_start = time.time()
    last_save = 0
    for sample in todo:
        q_id = sample["q_id"]
        q_id_str = str(q_id)

        top_ids = p3b_rankings.get(q_id, [])
        cands = filter_to_top_k(sample, top_ids, TOP_K_CANDIDATES)
        if len(cands) == 0:
            # 安全 fallback (理論上不應發生)
            cands = build_candidates(sample)[:TOP_K_CANDIDATES]

        allowed = {c["quote_id"].lower() for c in cands}
        user_content = PROMPT_TEMPLATE.format(
            question=sample["question"],
            candidates=format_candidates(cands),
        )
        full_prompt = build_qwen_prompt(user_content)

        try:
            r = llm.create_completion(
                prompt=full_prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            raw = r["choices"][0]["text"] or ""
            picks = parse_ids(raw, allowed, top_k=5)
            if len(picks) < 5:
                # fallback: 用 top-k 前 5 個
                picks = [c["quote_id"] for c in cands[:5]]
            cache[q_id_str] = picks
        except Exception as e:
            tqdm.write(f"❌ q={q_id_str}: {type(e).__name__}: {str(e)[:120]}")
            cache[q_id_str] = [c["quote_id"] for c in cands[:5]]

        pbar.update(1)
        if (pbar.n - last_save) >= SAVE_EVERY:
            save_cache(cache)
            last_save = pbar.n
    pbar.close()
    save_cache(cache)
    wall = time.time() - t_start
    print(f"  Inference wall: {wall:.1f}s  ({wall/max(len(todo),1):.2f}s / sample)")
    return {int(k): v for k, v in cache.items() if int(k) in {s["q_id"] for s in samples}}


def main():
    config.print_config()
    print(f"\n📋 Settings: TOP_K_CANDIDATES={TOP_K_CANDIDATES}, "
          f"CHAR_LIMIT_PER_CANDIDATE={os.environ.get('CHAR_LIMIT_PER_CANDIDATE', 800)}, "
          f"N_CTX={N_CTX}")

    train_data = load_train()
    test_data = load_test()
    _, dev_subset = split_train_dev(train_data)
    print(f"\n📊 Dev: {len(dev_subset)}, Test: {len(test_data)}")

    # ── Phase 3b rankings ────────────────────────────
    p3b_test_path = find_latest_rankings("phase_3b", "rankings_top100.json")
    p3b_dev_path = find_latest_rankings("phase_3b", "dev_rankings_top100.json")
    print(f"\n📂 Phase 3b rankings:")
    print(f"  test: {p3b_test_path}")
    print(f"  dev:  {p3b_dev_path}")
    p3b_test = load_rankings(p3b_test_path)
    p3b_dev = load_rankings(p3b_dev_path)

    # ── Load model ───────────────────────────────────
    gguf_path = download_gguf()
    # override N_CTX in load_llm (it reads from run_phase6_local at import time;
    # we patch the module global)
    import run_phase6_local
    run_phase6_local.N_CTX = N_CTX
    llm = load_llm(gguf_path)

    # ── DEV ──────────────────────────────────────────
    print("\n🔍 Stage 1: dev (top-10 candidates per question)")
    dev_preds = run_inference(dev_subset, llm, p3b_dev, "dev")
    dev_golds = get_gold_quotes_dict(dev_subset)
    dev_recall = recall_at_k(dev_preds, dev_golds, k=5)
    print(f"\n📈 Dev Recall@5 (Phase 3b top-{TOP_K_CANDIDATES} → Qwen3.6) = {dev_recall:.4f}")
    ea = error_analysis(dev_preds, dev_golds, k=5)
    print(f"  Perfect: {ea['summary']['perfect_count']}")
    print(f"  Partial: {ea['summary']['partial_count']}")
    print(f"  Zero:    {ea['summary']['zero_count']}")

    text_n, img_n = 0, 0
    for q, lst in dev_preds.items():
        for qid in lst[:5]:
            if qid.lower().startswith("image"):
                img_n += 1
            else:
                text_n += 1
    print(f"  Dev Top-5 modality: text {text_n} ({100*text_n/(text_n+img_n):.1f}%) "
          f"/ img {img_n} ({100*img_n/(text_n+img_n):.1f}%)")

    # ── TEST ─────────────────────────────────────────
    print(f"\n🔍 Stage 2: test (top-{TOP_K_CANDIDATES} candidates per question)")
    test_preds = run_inference(test_data, llm, p3b_test, "test")

    output_dir = config.get_output_dir("phase_6_local_top10",
                                        f"qwen36_27b_q6k_top{TOP_K_CANDIDATES}")
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds, test_data, submission_path)

    metrics = {
        "phase": "phase_6_local_top10",
        "model_repo": REPO_ID,
        "gguf_file": GGUF_FILE,
        "approach": f"Phase 3b top-{TOP_K_CANDIDATES} → Qwen3.6 picks 5",
        "p3b_test_source": p3b_test_path,
        "p3b_dev_source": p3b_dev_path,
        "top_k_candidates": TOP_K_CANDIDATES,
        "char_limit_per_candidate": int(os.environ.get("CHAR_LIMIT_PER_CANDIDATE", 800)),
        "n_ctx": N_CTX,
        "dev_recall_at_5": dev_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)
    append_leaderboard(
        f"Phase 6 hybrid top-{TOP_K_CANDIDATES}",
        f"Phase 3b top-{TOP_K_CANDIDATES} → Qwen3.6-27B Q6_K",
        dev_recall,
        note=f"CL={os.environ.get('CHAR_LIMIT_PER_CANDIDATE', 800)}, n_ctx={N_CTX}",
    )

    print(f"\n✅ Phase 6 hybrid 完成! 輸出: {output_dir}")
    print(f"📤 Submission: {submission_path}")


if __name__ == "__main__":
    main()
