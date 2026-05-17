"""
INLP HW3 - Phase 6 (local): Qwen3.6-27B GGUF Q6_K via llama-cpp-python

與 run_phase6_direct_llm.py 邏輯一致 (15 candidates → LLM 直接挑 5)，
但本地 GPU 推理, 無 NIM API.

需先安裝 (CUDA build of llama-cpp-python):
    CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall \
        llama-cpp-python --no-cache-dir

第一次執行會自動從 HF 下載 ~22 GB Q6_K GGUF (約 5-15 min).

執行:
    export CUDA_VISIBLE_DEVICES=0       # 單卡夠 (Q6_K ~22GB + KV ~2GB)
    python run_phase6_local.py
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
from submission_utils import generate_submission, save_metrics, append_leaderboard
from run_phase6_direct_llm import (
    PROMPT_TEMPLATE, parse_ids, format_candidates,
)


# ── 模型 ─────────────────────────────────────────────
REPO_ID = os.environ.get("REPO_ID", "unsloth/Qwen3.6-27B-GGUF")
GGUF_FILE = os.environ.get("GGUF_FILE", "Qwen3.6-27B-Q6_K.gguf")
# 若實際檔名不同 (e.g. sharded), 用 GGUF_FILE 環境變數覆蓋

# ── 推理參數 (預設值 = LB-best, Dev 0.8988 / LB 0.81895) ─────────
N_GPU_LAYERS = -1     # -1 = 全部 layers 上 GPU
# 環境變數可覆寫超參數, e.g. N_CTX=4096 CHAR_LIMIT_PER_CANDIDATE=800 python run_phase6_local.py
N_CTX = int(os.environ.get("N_CTX", 8192))
N_BATCH = 512         # prefill 批次大小
MAX_TOKENS = 2048
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
TOP_P = float(os.environ.get("TOP_P", 1.0))
SAVE_EVERY = 25
# 不同模型 / CHAR_LIMIT / temp 用不同 cache, 避免污染
_CL = int(os.environ.get("CHAR_LIMIT_PER_CANDIDATE", 1500))
_MODEL_TAG = os.path.splitext(GGUF_FILE)[0].lower().replace(".", "_")
_IS_DEFAULT_MODEL = GGUF_FILE == "Qwen3.6-27B-Q6_K.gguf"
_TEMP_SUF = "" if TEMPERATURE == 0.0 else f"_t{TEMPERATURE}".replace(".", "")
_CACHE_SUFFIX = os.environ.get("CACHE_SUFFIX", "")
if _IS_DEFAULT_MODEL:
    CACHE_PATH = os.path.join(
        config.OUTPUT_DIR, "phase_6_local",
        f"llm_picks_cache_cl{_CL}{_TEMP_SUF}{_CACHE_SUFFIX}.json" if _CL != 800
        else f"llm_picks_cache{_TEMP_SUF}{_CACHE_SUFFIX}.json",
    )
else:
    CACHE_PATH = os.path.join(
        config.OUTPUT_DIR, "phase_6_local",
        f"llm_picks_cache_{_MODEL_TAG}_cl{_CL}{_TEMP_SUF}{_CACHE_SUFFIX}.json",
    )


# ══════════════════════════════════════════════════════
#  模型載入 / 下載
# ══════════════════════════════════════════════════════

def download_gguf() -> str:
    """從 HF Hub 下載 GGUF, 回傳本地路徑 (有快取就直接用)"""
    from huggingface_hub import hf_hub_download
    print(f"📥 Resolving {REPO_ID} / {GGUF_FILE}...")
    t0 = time.time()
    path = hf_hub_download(repo_id=REPO_ID, filename=GGUF_FILE)
    dt = time.time() - t0
    size_mb = os.path.getsize(path) / 1e6
    print(f"  ✅ {path}  ({size_mb:.0f} MB, fetched/cached in {dt:.1f}s)")
    return path


def load_llm(gguf_path: str):
    from llama_cpp import Llama
    print(f"📦 Loading model (n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX})...")
    t0 = time.time()
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        verbose=False,
        seed=config.SEED,
    )
    print(f"  ✅ Model loaded in {time.time()-t0:.1f}s")
    return llm


def build_qwen_prompt(user_content: str) -> str:
    """
    手動組裝 Qwen3.5/3.6 chat template, **預先關閉 thinking** ─
    chat template 預設 enable_thinking=True 會塞 <think>\\n 給模型,
    使得 max_tokens=64 全部花在 reasoning 上、沒輸出答案,
    這正是 LB 0.32049 的根因. 用空 <think></think> 略過 reasoning.
    """
    return (
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


# ══════════════════════════════════════════════════════
#  Cache (resumable)
# ══════════════════════════════════════════════════════

def load_cache() -> Dict[str, List[str]]:
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


# ══════════════════════════════════════════════════════
#  Inference loop (single-process, sequential)
# ══════════════════════════════════════════════════════

def run_inference(samples: List[dict], llm, desc: str) -> Dict[int, List[str]]:
    cache = load_cache()
    # v4: 接受 <5 cache, 不重跑
    todo = [s for s in samples if str(s["q_id"]) not in cache]
    print(f"📥 cached: {len(cache)},  ⏳ todo: {len(todo)} ({desc})")
    if not todo:
        return {int(k): v for k, v in cache.items()
                if int(k) in {s["q_id"] for s in samples}}

    pbar = tqdm(total=len(todo), desc=desc)
    t_start = time.time()
    last_save = 0
    for sample in todo:
        q_id_str = str(sample["q_id"])
        cands = build_candidates(sample)
        allowed = {c["quote_id"].lower() for c in cands}
        prompt = PROMPT_TEMPLATE.format(
            questions=sample["question"],
            candidates=format_candidates(cands, char_limit=_CL),
        )
        try:
            # 用 raw completion + 手動 chat template (bypass 預設 thinking)
            full_prompt = build_qwen_prompt(prompt)
            r = llm.create_completion(
                prompt=full_prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            raw = r["choices"][0]["text"] or ""
            picks = parse_ids(raw, allowed, top_k=5)
            # v4: 接受 <5, 不 fallback
            cache[q_id_str] = picks
        except Exception as e:
            tqdm.write(f"❌ q={q_id_str}: {type(e).__name__}: {str(e)[:120]}")
            cache[q_id_str] = []  # v4: 失敗交空

        pbar.update(1)
        # 每 SAVE_EVERY 題存一次
        if (pbar.n - last_save) >= SAVE_EVERY:
            save_cache(cache)
            last_save = pbar.n
    pbar.close()
    save_cache(cache)
    wall = time.time() - t_start
    print(f"  Inference wall: {wall:.1f}s  ({wall/max(len(todo),1):.2f}s / sample)")
    return {int(k): v for k, v in cache.items() if int(k) in {s["q_id"] for s in samples}}


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════

def main():
    config.print_config()

    train_data = load_train()
    test_data = load_test()
    _, dev_subset = split_train_dev(train_data)
    print(f"\n📊 Dev: {len(dev_subset)}, Test: {len(test_data)}")

    gguf_path = download_gguf()
    llm = load_llm(gguf_path)

    # ── DEV 先 (200 題, 預期 ~3 min) ─────────────────
    print("\n🔍 Stage 1: dev evaluation")
    dev_preds = run_inference(dev_subset, llm, "dev")
    dev_golds = get_gold_quotes_dict(dev_subset)
    dev_recall = recall_at_k(dev_preds, dev_golds, k=5)
    print(f"\n📈 Dev Recall@5 (Qwen3.6-27B Q6_K local) = {dev_recall:.4f}")
    ea = error_analysis(dev_preds, dev_golds, k=5)
    print(f"  Perfect: {ea['summary']['perfect_count']}")
    print(f"  Partial: {ea['summary']['partial_count']}")
    print(f"  Zero:    {ea['summary']['zero_count']}")

    # modality split (dev)
    text_n, img_n = 0, 0
    for q, lst in dev_preds.items():
        for qid in lst[:5]:
            if qid.lower().startswith("image"):
                img_n += 1
            else:
                text_n += 1
    print(f"  Dev Top-5 modality: text {text_n} ({100*text_n/(text_n+img_n):.1f}%) "
          f"/ img {img_n} ({100*img_n/(text_n+img_n):.1f}%)")

    # ── TEST ────────────────────────────────────────
    print("\n🔍 Stage 2: test prediction")
    test_preds = run_inference(test_data, llm, "test")

    output_dir = config.get_output_dir("phase_6_local", _MODEL_TAG)
    submission_path = os.path.join(output_dir, "submission.csv")
    # v4: pad_short=False → 若 LLM 給 <5 直接交 <5
    generate_submission(test_preds, test_data, submission_path, pad_short=False)

    metrics = {
        "phase": "phase_6_local",
        "model_repo": REPO_ID,
        "gguf_file": GGUF_FILE,
        "approach": "direct 15->5 LLM selection (local GGUF inference)",
        "candidate_caption_source": "original img_description (no VLM)",
        "n_gpu_layers": N_GPU_LAYERS,
        "n_ctx": N_CTX,
        "temperature": TEMPERATURE,
        "dev_recall_at_5": dev_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)
    append_leaderboard(
        "Phase 6 (local)",
        f"{REPO_ID}/{GGUF_FILE}",
        dev_recall,
        note=f"local GGUF Q6_K via llama-cpp-python",
    )

    print(f"\n✅ Phase 6 (local) 完成! 輸出: {output_dir}")
    print(f"📤 Submission: {submission_path}")


if __name__ == "__main__":
    main()
