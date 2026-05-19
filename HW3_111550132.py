"""
INLP HW3 - Phase 6: 純 LLM 直接從 15 個候選挑 5 個

Model: google/gemma-4-31b-it (via NVIDIA NIM)
- Deterministic decoding (temperature=0, top_p=0.1)
- Modality pruning: evidence_modality_type 限制候選池
- Min-quotes retry loop (預設門檻 5, 最多 12 attempts)
- 多執行緒 (預設 4) + 60s sliding-window rate limit (預設 37 calls)
- 可中斷可續跑（cache resumable, atomic write）
"""
import collections
import json
import os
import re
import random
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from tqdm import tqdm
from openai import OpenAI

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict,
)
from evaluation import recall_at_k, error_analysis
from submission_utils import generate_submission, save_metrics, append_leaderboard


MODEL = "google/gemma-4-31b-it"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
# Sliding-window rate limit: max calls per 60-second window (NIM 上限 40, 預設 37 留 buffer)
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", 37))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 2048))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
TOP_P = float(os.environ.get("TOP_P", 0.1))
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 12))
MIN_QUOTES_THRESHOLD = int(os.environ.get("MIN_QUOTES_THRESHOLD", 5))  # SOTA.py default 5; retry if parsed picks < this
CHAR_LIMIT_PER_CANDIDATE = int(os.environ.get("CHAR_LIMIT_PER_CANDIDATE", 0))
SAVE_EVERY = 50
_SUF = os.environ.get("CACHE_SUFFIX", "")
CACHE_PATH = os.path.join(config.OUTPUT_DIR, "phase_6", f"llm_picks_cache{_SUF}.json")
# 觀察用：與 SOTA.py 相容的 per-question 完整回應紀錄
RESULTS_PATH = os.path.join(config.OUTPUT_DIR, "phase_6", f"gemma_answering_results{_SUF}.json")

# ── Evidence modality pruning rule (from SOTA.py) ────────────────────
# train.jsonl 觀察硬規則：
#   evidence_modality_type 只含 text          → gold 100% 只有 text*
#   evidence_modality_type 只含 image-like    → gold 100% 只有 image*
# 因此在純單一型態的題目，把另一型態的候選 prune 掉可大幅降低 distractor。
_TEXT_MODALITIES = {"text"}
_IMAGE_MODALITIES = {"table", "figure", "chart", "image"}


def filter_candidates_by_modality(sample: dict, cands: List[dict]) -> List[dict]:
    """依 evidence_modality_type 剔除不可能成為答案的型態。空 / 兼有兩者 → 不過濾。"""
    mods = set(sample.get("evidence_modality_type") or [])
    has_text = bool(mods & _TEXT_MODALITIES)
    has_image = bool(mods & _IMAGE_MODALITIES)
    keep_text = has_text or not (has_text or has_image)
    keep_image = has_image or not (has_text or has_image)

    out = []
    for c in cands:
        m = c.get("modality")
        if m == "text" and keep_text:
            out.append(c)
        elif m == "image" and keep_image:
            out.append(c)
    return out or cands  # 守備：filter 後空就回退


# ── RateLimiter — sliding-window cross-thread (from SOTA.py) ─────────
class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window = window_seconds
        self._ts: "collections.deque[float]" = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window
                while self._ts and self._ts[0] <= cutoff:
                    self._ts.popleft()
                if len(self._ts) < self.max_calls:
                    self._ts.append(now)
                    return
                wait_until = self._ts[0] + self.window
            time.sleep(max(0.05, wait_until - time.monotonic()))


RATE_LIMITER = RateLimiter(max_calls=RATE_LIMIT_PER_MIN, window_seconds=60.0)


PROMPT_TEMPLATE = """You are an expert retrieval assistant. I will provide you with a question and a list of evidence items. Your task is to analyze the evidence and extract the top 5 most important evidence IDs that best answer the question.

Question: {questions}

Evidence Items:
{candidates}

You MUST extract and rank EXACTLY 5 evidence IDs in descending order of importance. Even if you think fewer than 5 items are relevant, you MUST fill all 5 spots with your best guesses. DO NOT output fewer than 5 IDs. Output ONLY the 5 IDs separated by commas (e.g., text1, image3, text5, image2, text10)."""


# ── persistence ──────────────────────────────────────
SAVE_LOCK = threading.Lock()
STOP = threading.Event()


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with SAVE_LOCK:
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, CACHE_PATH)


def load_results() -> Dict[str, dict]:
    """以 q_id_str 索引的完整回應紀錄；檔案落地時是 SOTA.py 的 list 格式。"""
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(r["q_id"]): r for r in data if "q_id" in r}
    # 容錯：若曾被改成 dict 格式也讀得進來
    return {str(k): v for k, v in data.items()}


def save_results(results: Dict[str, dict]):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with SAVE_LOCK:
        tmp = RESULTS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
        os.replace(tmp, RESULTS_PATH)


# ── 解析 LLM 輸出 ────────────────────────────────────────────────
ID_PATTERN = re.compile(r"(text|image)[\s\-_]*(\d+)", re.IGNORECASE)


def parse_ids(text: str, allowed_ids: set, top_k: int = 5) -> List[str]:
    """從 LLM 輸出抽出合法 ID, 保序去重; 不足 top_k 就直接交"""
    if not text:
        return []
    found = ID_PATTERN.findall(text)
    seen = set()
    out = []
    for prefix, num in found:
        fid = f"{prefix.lower()}{int(num)}"
        if fid in allowed_ids and fid not in seen:
            out.append(fid)
            seen.add(fid)
            if len(out) >= top_k:
                break
    return out


# ── prompt 組裝 ──────────────────────────────────────
def format_candidates(cands, char_limit: int = None):
    """char_limit=None → 用模組層級 CHAR_LIMIT_PER_CANDIDATE; =0 表示不截斷"""
    limit = CHAR_LIMIT_PER_CANDIDATE if char_limit is None else char_limit
    lines = []
    for c in cands:
        # 不 strip：SOTA reference 直接灌 raw text；拿掉 strip 才能 byte-for-byte 對齊
        text = c["text_for_retrieval"] or ""
        if limit > 0 and len(text) > limit:
            text = text[:limit] + "..."
        lines.append(f"[{c['quote_id']}]: {text}")
    return "\n".join(lines)


# ── worker ───────────────────────────────────────────
def worker(
    worker_id: int,
    samples: List[dict],
    cache: Dict[str, List[str]],
    results: Dict[str, dict],
    pbar,
):
    client = OpenAI(base_url=config.NIM_BASE_URL, api_key=config.NIM_API_KEY)
    for sample in samples:
        if STOP.is_set():
            return
        q_id_str = str(sample["q_id"])
        # Only skip if we already have a full 5-pick result cached
        if q_id_str in cache and len(cache[q_id_str]) >= 5:
            pbar.update(1)
            continue

        cands = filter_candidates_by_modality(sample, build_candidates(sample))
        allowed_list = [c["quote_id"].lower() for c in cands]  # ordered fallback pool
        allowed = set(allowed_list)
        prompt = PROMPT_TEMPLATE.format(
            questions=sample["question"],
            candidates=format_candidates(cands),
        )

        # Retry until len(picks) >= MIN_QUOTES_THRESHOLD or attempts run out.
        # Threshold is always MIN_QUOTES_THRESHOLD (5) unless `allowed` itself
        # has fewer items — in which case we accept what we can get and pad
        # with the rest before caching.
        best_picks: List[str] = list(cache.get(q_id_str, []))
        last_raw: str = ""  # 最近一次 (或最佳一次) 的原始回應，方便事後檢視
        threshold = min(MIN_QUOTES_THRESHOLD, max(1, len(allowed)))
        for attempt in range(MAX_RETRIES):
            RATE_LIMITER.acquire()
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    extra_body={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
                )
                raw = r.choices[0].message.content or ""
                last_raw = raw
                picks = parse_ids(raw, allowed, top_k=5)
                if len(picks) > len(best_picks):
                    best_picks = picks
                if len(best_picks) >= threshold:
                    break
                # not enough IDs — backoff and retry
                backoff = min(2.0 * (attempt + 1), 30.0)
                time.sleep(backoff)
            except Exception as e:
                err = type(e).__name__
                msg = str(e)[:120]
                last_raw = f"[Error] {err}: {msg}"
                if "429" in msg or "rate" in msg.lower():
                    backoff = 4 * (2 ** attempt) + random.random()
                    tqdm.write(f"[w{worker_id}] 429, sleep {backoff:.1f}s")
                    time.sleep(backoff)
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(2 + random.random() * 2)
                else:
                    tqdm.write(f"[w{worker_id}] ❌ q={q_id_str} give up: {err}: {msg}")

        # Guarantee 5 picks before caching: pad from `allowed_list` (preserves
        # the original candidate order from build_candidates).
        if len(best_picks) < 5:
            seen = set(best_picks)
            for qid in allowed_list:
                if qid not in seen:
                    best_picks.append(qid)
                    seen.add(qid)
                    if len(best_picks) >= 5:
                        break
        cache[q_id_str] = best_picks[:5]
        results[q_id_str] = {
            "q_id": sample["q_id"],
            "question": sample["question"],
            "prompt": prompt,
            "predicted_quotes": best_picks[:5],
            "model_raw_response": last_raw,
        }

        if len(cache) % SAVE_EVERY == 0:
            save_cache(cache)
            save_results(results)
        pbar.update(1)


# ── 流程 ─────────────────────────────────────────────
def run_llm_pick(data: List[dict], desc: str) -> Dict[int, List[str]]:
    cache = load_cache()
    results = load_results()
    todo = [s for s in data if len(cache.get(str(s["q_id"]), [])) < 5]
    full = sum(1 for s in data if len(cache.get(str(s["q_id"]), [])) >= 5)
    print(f"📥 cached (>=5): {full},  ⏳ todo: {len(todo)} ({desc})")
    if not todo:
        return {int(k): v for k, v in cache.items() if int(k) in {s["q_id"] for s in data}}

    def sigh(_s, _f):
        print("\n⚠️ SIGINT, saving...")
        STOP.set()
    signal.signal(signal.SIGINT, sigh)

    shards = [[] for _ in range(NUM_WORKERS)]
    for i, s in enumerate(todo):
        shards[i % NUM_WORKERS].append(s)

    pbar = tqdm(total=len(todo), desc=desc)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [
            ex.submit(worker, i, shards[i], cache, results, pbar)
            for i in range(NUM_WORKERS)
        ]
        for f in futures:
            f.result()
    pbar.close()
    save_cache(cache)
    save_results(results)
    print(f"\n  wall: {time.time()-t0:.1f}s, cached: {len(cache)}")
    print(f"📝 Per-question raw responses: {RESULTS_PATH}")
    return {int(k): v for k, v in cache.items() if int(k) in {s["q_id"] for s in data}}


def main():
    train_data = load_train()
    test_data = load_test()
    _, dev_subset = split_train_dev(train_data)
    print(f"📊 Dev: {len(dev_subset)}, Test: {len(test_data)}")

    # ── DEV ────────────────────────────────────────
    print("\n🔍 Stage 1: dev evaluation")
    dev_preds = run_llm_pick(dev_subset, "dev")
    dev_golds = get_gold_quotes_dict(dev_subset)
    dev_recall = recall_at_k(dev_preds, dev_golds, k=5)
    print(f"\n📈 Dev Recall@5 (Gemma-4-31B direct) = {dev_recall:.4f}")

    ea = error_analysis(dev_preds, dev_golds, k=5)
    print(f"  Perfect: {ea['summary']['perfect_count']}")
    print(f"  Partial: {ea['summary']['partial_count']}")
    print(f"  Zero:    {ea['summary']['zero_count']}")

    # modality split
    text_n, img_n = 0, 0
    for q, lst in dev_preds.items():
        for qid in lst[:5]:
            if qid.lower().startswith("image"):
                img_n += 1
            else:
                text_n += 1
    print(f"  Top-5 modality: text {text_n} ({100*text_n/(text_n+img_n):.1f}%) / img {img_n} ({100*img_n/(text_n+img_n):.1f}%)")

    # ── TEST ─────────────────────────────────────────
    print("\n🔍 Stage 2: test prediction")
    test_preds = run_llm_pick(test_data, "test")

    output_dir = config.get_output_dir("phase_6", "gemma_4_31b_direct")
    submission_path = os.path.join(output_dir, "submission.csv")
    # pad_short=False → worker 已在快取前補滿到 5，這裡留 False 可在補位邏輯出包時立刻發現
    generate_submission(test_preds, test_data, submission_path, pad_short=False)

    # 額外把 test 子集的完整回應 dump 一份到 output_dir，方便跟 submission 一起檢視
    all_results = load_results()
    test_only = [all_results[str(s["q_id"])] for s in test_data if str(s["q_id"]) in all_results]
    results_out = os.path.join(output_dir, "gemma_answering_results.json")
    with open(results_out, "w", encoding="utf-8") as f:
        json.dump(test_only, f, ensure_ascii=False, indent=2)
    print(f"📝 Test answering results: {results_out}")

    metrics = {
        "phase": "phase_6",
        "model": MODEL,
        "approach": "direct 15->5 LLM selection (no retriever)",
        "candidate_caption_source": "original img_description (no VLM)",
        "dev_recall_at_5": dev_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)
    append_leaderboard("Phase 6", f"{MODEL} direct 15->5", dev_recall,
                       note="pure LLM selection, no retriever")

    print(f"\n✅ Phase 6 完成! 輸出: {output_dir}")
    print(f"📤 Submission: {submission_path}")


if __name__ == "__main__":
    main()
