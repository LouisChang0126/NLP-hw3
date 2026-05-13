"""
INLP HW3 - Phase 6: 純 LLM 直接從 15 個候選挑 5 個

Model: google/gemma-4-31b-it (via NVIDIA NIM)
原本的 img_description (不用 VLM 重生成)
多執行緒、可中斷可續跑
"""
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
NUM_WORKERS = 8
MIN_INTERVAL_SEC = 0.3
MAX_TOKENS = 64
TEMPERATURE = 0.0
MAX_RETRIES = 4
CHAR_LIMIT_PER_CANDIDATE = 800
SAVE_EVERY = 50
CACHE_PATH = os.path.join(config.OUTPUT_DIR, "phase_6", "llm_picks_cache.json")


PROMPT_TEMPLATE = """You are selecting evidence to support answering a question about a document. Below are candidate evidence items (text snippets and image descriptions). Pick exactly 5 that are most useful for answering the question.

Question: {question}

Candidates:
{candidates}

Output ONLY 5 IDs separated by spaces, in order of relevance (most relevant first). No explanation. Example: text3 image2 text5 text1 image4"""


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


# ── 解析 LLM 輸出 ────────────────────────────────────
ID_PATTERN = re.compile(r"\b(text\d+|image\d+)\b", re.IGNORECASE)


def parse_ids(text: str, allowed_ids: set, top_k: int = 5) -> List[str]:
    """從 LLM 輸出抽出合法 ID, 保序去重, 不夠 5 個就用候選池補"""
    if not text:
        return []
    found = ID_PATTERN.findall(text)
    seen = set()
    out = []
    for fid in found:
        fid = fid.lower()
        if fid in allowed_ids and fid not in seen:
            out.append(fid)
            seen.add(fid)
            if len(out) >= top_k:
                break
    # 補位 (用 allowed_ids 中尚未挑到的)
    if len(out) < top_k:
        for cid in allowed_ids:
            if cid not in seen:
                out.append(cid)
                seen.add(cid)
                if len(out) >= top_k:
                    break
    return out[:top_k]


# ── prompt 組裝 ──────────────────────────────────────
def format_candidates(cands):
    lines = []
    for c in cands:
        text = (c["text_for_retrieval"] or "").strip()
        if len(text) > CHAR_LIMIT_PER_CANDIDATE:
            text = text[:CHAR_LIMIT_PER_CANDIDATE] + "..."
        lines.append(f"[{c['quote_id']}] {text}")
    return "\n".join(lines)


# ── worker ───────────────────────────────────────────
def worker(worker_id: int, samples: List[dict], cache: Dict[str, List[str]], pbar):
    client = OpenAI(base_url=config.NIM_BASE_URL, api_key=config.NIM_API_KEY)
    last_call = 0.0
    for sample in samples:
        if STOP.is_set():
            return
        q_id_str = str(sample["q_id"])
        if q_id_str in cache and len(cache[q_id_str]) == 5:
            pbar.update(1)
            continue

        cands = build_candidates(sample)
        allowed = {c["quote_id"].lower() for c in cands}
        prompt = PROMPT_TEMPLATE.format(
            question=sample["question"],
            candidates=format_candidates(cands),
        )

        wait = MIN_INTERVAL_SEC - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait + random.random() * 0.2)

        # retry loop
        picks = None
        for attempt in range(MAX_RETRIES):
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                last_call = time.time()
                raw = r.choices[0].message.content or ""
                picks = parse_ids(raw, allowed, top_k=5)
                if len(picks) == 5:
                    break
                tqdm.write(f"[w{worker_id}] q={q_id_str} parsed only {len(picks)}: {raw!r}; retry")
            except Exception as e:
                err = type(e).__name__
                msg = str(e)[:120]
                if "429" in msg or "rate" in msg.lower():
                    backoff = 4 * (2 ** attempt) + random.random()
                    tqdm.write(f"[w{worker_id}] 429, sleep {backoff:.1f}s")
                    time.sleep(backoff)
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(2 + random.random() * 2)
                else:
                    tqdm.write(f"[w{worker_id}] ❌ q={q_id_str} give up: {err}: {msg}")

        if not picks or len(picks) < 5:
            # fallback: 用候選池前 5 個
            picks = [c["quote_id"] for c in cands[:5]]
        cache[q_id_str] = picks

        if len(cache) % SAVE_EVERY == 0:
            save_cache(cache)
        pbar.update(1)


# ── 流程 ─────────────────────────────────────────────
def run_llm_pick(data: List[dict], desc: str) -> Dict[int, List[str]]:
    cache = load_cache()
    todo = [s for s in data if str(s["q_id"]) not in cache or len(cache[str(s["q_id"])]) != 5]
    print(f"📥 cached: {len(cache)},  ⏳ todo: {len(todo)} ({desc})")
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
        futures = [ex.submit(worker, i, shards[i], cache, pbar) for i in range(NUM_WORKERS)]
        for f in futures:
            f.result()
    pbar.close()
    save_cache(cache)
    print(f"\n  wall: {time.time()-t0:.1f}s, cached: {len(cache)}")
    return {int(k): v for k, v in cache.items() if int(k) in {s["q_id"] for s in data}}


def main():
    train_data = load_train()
    test_data = load_test()
    _, dev_subset = split_train_dev(train_data)
    print(f"📊 Dev: {len(dev_subset)}, Test: {len(test_data)}")

    # ── DEV 先 (200 題, 預期 ~10 分鐘 with 8 workers) ─
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
    generate_submission(test_preds, test_data, submission_path)

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
