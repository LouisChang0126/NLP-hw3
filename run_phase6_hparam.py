"""
Phase 6 hyperparameter sweep on dev set.

對 google/gemma-4-31b-it 直接 15→5 LLM 選證測試多個 prompt/sampling 設定。
每個 variant 各自一份 cache, 結束後印出 dev recall 對照表。
"""
import json
import os
import re
import random
import signal
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

from tqdm import tqdm
from openai import OpenAI

import config
from dataset import (
    load_train, build_candidates, split_train_dev, get_gold_quotes_dict,
)
from evaluation import recall_at_k, error_analysis


MODEL = "google/gemma-4-31b-it"
NUM_WORKERS = 8
MIN_INTERVAL_SEC = 0.3
MAX_RETRIES = 4
SAVE_EVERY = 25
HPARAM_DIR = os.path.join(config.OUTPUT_DIR, "phase_6", "hparam")
BASELINE_SRC = os.path.join(config.OUTPUT_DIR, "phase_6", "llm_picks_cache.json")


# ── Prompts ────────────────────────────────────────────
PROMPT_BASELINE = """You are selecting evidence to support answering a question about a document. Below are candidate evidence items (text snippets and image descriptions). Pick exactly 5 that are most useful for answering the question.

Question: {question}

Candidates:
{candidates}

Output ONLY 5 IDs separated by spaces, in order of relevance (most relevant first). No explanation. Example: text3 image2 text5 text1 image4"""


PROMPT_MODALITY = """You are an expert at finding evidence in mixed text+image documents. You are given a question and 15 candidate quotes — a mix of text snippets and image descriptions. Select exactly 5 quotes that, together, give the BEST evidence to answer the question.

Important guidance:
- Many questions need BOTH text and image evidence (e.g. a chart plus its label, or a number from a table plus surrounding context). Consider both modalities.
- Image descriptions describe figures, charts, and tables, and may contain the numerical or visual data that directly answers the question. Do not skip them.
- Pick complementary evidence — avoid 5 near-duplicates.

Question: {question}

Candidates:
{candidates}

Output ONLY 5 IDs separated by spaces, ordered by relevance (most relevant first). No explanation. Example: text3 image2 text5 text1 image4"""


PROMPT_COT = """You are selecting evidence to support answering a question about a document. Below are 15 candidate quotes (text snippets and image descriptions). Pick exactly 5 that are most useful for answering the question.

Question: {question}

Candidates:
{candidates}

Briefly reason in ONE short line (≤25 words) about what info is needed and which modalities likely hold it.
Then on a NEW final line output exactly 5 IDs separated by spaces, ordered by relevance.
Final line example:
text3 image2 text5 text1 image4"""


# ── ID parsing ─────────────────────────────────────────
ID_PATTERN = re.compile(r"\b(text\d+|image\d+)\b", re.IGNORECASE)


def parse_ids(raw: str, allowed: set, top_k: int = 5,
              prefer_last_line: bool = False) -> List[str]:
    if not raw:
        return []
    scope = raw
    if prefer_last_line:
        lines = [l for l in raw.strip().splitlines() if l.strip()]
        if lines:
            last = lines[-1]
            if len(ID_PATTERN.findall(last)) >= top_k:
                scope = last
    found = ID_PATTERN.findall(scope)
    seen, out = set(), []
    for fid in found:
        fid = fid.lower()
        if fid in allowed and fid not in seen:
            out.append(fid); seen.add(fid)
            if len(out) >= top_k:
                break
    if len(out) < top_k:
        for cid in allowed:
            if cid not in seen:
                out.append(cid); seen.add(cid)
                if len(out) >= top_k:
                    break
    return out[:top_k]


def format_candidates(cands, char_limit: int) -> str:
    out = []
    for c in cands:
        t = (c["text_for_retrieval"] or "").strip()
        if char_limit > 0 and len(t) > char_limit:
            t = t[:char_limit] + "..."
        out.append(f"[{c['quote_id']}] {t}")
    return "\n".join(out)


# ── Variant config ─────────────────────────────────────
@dataclass
class Variant:
    name: str
    prompt: str
    char_limit: int = 800
    temperature: float = 0.0
    max_tokens: int = 64
    n_samples: int = 1
    prefer_last_line: bool = False


VARIANTS: List[Variant] = [
    Variant("baseline",          PROMPT_BASELINE, char_limit=800,  temperature=0.0, max_tokens=64,  n_samples=1),
    Variant("long_ctx_2000",     PROMPT_BASELINE, char_limit=2000, temperature=0.0, max_tokens=64,  n_samples=1),
    Variant("modality_prompt",   PROMPT_MODALITY, char_limit=1500, temperature=0.0, max_tokens=64,  n_samples=1),
    Variant("cot_brief",         PROMPT_COT,      char_limit=1500, temperature=0.0, max_tokens=192, n_samples=1, prefer_last_line=True),
    Variant("self_consistency3", PROMPT_MODALITY, char_limit=1500, temperature=0.5, max_tokens=64,  n_samples=3),
]


# ── persistence ────────────────────────────────────────
SAVE_LOCKS: Dict[str, threading.Lock] = {}
STOP = threading.Event()


def cache_path(name: str) -> str:
    return os.path.join(HPARAM_DIR, f"{name}_cache.json")


def load_cache(name: str) -> dict:
    p = cache_path(name)
    if not os.path.exists(p):
        return {}
    with open(p, "r") as f:
        return json.load(f)


def save_cache(name: str, cache: dict):
    os.makedirs(HPARAM_DIR, exist_ok=True)
    lock = SAVE_LOCKS.setdefault(name, threading.Lock())
    with lock:
        tmp = cache_path(name) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, cache_path(name))


def bootstrap_baseline(dev_subset):
    dst = cache_path("baseline")
    if os.path.exists(dst):
        return
    if not os.path.exists(BASELINE_SRC):
        return
    with open(BASELINE_SRC, "r") as f:
        src = json.load(f)
    dev_ids = {str(s["q_id"]) for s in dev_subset}
    snap = {k: v for k, v in src.items() if k in dev_ids and len(v) == 5}
    os.makedirs(HPARAM_DIR, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(snap, f)
    print(f"  bootstrapped baseline cache: {len(snap)} entries from {BASELINE_SRC}")


# ── per-question prediction ────────────────────────────
def call_llm(client, prompt, max_tokens, temperature) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return r.choices[0].message.content or ""


def predict_one(client, variant: Variant, sample: dict, worker_id: int) -> List[str]:
    cands = build_candidates(sample)
    allowed = {c["quote_id"].lower() for c in cands}
    prompt = variant.prompt.format(
        question=sample["question"],
        candidates=format_candidates(cands, variant.char_limit),
    )

    all_picks: List[List[str]] = []
    for _ in range(variant.n_samples):
        picks = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = call_llm(client, prompt, variant.max_tokens, variant.temperature)
                picks = parse_ids(raw, allowed, 5, variant.prefer_last_line)
                if len(picks) == 5:
                    break
                tqdm.write(f"[w{worker_id}/{variant.name}] q={sample['q_id']} parsed {len(picks)}: {raw!r}; retry")
            except Exception as e:
                msg = str(e)[:120]
                if "429" in msg or "rate" in msg.lower():
                    backoff = 4 * (2 ** attempt) + random.random()
                    tqdm.write(f"[w{worker_id}/{variant.name}] 429, sleep {backoff:.1f}s")
                    time.sleep(backoff)
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(2 + random.random() * 2)
                else:
                    tqdm.write(f"[w{worker_id}/{variant.name}] ❌ q={sample['q_id']} {type(e).__name__}: {msg}")
        if picks and len(picks) == 5:
            all_picks.append(picks)

    if not all_picks:
        return [c["quote_id"] for c in cands[:5]]

    if len(all_picks) == 1:
        return all_picks[0]

    # Self-consistency: rank by (appearance_count, sum_inverted_rank)
    counts: Counter = Counter()
    score: Dict[str, float] = {}
    for picks in all_picks:
        for pos, qid in enumerate(picks):
            counts[qid] += 1
            score[qid] = score.get(qid, 0.0) + (5 - pos)
    ranked = sorted(counts.keys(), key=lambda q: (-counts[q], -score[q]))
    return ranked[:5]


# ── worker ─────────────────────────────────────────────
def worker(worker_id: int, variant: Variant, samples: List[dict],
           cache: Dict[str, List[str]], pbar: tqdm):
    client = OpenAI(base_url=config.NIM_BASE_URL, api_key=config.NIM_API_KEY)
    last_call = 0.0
    for sample in samples:
        if STOP.is_set():
            return
        q_id_str = str(sample["q_id"])
        if q_id_str in cache and len(cache[q_id_str]) == 5:
            pbar.update(1); continue

        wait = MIN_INTERVAL_SEC - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait + random.random() * 0.2)
        picks = predict_one(client, variant, sample, worker_id)
        last_call = time.time()

        cache[q_id_str] = picks
        if len(cache) % SAVE_EVERY == 0:
            save_cache(variant.name, cache)
        pbar.update(1)


def run_variant(variant: Variant, data: List[dict]) -> Dict[int, List[str]]:
    print(f"\n=== Variant: {variant.name} ===")
    print(f"  char_limit={variant.char_limit}, T={variant.temperature}, "
          f"n_samples={variant.n_samples}, max_tokens={variant.max_tokens}")
    cache = load_cache(variant.name)
    todo = [s for s in data if str(s["q_id"]) not in cache or len(cache[str(s["q_id"])]) != 5]
    print(f"  cached: {len(cache)}, todo: {len(todo)}")
    if todo:
        shards = [[] for _ in range(NUM_WORKERS)]
        for i, s in enumerate(todo):
            shards[i % NUM_WORKERS].append(s)
        pbar = tqdm(total=len(todo), desc=variant.name)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = [ex.submit(worker, i, variant, shards[i], cache, pbar)
                       for i in range(NUM_WORKERS)]
            for f in futures:
                f.result()
        pbar.close()
        save_cache(variant.name, cache)
        print(f"  wall: {time.time()-t0:.1f}s")
    return {int(k): v for k, v in cache.items() if int(k) in {s["q_id"] for s in data}}


def main():
    train = load_train()
    _, dev = split_train_dev(train)
    golds = get_gold_quotes_dict(dev)
    print(f"📊 Dev: {len(dev)} samples")

    bootstrap_baseline(dev)

    def sigh(_s, _f):
        print("\n⚠️ SIGINT — saving caches"); STOP.set()
    signal.signal(signal.SIGINT, sigh)

    results: Dict[str, dict] = {}
    for v in VARIANTS:
        try:
            preds = run_variant(v, dev)
            r = recall_at_k(preds, golds, k=5)
            ea = error_analysis(preds, golds, k=5)["summary"]
            results[v.name] = {"recall": r, "ea": ea}
            print(f"  → {v.name}: Recall@5 = {r:.4f}  (perfect {ea['perfect_count']} / partial {ea['partial_count']} / zero {ea['zero_count']})")
        except Exception as e:
            print(f"  ❌ {v.name} failed: {e}")
            results[v.name] = None

    print("\n" + "=" * 70)
    print("SUMMARY  (Dev Recall@5)")
    print("=" * 70)
    base = (results.get("baseline") or {}).get("recall")
    print(f"  {'variant':<22} {'recall':>8}  {'Δ vs base (pp)':>16}  perfect/partial/zero")
    for v in VARIANTS:
        info = results.get(v.name)
        if not info:
            print(f"  {v.name:<22} FAILED")
            continue
        r = info["recall"]; ea = info["ea"]
        delta = f"{(r - base) * 100:+.2f}" if base is not None else "  —  "
        print(f"  {v.name:<22} {r:.4f}   {delta:>10}      {ea['perfect_count']}/{ea['partial_count']}/{ea['zero_count']}")

    # save summary
    os.makedirs(HPARAM_DIR, exist_ok=True)
    with open(os.path.join(HPARAM_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {os.path.join(HPARAM_DIR, 'summary.json')}")


if __name__ == "__main__":
    main()
