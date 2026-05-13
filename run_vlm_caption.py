"""
INLP HW3 - VLM Re-Captioning via NVIDIA NIM API
    model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning

針對 retrieval 任務重新生成更詳盡的 image caption：
  - 含 OCR / 表格數字 / 圖表標籤與趨勢 / 命名實體
  - 結果存 data/img_captions_vlm.json (incremental save, resumable)
  - 多執行緒 + 每 worker 延遲 + 429 退避
"""
import base64
import json
import os
import random
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from tqdm import tqdm
from openai import OpenAI

import config
from dataset import load_train, load_test


# ── 設定 ──────────────────────────────────────────────
NIM_API_KEY = "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"
MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
OUTPUT_FILE = os.path.join(config.DATA_DIR, "img_captions_vlm.json")

NUM_WORKERS = 6              # 並行 worker 數
MIN_INTERVAL_SEC = 0.5       # 每個 worker 兩次呼叫之間至少間隔
MAX_TOKENS = 1024            # enable_thinking=False 下 1024 通常夠用
TEMPERATURE = 0.2
SAVE_EVERY = 25
MAX_RETRIES = 4
RETRY_BASE = 4.0             # 429 / timeout: base * 2^attempt


PROMPT = (
    "Describe this document image for retrieval in 3-5 sentences. "
    "Include exact numbers, units, labels, named entities (companies, "
    "products, dates), and the key trend or takeaway. "
    "Reply with only the caption — no preamble."
)


def clean_caption(s: str) -> str:
    """剔除多模態模型偶爾輸出的 <point> token / 標記"""
    if not s:
        return ""
    s = re.sub(r"<point>\[?\(?[^)]*?\)?\]?:?", "", s)
    s = re.sub(r"</?point>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── 持久化 (lock + atomic write) ─────────────────────
SAVE_LOCK = threading.Lock()
STOP_FLAG = threading.Event()


def load_cache() -> Dict[str, str]:
    if not os.path.exists(OUTPUT_FILE):
        return {}
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache: Dict[str, str]):
    with SAVE_LOCK:
        tmp = OUTPUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
        os.replace(tmp, OUTPUT_FILE)


# ── API ──────────────────────────────────────────────
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_nim(client: OpenAI, b64: str) -> str:
    r = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = r.choices[0].message
    return clean_caption((msg.content or "").strip())


def worker(worker_id: int, img_paths: List[str], cache: Dict[str, str], pbar):
    """每個 worker 是一條獨立的 thread，使用獨立 OpenAI client"""
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NIM_API_KEY)
    last_call = 0.0

    for rel_path in img_paths:
        if STOP_FLAG.is_set():
            return
        # 兩次呼叫之間 enforce 間隔
        wait = MIN_INTERVAL_SEC - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait + random.random() * 0.3)

        full_path = os.path.join(config.IMAGE_DIR, os.path.basename(rel_path))
        if not os.path.exists(full_path):
            pbar.update(1)
            continue

        # ── retry loop ─────────────────────────────
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                b64 = encode_image(full_path)
                caption = call_nim(client, b64)
                if not caption:
                    # 模型偶爾回空字串 (reasoning 用完 tokens) — 視為失敗重試
                    raise RuntimeError("empty content")
                last_call = time.time()
                cache[rel_path] = caption
                # 增量存檔
                if len(cache) % SAVE_EVERY == 0:
                    save_cache(cache)
                success = True
                break
            except Exception as e:
                err = type(e).__name__
                msg = str(e)[:120]
                if "429" in msg or "rate" in msg.lower():
                    backoff = RETRY_BASE * (2 ** attempt) + random.random()
                    tqdm.write(f"  [w{worker_id}] 429, sleep {backoff:.1f}s")
                    time.sleep(backoff)
                elif attempt < MAX_RETRIES - 1:
                    backoff = 2.0 + random.random() * 2
                    tqdm.write(f"  [w{worker_id}] {err}: {msg} → retry {attempt+1} in {backoff:.1f}s")
                    time.sleep(backoff)
                else:
                    tqdm.write(f"  [w{worker_id}] ❌ giving up on {rel_path}: {err}: {msg}")

        if not success:
            cache[rel_path] = ""    # 標記試過但失敗，避免下次重跑
        pbar.update(1)


def collect_unique_image_paths() -> List[str]:
    train = load_train()
    test = load_test()
    paths = set()
    for s in train + test:
        for iq in s.get("img_quotes", []):
            p = iq.get("img_path")
            if p:
                paths.add(p)
    return sorted(paths)


def main():
    all_paths = collect_unique_image_paths()
    print(f"📊 Unique image paths: {len(all_paths)}")

    cache = load_cache()
    todo = [p for p in all_paths if p not in cache or cache[p] == ""]
    print(f"📥 Already captioned: {sum(1 for v in cache.values() if v)}")
    print(f"⏳ Todo:              {len(todo)}")

    if not todo:
        print("✅ Nothing to do")
        return

    # graceful Ctrl-C
    def sig_handler(signum, frame):
        print("\n⚠️ SIGINT — saving cache and exiting...")
        STOP_FLAG.set()
    signal.signal(signal.SIGINT, sig_handler)

    pbar = tqdm(total=len(todo), desc="VLM caption")

    # 平均切分給 workers
    shards = [[] for _ in range(NUM_WORKERS)]
    for i, p in enumerate(todo):
        shards[i % NUM_WORKERS].append(p)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [ex.submit(worker, i, shards[i], cache, pbar) for i in range(NUM_WORKERS)]
        try:
            for f in futures:
                f.result()
        except KeyboardInterrupt:
            STOP_FLAG.set()

    pbar.close()
    save_cache(cache)
    done = sum(1 for v in cache.values() if v)
    print(f"\n✅ Done. {done}/{len(all_paths)} captioned in {time.time()-t0:.1f}s")
    print(f"📤 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
