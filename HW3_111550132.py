"""
RAG submission pipeline.

End-to-end:
  1. 讀 test.jsonl
  2. 對每一筆 q_id 打 Gemma API 取得 top-5 證據排序
  3. 自動重試直到該筆 gold_quotes 至少有 MIN_QUOTES 個 ID
  4. 寫出 submission.csv 與 gemma_answering_results.json
  5. 每一筆都即時存檔，可斷點續跑（再次執行只會重打不足門檻的列）

用法：
  python HW3_111550132.py
  python HW3_111550132.py --input test.jsonl --output submission.csv \
      --backup gemma_answering_results.json --min-quotes 4 --max-attempts 12 \
      --rate 37 --workers 8

設計重點：
  - Windows cp950 控制台相容：強制 stdout 使用 utf-8，純文字訊息（無 emoji）
  - 不會因為 API timeout / 空回應 / 解析失敗而崩潰；該筆會持續重試
  - 多執行緒 + 60 秒滑動視窗 rate limit（預設 37 calls/min，符合 NIM 40 上限）
  - Resume：偵測既有 backup JSON 與 submission.csv，已達門檻者直接沿用
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as _dt
import json
import os
import re
import sys
import threading
import time
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Console / encoding helpers
# ---------------------------------------------------------------------------

def _configure_stdout() -> None:
    """讓中文訊息在 Windows cp950 主控台也能正常輸出。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    """執行緒安全輸出。透過 tqdm.write 避免破壞進度條畫面。"""
    with _LOG_LOCK:
        try:
            tqdm.write(msg)
        except UnicodeEncodeError:
            sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Rate limiter — sliding window across all worker threads
# ---------------------------------------------------------------------------

class RateLimiter:
    """限制 60 秒滑動視窗內 API 呼叫次數，多執行緒安全。"""

    def __init__(self, max_calls: int, window_seconds: float = 60.0) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls 必須 > 0")
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: "collections.deque[float]" = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                wait_until = self._timestamps[0] + self.window
            sleep_for = max(0.05, wait_until - time.monotonic())
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "google/gemma-4-31b-it"


def slugify_model_name(name: str) -> str:
    """把 model id (例如 'google/gemma-4-31b-it') 轉成可當資料夾名稱的 slug。"""
    tail = name.rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", tail).strip("_")
    return slug or "model"


def default_output_dir(model_name: str = MODEL_NAME) -> str:
    """outputs/phase_6/{model_slug}_{mmddhhmm}"""
    stamp = _dt.datetime.now().strftime("%m%d%H%M")
    return os.path.join("outputs", "phase_6", f"{slugify_model_name(model_name)}_{stamp}")


def load_api_key(filepath: str) -> str:
    if not os.path.exists(filepath):
        log(f"[FATAL] 找不到 API Key 檔案 '{filepath}'")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        log(f"[FATAL] API Key 檔案 '{filepath}' 是空的")
        sys.exit(1)
    return key


def query_gemma(prompt: str, api_key: str, timeout: float) -> str:
    """打一次 API，永遠回傳字串；失敗時回傳 '[Error] ...'。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 0.1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            return content.strip() if content else "[Error] 模型回傳了空白內容"
        return "[Error] API 回應格式不符預期 (無 choices)"
    except requests.exceptions.RequestException as exc:
        return f"[Error] API 請求失敗: {exc}"
    except json.JSONDecodeError:
        return "[Error] 無法解析 API 傳回的 JSON"
    except Exception as exc:  # pragma: no cover
        return f"[Error] 發生未知例外: {exc}"


# ---------------------------------------------------------------------------
# Prompt / parsing
# ---------------------------------------------------------------------------

ID_PATTERN = re.compile(r"(text|image)[\s\-_]*(\d+)")


def extract_top_5_ids(response_text: str, valid_quote_ids: Set[str]) -> List[str]:
    if not response_text or response_text.startswith("[Error]"):
        return []
    seen: Set[str] = set()
    valid_ids: List[str] = []
    for prefix, num in ID_PATTERN.findall(response_text.lower()):
        normalized = f"{prefix}{int(num)}"
        if normalized in valid_quote_ids and normalized not in seen:
            valid_ids.append(normalized)
            seen.add(normalized)
            if len(valid_ids) == 5:
                break
    return valid_ids


def build_prompt(question: str, corpus_dict: Dict[str, str]) -> str:
    # evidence_str = "\n".join(f"[{qid}]: {content}" for qid, content in corpus_dict.items())
    evidence_str = "\n".join(f"<{qid}>{content}</{qid}>" for qid, content in corpus_dict.items())
    return (
        "You are an expert retrieval assistant. I will provide you with a question and a list of evidence items. "
        "Your task is to analyze the evidence and extract the top 5 most important evidence IDs that best answer the question.\n\n"
        f"Question: {question}\n\n"
        f"Evidence Items:\n{evidence_str}\n\n"
        "You MUST extract and rank EXACTLY 5 evidence IDs in descending order of importance. "
        "Even if you think fewer than 5 items are relevant, you MUST fill all 5 spots with your best guesses. "
        "DO NOT output fewer than 5 IDs. Output ONLY the 5 IDs separated by commas (e.g., text1, image3, text5, image2, text10)."
    )


_TEXT_MODALITIES = {"text"}
_IMAGE_MODALITIES = {"table", "figure", "chart", "image"}


def collect_corpus(sample: dict) -> Dict[str, str]:
    """組成候選證據池。會依 evidence_modality_type 剔除不可能成為答案的型態。

    觀察 train.jsonl 後得到的硬規則：
      - evidence_modality_type 只含 text → gold 100% 只有 text*
      - evidence_modality_type 只含 image-like (table/figure/chart) → gold 100% 只有 image*
    因此在純單一型態的題目，把另一型態的候選整批 prune 掉，可大幅降低 distractor。
    若 modality 為空或同時含兩者，則完整保留。
    """
    mods = set(sample.get("evidence_modality_type") or [])
    has_text = bool(mods & _TEXT_MODALITIES)
    has_image = bool(mods & _IMAGE_MODALITIES)
    # 兩者皆無（modality 缺失或全是未知型態）時 → 不過濾，安全 fallback
    keep_text = has_text or not (has_text or has_image)
    keep_image = has_image or not (has_text or has_image)

    corpus: Dict[str, str] = {}
    if keep_text:
        for tq in sample.get("text_quotes", []) or []:
            corpus[tq["quote_id"]] = tq["text"]
    if keep_image:
        for iq in sample.get("img_quotes", []) or []:
            corpus[iq["quote_id"]] = iq["img_description"]

    # 守備：若 filter 後變空但原本有候選 → 回退到不過濾，避免送出空 prompt
    if not corpus:
        for tq in sample.get("text_quotes", []) or []:
            corpus[tq["quote_id"]] = tq["text"]
        for iq in sample.get("img_quotes", []) or []:
            corpus[iq["quote_id"]] = iq["img_description"]
    return corpus


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        log(f"[FATAL] 找不到輸入檔 '{path}'")
        sys.exit(1)
    samples: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log(f"[WARN] 第 {line_no} 行 JSON 解析失敗，跳過: {exc}")
    return samples


def load_existing_results(path: str) -> Dict[int, dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(r["q_id"]): r for r in data if "q_id" in r}
    except Exception as exc:
        log(f"[WARN] 讀取 {path} 失敗，視為空: {exc}")
        return {}


def _atomic_replace(src: str, dst: str, max_retries: int = 8) -> None:
    """Windows 上 os.replace 偶爾會被防毒/索引器/IDE 短暫鎖住而丟 PermissionError，
    這裡做指數型 backoff 重試 (100ms ~ 12.8s)。"""
    delay = 0.1
    last_exc: Optional[BaseException] = None
    for _ in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
    raise last_exc  # type: ignore[misc]


def save_outputs(
    output_csv: str,
    backup_json: str,
    submission_rows: List[dict],
    results_by_id: Dict[int, dict],
) -> None:
    """寫入 CSV 與 JSON。使用臨時檔 + atomic rename，避免寫到一半被中斷造成損毀。"""
    tmp_csv = output_csv + ".tmp"
    tmp_json = backup_json + ".tmp"

    pd.DataFrame(submission_rows).to_csv(tmp_csv, index=False)
    _atomic_replace(tmp_csv, output_csv)

    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(list(results_by_id.values()), f, ensure_ascii=False, indent=2)
    _atomic_replace(tmp_json, backup_json)


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def process_sample(
    sample: dict,
    api_key: str,
    min_quotes: int,
    max_attempts: int,
    rate_limiter: RateLimiter,
    previous_best: Optional[dict] = None,
) -> dict:
    """重試直到至少達到 min_quotes，或耗盡 max_attempts；回傳 result dict。"""
    q_id = sample["q_id"]
    question = sample["question"]
    corpus = collect_corpus(sample)
    valid_ids = set(corpus.keys())
    prompt = build_prompt(question, corpus)
    needed = min(min_quotes, max(1, len(valid_ids)))

    if previous_best:
        best_quotes = list(previous_best.get("predicted_quotes") or [])
        last_response = previous_best.get("model_raw_response", "")
    else:
        best_quotes = []
        last_response = ""

    for attempt in range(1, max_attempts + 1):
        rate_limiter.acquire()

        timeout = 180.0 if attempt <= 2 else min(300.0 + 60.0 * (attempt - 3), 600.0)
        response_text = query_gemma(prompt, api_key, timeout)
        last_response = response_text

        quotes = extract_top_5_ids(response_text, valid_ids)
        if len(quotes) > len(best_quotes):
            best_quotes = quotes

        # 只在錯誤或不達標時輸出，避免洗版 tqdm bar
        if response_text.startswith("[Error]"):
            preview = response_text[:120].replace("\n", " ")
            log(f"[WARN] q_id={q_id} attempt {attempt} error: {preview}")
        elif len(best_quotes) < needed and attempt == max_attempts:
            log(f"[WARN] q_id={q_id} 用盡 {max_attempts} 次仍只有 {len(best_quotes)} IDs (門檻 {needed})")

        if len(best_quotes) >= needed:
            break

        # 指數型 backoff，最高 30 秒（請求節流仍由 RateLimiter 保證）
        backoff = min(2.0 * attempt, 30.0)
        time.sleep(backoff)

    return {
        "q_id": q_id,
        "question": question,
        "prompt": prompt,
        "predicted_quotes": best_quotes,
        "model_raw_response": last_response,
    }


def _build_submission_rows(samples: List[dict], results_by_id: Dict[int, dict]) -> List[dict]:
    rows = []
    for s in samples:
        q_id = s["q_id"]
        quotes = list((results_by_id.get(q_id) or {}).get("predicted_quotes") or [])
        rows.append({"q_id": q_id, "gold_quotes": " ".join(quotes)})
    return rows


def run_pipeline(
    input_path: str,
    output_csv: str,
    backup_json: str,
    api_key_path: str,
    min_quotes: int,
    max_attempts: int,
    rate_per_minute: int,
    workers: int,
    save_every: int,
) -> None:
    api_key = load_api_key(api_key_path)
    samples = load_jsonl(input_path)
    log(f"載入 {len(samples)} 筆樣本：{input_path}")

    existing = load_existing_results(backup_json)
    if existing:
        log(f"偵測到既有結果 {len(existing)} 筆：{backup_json}")

    results_by_id: Dict[int, dict] = dict(existing)
    results_lock = threading.Lock()
    save_lock = threading.Lock()

    # 分流：已達門檻 → skip；其餘 → 丟到 thread pool
    to_process: List[dict] = []
    skipped = 0
    for sample in samples:
        q_id = sample["q_id"]
        prev = results_by_id.get(q_id)
        prev_quotes = list(prev.get("predicted_quotes") or []) if prev else []
        needed = min(min_quotes, max(1, len(collect_corpus(sample))))
        if prev and len(prev_quotes) >= needed:
            skipped += 1
        else:
            to_process.append(sample)

    log(f"沿用 {skipped} 筆；需要重打 {len(to_process)} 筆")
    log(f"並行 worker 數: {workers}，全域 rate limit: {rate_per_minute} calls / 60s")

    rate_limiter = RateLimiter(max_calls=rate_per_minute, window_seconds=60.0)

    completed = 0
    total = len(to_process)

    def _worker(sample: dict) -> int:
        q_id = sample["q_id"]
        with results_lock:
            prev = results_by_id.get(q_id)
        prev_quotes = list((prev or {}).get("predicted_quotes") or [])
        try:
            result = process_sample(
                sample,
                api_key=api_key,
                min_quotes=min_quotes,
                max_attempts=max_attempts,
                rate_limiter=rate_limiter,
                previous_best=prev,
            )
        except Exception as exc:  # pragma: no cover
            log(f"[WARN] q_id={q_id} 發生例外，保留舊值: {exc}")
            result = prev or {
                "q_id": q_id,
                "question": sample.get("question", ""),
                "prompt": "",
                "predicted_quotes": prev_quotes,
                "model_raw_response": "",
            }
        with results_lock:
            results_by_id[q_id] = result
        return len(list(result.get("predicted_quotes") or []))

    below_threshold: List[int] = []

    if to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor, \
                tqdm(total=len(samples), initial=skipped, desc="處理", unit="q",
                     dynamic_ncols=True) as bar:
            bar.set_postfix(skip=skipped, low=0)
            future_to_sample = {
                executor.submit(_worker, sample): sample for sample in to_process
            }
            for future in concurrent.futures.as_completed(future_to_sample):
                sample = future_to_sample[future]
                q_id = sample["q_id"]
                try:
                    n_ids = future.result()
                except Exception as exc:  # pragma: no cover
                    log(f"[WARN] q_id={q_id} worker 失敗: {exc}")
                    n_ids = 0

                needed = min(min_quotes, max(1, len(collect_corpus(sample))))
                if n_ids < needed:
                    below_threshold.append(q_id)

                completed += 1
                bar.set_postfix(skip=skipped, low=len(below_threshold))
                bar.update(1)

                if completed % save_every == 0 or completed == total:
                    with save_lock, results_lock:
                        rows = _build_submission_rows(samples, results_by_id)
                        try:
                            save_outputs(output_csv, backup_json, rows, results_by_id)
                        except Exception as exc:
                            # 寫檔暫時失敗（多半是 Windows 檔案鎖）→ 下一筆完成時會再試
                            log(f"[WARN] 寫檔失敗，跳過此次 flush，下次再試: {exc}")

    # 最終再存一次，確保即使全部 skip 也會寫出檔
    with save_lock, results_lock:
        rows = _build_submission_rows(samples, results_by_id)
        try:
            save_outputs(output_csv, backup_json, rows, results_by_id)
        except Exception as exc:
            log(f"[WARN] 最終寫檔失敗，請手動重跑: {exc}")

    log("\n" + "=" * 60)
    log(f"完成。樣本總數 {len(samples)}：沿用 {skipped} 筆 / 重打 {len(to_process)} 筆")
    log(f"輸出：{output_csv}")
    log(f"備份：{backup_json}")
    if below_threshold:
        log(f"[WARN] 仍未達門檻 {min_quotes} 的 q_id 共 {len(below_threshold)} 筆：{below_threshold[:30]}"
            + (" ..." if len(below_threshold) > 30 else ""))
        log("       直接再執行一次本腳本即可從這些 q_id 繼續重試。")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG submission pipeline (NVIDIA Gemma)")
    parser.add_argument("--input", default=os.path.join("data", "test.jsonl"),
                        help="輸入的 JSONL 檔 (預設 data/test.jsonl)")
    parser.add_argument("--output-dir", default=None,
                        help="輸出資料夾 (預設 outputs/phase_6/{model_slug}_{mmddhhmm})")
    parser.add_argument("--output", default="submission.csv",
                        help="輸出 CSV 檔名，會放進 --output-dir (預設 submission.csv)")
    parser.add_argument("--backup", default="gemma_answering_results.json",
                        help="完整回應備份 JSON，會放進 --output-dir (預設 gemma_answering_results.json)")
    parser.add_argument("--api-key", default="api_key.txt", help="API key 檔路徑 (預設 api_key.txt)")
    parser.add_argument("--min-quotes", type=int, default=5,
                        help="每筆至少要拿到的 ID 數，未達會持續重試 (預設 5)")
    parser.add_argument("--max-attempts", type=int, default=12,
                        help="同一筆最多重試幾次 (預設 12)")
    parser.add_argument("--rate", type=int, default=37,
                        help="60 秒滑動視窗內最多 API 呼叫次數 (預設 37，NIM 上限為 40)")
    parser.add_argument("--workers", type=int, default=4,
                        help="並行 worker 執行緒數 (預設 4)")
    parser.add_argument("--save-every", type=int, default=1,
                        help="每完成 N 筆就 flush 到磁碟 (預設 1，每筆都即時寫檔最安全)")
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()

    output_dir = args.output_dir or default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    output_csv = args.output if os.path.isabs(args.output) else os.path.join(output_dir, args.output)
    backup_json = args.backup if os.path.isabs(args.backup) else os.path.join(output_dir, args.backup)

    log(f"輸出資料夾: {output_dir}")

    run_pipeline(
        input_path=args.input,
        output_csv=output_csv,
        backup_json=backup_json,
        api_key_path=args.api_key,
        min_quotes=args.min_quotes,
        max_attempts=args.max_attempts,
        rate_per_minute=args.rate,
        workers=args.workers,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
