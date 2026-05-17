"""Re-run LLM on a sample of dev questions, capture raw output, and
count how many cases have <5 valid IDs (triggering parse_ids' internal padding).

This is the REAL fallback measurement — outer fallback (picks = first5 of
cands) is 0% but parse_ids' inner padding may still kick in.
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("CHAR_LIMIT_PER_CANDIDATE", "1500")
os.environ.setdefault("N_CTX", "8192")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset import load_train, build_candidates, split_train_dev
from run_phase6_direct_llm import PROMPT_TEMPLATE, format_candidates, ID_PATTERN
from run_phase6_local import build_qwen_prompt, download_gguf, load_llm

# Tweak this if you want larger sample
N_SAMPLES = 50


def count_valid_unique_ids(raw, allowed):
    found = ID_PATTERN.findall(raw or "")
    seen, count = set(), 0
    for fid in found:
        fid = fid.lower()
        if fid in allowed and fid not in seen:
            seen.add(fid)
            count += 1
    return count, len(found)


def main():
    train = load_train()
    _, dev = split_train_dev(train)
    sample = dev[:N_SAMPLES]
    print(f"Diagnostic: re-running LLM on first {len(sample)} dev questions")

    gguf = download_gguf()
    llm = load_llm(gguf)

    rows = []
    n_padded = 0
    t0 = time.time()
    for s in sample:
        cands = build_candidates(s)
        allowed = {c["quote_id"].lower() for c in cands}
        prompt = PROMPT_TEMPLATE.format(
            question=s["question"],
            candidates=format_candidates(cands),
        )
        full = build_qwen_prompt(prompt)
        r = llm.create_completion(
            prompt=full, max_tokens=64, temperature=0.0,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        raw = (r["choices"][0]["text"] or "")
        n_valid, n_found = count_valid_unique_ids(raw, allowed)
        padded = n_valid < 5
        if padded:
            n_padded += 1
        rows.append({
            "q_id": s["q_id"],
            "raw": raw.strip()[:200],
            "regex_matches": n_found,
            "valid_unique_ids": n_valid,
            "needs_padding": padded,
        })

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s ({dt/len(sample):.2f}s / sample)\n")
    print(f"{'q_id':>6} {'n_match':>8} {'n_valid':>8} {'pad?':>5}  raw")
    print("-" * 90)
    for r in rows:
        flag = "YES" if r["needs_padding"] else "."
        print(f"{r['q_id']:>6} {r['regex_matches']:>8} {r['valid_unique_ids']:>8} {flag:>5}  {r['raw']}")

    print(f"\n=== Summary ===")
    print(f"Padded by parse_ids: {n_padded}/{len(sample)} ({100*n_padded/len(sample):.2f}%)")

    out = os.path.join(config.OUTPUT_DIR, "phase_6_local", "padding_diagnostic.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "sample_size": len(sample),
            "n_padded": n_padded,
            "padding_rate": n_padded / len(sample),
            "rows": rows,
        }, f, indent=2)
    print(f"💾 saved → {out}")


if __name__ == "__main__":
    main()
