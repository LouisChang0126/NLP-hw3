"""Estimate fallback rate of Phase 6 local CL=1500.

Fallback signature: cached picks == first-5 of build_candidates() output
(exact IDs AND order). False-positive rate is negligible (the LLM is asked
to rank by relevance, not by candidate order).

Edge cases flagged:
- Samples with <5 candidates: fallback is forced regardless of LLM output.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset import load_train, load_test, build_candidates


def first5(sample):
    return [c["quote_id"] for c in build_candidates(sample)[:5]]


def n_cands(sample):
    return len(build_candidates(sample))


def analyze(samples, cache, label):
    fb = forced = ll5 = 0
    total = 0
    not_in_cache = 0
    for s in samples:
        k = str(s["q_id"])
        if k not in cache:
            not_in_cache += 1
            continue
        total += 1
        picks = cache[k]
        n_c = n_cands(s)
        f5 = first5(s)
        if n_c < 5:
            forced += 1
        if picks == f5:
            fb += 1
        if n_c < 5 and picks == f5:
            ll5 += 1   # forced fallback (LLM had no chance)

    print(f"\n── {label}  (n={total}, missing={not_in_cache}) ──")
    pct = lambda x: f"{100*x/max(total,1):.2f}%"
    print(f"  Fallback (picks==first5 of build):  {fb} / {total}  ({pct(fb)})")
    print(f"    of which forced (<5 cands):       {ll5}")
    print(f"    of which LLM-output<5 (real bad): {fb - ll5}  ({pct(fb-ll5)})")
    print(f"  Samples with <5 candidates total:   {forced}  ({pct(forced)})")


def main():
    train = load_train()
    test = load_test()

    with open(os.path.join(config.SPLITS_DIR, "dev_ids.json")) as f:
        dev_ids = set(json.load(f))
    with open(os.path.join(config.SPLITS_DIR, "large_dev_ids.json")) as f:
        large_dev_ids = set(json.load(f))

    dev = [s for s in train if s["q_id"] in dev_ids]
    large_dev = [s for s in train if s["q_id"] in large_dev_ids]

    # cache files
    cache_small_test = json.load(open(
        os.path.join(config.OUTPUT_DIR, "phase_6_local",
                     "llm_picks_cache_cl1500.json")
    ))
    cache_large_dev = json.load(open(
        os.path.join(config.OUTPUT_DIR, "phase_6_local",
                     "llm_picks_cache_cl1500_largedev.json")
    ))

    # split cache_small_test into dev / test parts
    # (cache_small_test was written dev-first, so 200 keys for dev q_ids are dev preds,
    #  the remaining 1598 are test preds for OTHER q_ids — those happen to be test-only.)
    # For test set evaluation, use cache_small_test minus dev_id overlaps if any.
    dev_cache = {k: v for k, v in cache_small_test.items() if int(k) in dev_ids}
    test_cache = {k: v for k, v in cache_small_test.items()
                  if int(k) not in dev_ids}

    print(f"cache_small_test:  {len(cache_small_test)} entries")
    print(f"  dev_cache slice: {len(dev_cache)}")
    print(f"  test_cache slice (dev-overlap removed): {len(test_cache)}")
    print(f"cache_large_dev:   {len(cache_large_dev)} entries")

    # for test, we need test samples but their cache entries are in `test_cache`
    # (also note q_id collision: cache_small_test has 1798 entries total
    #  for 200 dev + 1598 unique-test = 1798, meaning 200 test rows in
    #  the submission ARE actually dev predictions per MEMORY.md warning)

    analyze(dev, dev_cache, "dev (n=200)")
    analyze(large_dev, cache_large_dev, "large_dev (n=899)")
    analyze(test, test_cache, "test slice (n=1798, but 200 contaminated by dev preds)")


if __name__ == "__main__":
    main()
