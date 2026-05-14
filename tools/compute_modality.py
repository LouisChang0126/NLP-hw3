"""Compute modality split (text vs image) for a Phase 6 cache vs the dev gold."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_train, split_train_dev, get_gold_quotes_dict

CACHE_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/phase_6_local/llm_picks_cache_cl1500.json"

with open(CACHE_PATH) as f:
    cache = json.load(f)

train = load_train()
_, dev = split_train_dev(train)
dev_ids = {s["q_id"] for s in dev}
golds = get_gold_quotes_dict(dev)

# predicted modality split (Top-5)
pred_t, pred_i = 0, 0
for qid_str, picks in cache.items():
    if int(qid_str) not in dev_ids:
        continue
    for pid in picks[:5]:
        if pid.lower().startswith("image"):
            pred_i += 1
        else:
            pred_t += 1

# gold modality split
gold_t, gold_i = 0, 0
for qid, gset in golds.items():
    for g in gset:
        if g.lower().startswith("image"):
            gold_i += 1
        else:
            gold_t += 1

# hit-rate per modality
hit_t = hit_i = 0
miss_t = miss_i = 0
for qid, gset in golds.items():
    picks = set(cache.get(str(qid), [])[:5])
    for g in gset:
        is_img = g.lower().startswith("image")
        if g in picks:
            if is_img: hit_i += 1
            else: hit_t += 1
        else:
            if is_img: miss_i += 1
            else: miss_t += 1

tot_p = pred_t + pred_i
tot_g = gold_t + gold_i
print(f"Cache: {CACHE_PATH}")
print(f"Predicted Top-5 modality: text {pred_t} ({100*pred_t/tot_p:.1f}%) / image {pred_i} ({100*pred_i/tot_p:.1f}%)")
print(f"Gold modality:            text {gold_t} ({100*gold_t/tot_g:.1f}%) / image {gold_i} ({100*gold_i/tot_g:.1f}%)")
print(f"Bias: text {100*pred_t/tot_p - 100*gold_t/tot_g:+.1f} pp,  image {100*pred_i/tot_p - 100*gold_i/tot_g:+.1f} pp")
print()
print(f"Hit-rate per modality:")
print(f"  text:  {hit_t}/{hit_t+miss_t} = {hit_t/(hit_t+miss_t):.4f}")
print(f"  image: {hit_i}/{hit_i+miss_i} = {hit_i/(hit_i+miss_i):.4f}")
