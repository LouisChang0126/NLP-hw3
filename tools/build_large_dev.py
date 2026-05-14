"""
Generate data/splits/large_dev_ids.json — half-test-size dev set
sampled via the same seed=42 shuffle so dev_ids.json is a subset.
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SEED = config.SEED
TRAIN = config.TRAIN_FILE
TEST = config.TEST_FILE
OUT = os.path.join(config.SPLITS_DIR, "large_dev_ids.json")


def main():
    train = [json.loads(l) for l in open(TRAIN)]
    n_test = sum(1 for _ in open(TEST))
    target_n = n_test // 2

    random.seed(SEED)
    idx = list(range(len(train)))
    random.shuffle(idx)
    dev_idx = idx[-target_n:]
    dev_ids = [train[i]["q_id"] for i in dev_idx]

    with open(os.path.join(config.SPLITS_DIR, "dev_ids.json")) as f:
        existing_200 = set(json.load(f))
    is_superset = set(dev_ids).issuperset(existing_200)

    os.makedirs(config.SPLITS_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dev_ids, f)

    dom = Counter(train[i].get("domain", "?") for i in dev_idx)
    full_dom = Counter(s.get("domain", "?") for s in train)

    print(f"✅ Saved: {OUT}")
    print(f"   N = {len(dev_ids)} (target = test/2 = {target_n})")
    print(f"   Superset of existing dev_ids.json (200)?  {is_superset}")
    print()
    print(f"{'Domain':<35} {'train':>10} {'large_dev':>12}")
    print("-" * 60)
    n_train = len(train)
    for d, c in full_dom.most_common():
        ld = dom.get(d, 0)
        print(f"{d:<35} {c:>4d} ({100*c/n_train:>4.1f}%)  {ld:>4d} ({100*ld/target_n:>4.1f}%)")


if __name__ == "__main__":
    main()
