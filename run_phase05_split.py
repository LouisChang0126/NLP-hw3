"""
INLP HW3 - Phase 0.5: 本地驗證框架
任務 1: 切分 dev set
任務 3 & 4: metrics 存檔 + leaderboard 初始化
"""
import json
import os
import random

import config
from dataset import load_train

config.print_config()

random.seed(config.SEED)


def create_dev_split():
    """從 train.jsonl 切出 hold-out dev set (seed=42 shuffle 後取最後 200 筆)"""
    train_data = load_train()
    print(f"📊 Train samples: {len(train_data)}")
    
    # 固定 seed shuffle 後取最後 DEV_SIZE 筆
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    dev_indices = indices[-config.DEV_SIZE:]
    dev_ids = [train_data[i]["q_id"] for i in dev_indices]
    
    # Domain 分佈檢查
    from collections import Counter
    all_domains = Counter(s["domain"] for s in train_data)
    dev_domains = Counter(train_data[i]["domain"] for i in dev_indices)
    train_subset_domains = Counter(
        train_data[i]["domain"] for i in indices[:-config.DEV_SIZE]
    )
    
    print(f"\n📌 Dev split: {len(dev_ids)} samples")
    print(f"  Train subset: {len(train_data) - len(dev_ids)} samples")
    print(f"\n  Domain 分佈對照:")
    print(f"  {'Domain':<25} {'All':>6} {'Train':>6} {'Dev':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
    for d in sorted(all_domains.keys()):
        print(f"  {d:<25} {all_domains[d]:>6} {train_subset_domains.get(d,0):>6} {dev_domains.get(d,0):>6}")
    
    # 儲存
    os.makedirs(config.SPLITS_DIR, exist_ok=True)
    dev_ids_path = os.path.join(config.SPLITS_DIR, "dev_ids.json")
    with open(dev_ids_path, "w") as f:
        json.dump(dev_ids, f, indent=2)
    
    print(f"\n✅ Dev IDs saved: {dev_ids_path}")
    
    # 初始化 leaderboard
    lb_path = os.path.join(config.OUTPUT_DIR, "leaderboard.md")
    if not os.path.exists(lb_path):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(lb_path, "w", encoding="utf-8") as f:
            f.write("# Leaderboard\n\n")
            f.write("| Phase | Model | Dev Recall@5 | Public LB | Note |\n")
            f.write("|-------|-------|-------------|-----------|------|\n")
        print(f"✅ Leaderboard initialized: {lb_path}")
    
    return dev_ids


if __name__ == "__main__":
    create_dev_split()
