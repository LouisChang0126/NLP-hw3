"""
INLP HW3 - 資料讀取與候選池建構
Phase 0 任務 1 & 2
"""
import json
import os
import random
from typing import Dict, List, Optional

import config


def load_jsonl(filepath: str) -> List[dict]:
    """讀取 JSONL 檔案，回傳 list of dict"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_train() -> List[dict]:
    """讀取 train.jsonl"""
    return load_jsonl(config.TRAIN_FILE)


def load_test() -> List[dict]:
    """讀取 test.jsonl"""
    return load_jsonl(config.TEST_FILE)


def parse_sample(sample: dict) -> dict:
    """
    解析單筆 sample，統一欄位名稱。
    回傳：
        {
            'q_id': int/str,
            'doc_name': str,
            'domain': str,
            'question': str,
            'evidence_modality_type': str,
            'text_quotes': [{quote_id, text, ...}],
            'img_quotes': [{quote_id, img_path, img_description, ...}],
            'gold_quotes': [str] or None,  # train only
        }
    """
    return {
        "q_id": sample["q_id"],
        "doc_name": sample.get("doc_name", ""),
        "domain": sample.get("domain", ""),
        "question": sample.get("question", ""),
        "evidence_modality_type": sample.get("evidence_modality_type", ""),
        "question_type": sample.get("question_type", ""),
        "text_quotes": sample.get("text_quotes", []),
        "img_quotes": sample.get("img_quotes", []),
        "gold_quotes": sample.get("gold_quotes", None),
    }


def build_candidates(sample: dict) -> List[dict]:
    """
    Phase 0 任務 2：為單題建立候選清單 (per-question)
    回傳 list of {quote_id, modality, text_for_retrieval}
    """
    candidates = []

    for tq in sample.get("text_quotes", []):
        candidates.append({
            "quote_id": tq["quote_id"],
            "modality": "text",
            "text_for_retrieval": tq.get("text", ""),
        })

    for iq in sample.get("img_quotes", []):
        candidates.append({
            "quote_id": iq["quote_id"],
            "modality": "image",
            "text_for_retrieval": iq.get("img_description", ""),
            "img_path": iq.get("img_path", ""),
        })

    return candidates


def build_all_candidates(data: List[dict]) -> Dict[int, List[dict]]:
    """
    對整個 dataset 建立 per-question 候選池。
    回傳 {q_id: [candidate_dict, ...]}
    """
    all_candidates = {}
    for sample in data:
        q_id = sample["q_id"]
        all_candidates[q_id] = build_candidates(sample)
    return all_candidates


def get_gold_quotes_dict(data: List[dict]) -> Dict[int, List[str]]:
    """從 train data 抽出 {q_id: [gold_quote_id, ...]}"""
    golds = {}
    for sample in data:
        if sample.get("gold_quotes"):
            golds[sample["q_id"]] = sample["gold_quotes"]
    return golds


def get_img_path(relative_path: str) -> str:
    """
    將 sample 中的 img_path (如 'images/xxx.jpg')
    映射到實際檔案路徑 (data/images/images/xxx.jpg)
    """
    # relative_path 格式: "images/xxx.jpg"
    filename = os.path.basename(relative_path)
    return os.path.join(config.IMAGE_DIR, filename)


def load_dev_ids() -> List[int]:
    """載入 dev split 的 q_id list"""
    dev_ids_path = os.path.join(config.SPLITS_DIR, "dev_ids.json")
    with open(dev_ids_path, "r") as f:
        return json.load(f)


def split_train_dev(train_data: List[dict]) -> tuple:
    """
    將 train_data 切分成 train_subset 與 dev_subset。
    使用固定 seed=42，取最後 DEV_SIZE 筆作為 dev。
    回傳 (train_subset, dev_subset)
    """
    dev_ids_path = os.path.join(config.SPLITS_DIR, "dev_ids.json")
    if os.path.exists(dev_ids_path):
        dev_ids = set(load_dev_ids())
    else:
        # 尚未切分，用 seed 固定順序取最後 N 筆
        random.seed(config.SEED)
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        dev_indices = set(indices[-config.DEV_SIZE:])
        dev_ids = set(train_data[i]["q_id"] for i in dev_indices)

    train_subset = [s for s in train_data if s["q_id"] not in dev_ids]
    dev_subset = [s for s in train_data if s["q_id"] in dev_ids]
    return train_subset, dev_subset


if __name__ == "__main__":
    # 快速驗證
    train_data = load_train()
    test_data = load_test()
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    sample = parse_sample(train_data[0])
    print(f"\nSample q_id={sample['q_id']}")
    print(f"  doc_name: {sample['doc_name']}")
    print(f"  domain: {sample['domain']}")
    print(f"  question: {sample['question'][:80]}...")
    print(f"  text_quotes: {len(sample['text_quotes'])} items")
    print(f"  img_quotes: {len(sample['img_quotes'])} items")
    print(f"  gold_quotes: {sample['gold_quotes']}")

    candidates = build_candidates(sample)
    print(f"\n  Candidates pool: {len(candidates)} items")
    for c in candidates[:3]:
        print(f"    {c['quote_id']} ({c['modality']}): {c['text_for_retrieval'][:60]}...")
