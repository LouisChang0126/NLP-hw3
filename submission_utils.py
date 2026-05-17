"""
INLP HW3 - 提交檔產生與驗證工具
"""
import csv
import json
import os
from typing import Dict, List

import config


def generate_submission(
    preds: Dict[int, List[str]],
    test_data: List[dict],
    output_path: str,
    top_k: int = 5,
    pad_short: bool = True,
) -> str:
    """
    產生 Kaggle 提交檔 (CSV)，精確對齊官方格式。
    
    Args:
        preds: {q_id: [quote_id, ...]}
        test_data: test.jsonl 的原始資料
        output_path: submission.csv 的完整路徑
        top_k: 每題取前幾個
        
    Returns:
        output_path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    for sample in test_data:
        q_id = sample["q_id"]
        pred_list = preds.get(q_id, [])[:top_k]
        
        # 如果預測不足 top_k，用候選池中的 quote_id 補位 (除非 pad_short=False)
        if pad_short and len(pred_list) < top_k:
            existing = set(pred_list)
            # 從 text_quotes 和 img_quotes 中補
            for tq in sample.get("text_quotes", []):
                if tq["quote_id"] not in existing:
                    pred_list.append(tq["quote_id"])
                    existing.add(tq["quote_id"])
                    if len(pred_list) >= top_k:
                        break
            if len(pred_list) < top_k:
                for iq in sample.get("img_quotes", []):
                    if iq["quote_id"] not in existing:
                        pred_list.append(iq["quote_id"])
                        existing.add(iq["quote_id"])
                        if len(pred_list) >= top_k:
                            break
        
        gold_quotes_str = " ".join(pred_list[:top_k])
        rows.append({"q_id": q_id, "gold_quotes": gold_quotes_str})
    
    # 驗證
    assert len(rows) == len(test_data), \
        f"行數 {len(rows)} != test 筆數 {len(test_data)}"
    
    for row in rows:
        tokens = row["gold_quotes"].split()
        assert len(tokens) <= top_k, \
            f"q_id={row['q_id']} 有 {len(tokens)} 個 token (> {top_k})"
    
    # 寫入 CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["q_id", "gold_quotes"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Submission saved: {output_path} ({len(rows)} rows)")
    return output_path


def save_metrics(metrics: dict, output_dir: str):
    """將 metrics 存入 output_dir/metrics.json"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ Metrics saved: {path}")


def append_leaderboard(
    phase: str,
    model: str,
    dev_recall: float,
    public_lb: str = "-",
    note: str = "",
):
    """Append 一行到 outputs/leaderboard.md"""
    lb_path = os.path.join(config.OUTPUT_DIR, "leaderboard.md")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 若檔案不存在，先寫 header
    if not os.path.exists(lb_path):
        with open(lb_path, "w", encoding="utf-8") as f:
            f.write("# Leaderboard\n\n")
            f.write("| Phase | Model | Dev Recall@5 | Public LB | Note |\n")
            f.write("|-------|-------|-------------|-----------|------|\n")
    
    with open(lb_path, "a", encoding="utf-8") as f:
        f.write(f"| {phase} | {model} | {dev_recall:.4f} | {public_lb} | {note} |\n")
    
    print(f"✅ Leaderboard updated: {lb_path}")


def save_rankings(
    rankings: Dict[int, List[str]],
    output_path: str,
):
    """將 per-question 排名結果存為 JSON (供後續 phase 重用)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert int keys to str for JSON
    serializable = {str(k): v for k, v in rankings.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"✅ Rankings saved: {output_path}")


def load_rankings(input_path: str) -> Dict[int, List[str]]:
    """讀取 per-question 排名結果"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
