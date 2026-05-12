"""
INLP HW3 - 本地評估框架
Phase 0.5 任務 2: Recall@K 計算
"""
from typing import Dict, List, Optional


def recall_at_k(
    preds: Dict[int, List[str]],
    golds: Dict[int, List[str]],
    k: int = 5
) -> float:
    """
    計算 Recall@K。
    
    Args:
        preds: {q_id: [predicted_quote_id, ...]}  (已排序，取前 k)
        golds: {q_id: [gold_quote_id, ...]}
        k: 取前 k 個預測
        
    Returns:
        平均 Recall@K (macro average over all questions)
    """
    per_sample = recall_at_k_per_sample(preds, golds, k)
    if not per_sample:
        return 0.0
    return sum(per_sample.values()) / len(per_sample)


def recall_at_k_per_sample(
    preds: Dict[int, List[str]],
    golds: Dict[int, List[str]],
    k: int = 5
) -> Dict[int, float]:
    """
    計算每筆 sample 的 Recall@K。
    
    Returns:
        {q_id: recall_score}
    """
    results = {}
    for q_id, gold_list in golds.items():
        if q_id not in preds:
            results[q_id] = 0.0
            continue

        pred_list = preds[q_id][:k]
        gold_set = set(gold_list)

        if len(gold_set) == 0:
            results[q_id] = 1.0
            continue

        hits = sum(1 for p in pred_list if p in gold_set)
        results[q_id] = hits / len(gold_set)

    return results


def error_analysis(
    preds: Dict[int, List[str]],
    golds: Dict[int, List[str]],
    k: int = 5
) -> dict:
    """
    錯誤分析：列出 recall < 1.0 的案例。
    
    Returns:
        {
            'perfect': [(q_id, recall), ...],
            'partial': [(q_id, recall, missed, extra), ...],
            'zero': [(q_id, gold_list), ...],
        }
    """
    per_sample = recall_at_k_per_sample(preds, golds, k)
    
    perfect = []
    partial = []
    zero = []
    
    for q_id, score in per_sample.items():
        if score >= 1.0:
            perfect.append((q_id, score))
        elif score > 0.0:
            pred_set = set(preds.get(q_id, [])[:k])
            gold_set = set(golds[q_id])
            missed = gold_set - pred_set
            extra = pred_set - gold_set
            partial.append((q_id, score, list(missed), list(extra)))
        else:
            zero.append((q_id, golds[q_id]))
    
    return {
        "perfect": perfect,
        "partial": partial,
        "zero": zero,
        "summary": {
            "total": len(per_sample),
            "perfect_count": len(perfect),
            "partial_count": len(partial),
            "zero_count": len(zero),
            "mean_recall": sum(per_sample.values()) / max(len(per_sample), 1),
        }
    }


if __name__ == "__main__":
    # 簡單測試
    preds = {0: ["text1", "text2", "image1"], 1: ["text3", "text4"]}
    golds = {0: ["text1", "image1", "text5"], 1: ["text3"]}
    
    print(f"Recall@5 = {recall_at_k(preds, golds, k=5):.4f}")
    print(f"Per-sample: {recall_at_k_per_sample(preds, golds, k=5)}")
