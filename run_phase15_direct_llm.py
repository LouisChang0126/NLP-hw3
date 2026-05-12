"""
INLP HW3 - Phase 1.5: Direct LLM Selection Baseline
直接讓 LLM 從候選池中挑選 Top-5
"""
import json
import os
import re
from typing import Dict, List

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict,
)
from evaluation import recall_at_k, error_analysis
from submission_utils import (
    generate_submission, save_metrics, append_leaderboard,
)

# 匯入 BM25 fallback
from run_phase1_bm25 import bm25_retrieve_per_question

config.print_config()


def load_llm():
    """載入 LLM 模型"""
    model_name = config.PHASE_DIRECT_LLM_MODEL
    print(f"📦 Loading LLM: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def build_prompt(question: str, candidates: List[dict], max_candidates: int = 50) -> str:
    """
    建構 prompt，要求 LLM 從候選中選出 Top-5。
    若候選數太多則截斷。
    """
    cand_text = ""
    for i, c in enumerate(candidates[:max_candidates]):
        cand_text += f"\n[{c['quote_id']}] ({c['modality']}): {c['text_for_retrieval'][:500]}"
    
    prompt = f"""You are a document evidence retrieval expert. Given a question and a list of candidate evidence passages (text or image descriptions), select the TOP 5 most relevant candidates that best answer the question.

Question: {question}

Candidates:{cand_text}

Return ONLY a JSON object with the top 5 quote_ids, ordered by relevance:
{{"top5": ["quote_id_1", "quote_id_2", "quote_id_3", "quote_id_4", "quote_id_5"]}}
"""
    return prompt


def parse_llm_output(output: str, valid_ids: set) -> List[str]:
    """解析 LLM 輸出，抽取 quote_id list"""
    # 嘗試找 JSON
    json_match = re.search(r'\{[^{}]*"top5"\s*:\s*\[([^\]]*)\][^{}]*\}', output)
    if json_match:
        try:
            full_match = json_match.group(0)
            parsed = json.loads(full_match)
            ids = parsed.get("top5", [])
            # 過濾有效 ID
            return [qid for qid in ids if qid in valid_ids][:5]
        except json.JSONDecodeError:
            pass
    
    # Fallback: 用 regex 找所有 quote_id 格式的字串
    found = re.findall(r'(text\d+|image\d+)', output)
    return [qid for qid in found if qid in valid_ids][:5]


def llm_select_per_question(
    question: str,
    candidates: List[dict],
    model,
    tokenizer,
) -> List[str]:
    """用 LLM 對單題進行 Top-5 選擇"""
    if not candidates:
        return []
    
    prompt = build_prompt(question, candidates)
    valid_ids = {c["quote_id"] for c in candidates}
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
        )
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    
    parsed = parse_llm_output(result, valid_ids)
    return parsed


def run_llm_selection(
    data: List[dict],
    model,
    tokenizer,
) -> Dict[int, List[str]]:
    """對整個 dataset 做 LLM selection"""
    results = {}
    for sample in tqdm(data, desc="LLM selection"):
        q_id = sample["q_id"]
        question = sample["question"]
        candidates = build_candidates(sample)
        
        try:
            selected = llm_select_per_question(question, candidates, model, tokenizer)
        except Exception as e:
            print(f"  ⚠️ q_id={q_id} LLM failed: {e}")
            selected = []
        
        # 若數量不足，用 BM25 補位
        if len(selected) < 5:
            bm25_fallback = bm25_retrieve_per_question(question, candidates, top_k=5)
            existing = set(selected)
            for qid in bm25_fallback:
                if qid not in existing:
                    selected.append(qid)
                    existing.add(qid)
                    if len(selected) >= 5:
                        break
        
        results[q_id] = selected[:5]
    
    return results


def main():
    # ── 載入資料 ──────────────────────────────────────
    train_data = load_train()
    test_data = load_test()
    train_subset, dev_subset = split_train_dev(train_data)
    
    print(f"📊 Train: {len(train_subset)}, Dev: {len(dev_subset)}, Test: {len(test_data)}")
    
    # ── 載入模型 ──────────────────────────────────────
    model, tokenizer = load_llm()
    
    # ── 輸出目錄 ──────────────────────────────────────
    output_dir = config.get_output_dir("phase_direct_llm", config.PHASE_DIRECT_LLM_MODEL)
    
    # ── Dev 評估 ──────────────────────────────────────
    print("\n🔍 Running Direct LLM Selection on dev set...")
    dev_preds = run_llm_selection(dev_subset, model, tokenizer)
    dev_golds = get_gold_quotes_dict(dev_subset)
    
    dev_recall = recall_at_k(dev_preds, dev_golds, k=5)
    print(f"\n📈 Dev Recall@5 = {dev_recall:.4f}")
    
    ea = error_analysis(dev_preds, dev_golds, k=5)
    print(f"  Perfect: {ea['summary']['perfect_count']}")
    print(f"  Partial: {ea['summary']['partial_count']}")
    print(f"  Zero:    {ea['summary']['zero_count']}")
    
    # ── Test 預測 ─────────────────────────────────────
    print("\n🔍 Running Direct LLM Selection on test set...")
    test_preds = run_llm_selection(test_data, model, tokenizer)
    
    # ── 儲存結果 ──────────────────────────────────────
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds, test_data, submission_path)
    
    metrics = {
        "phase": "phase_direct_llm",
        "model": config.PHASE_DIRECT_LLM_MODEL,
        "dev_recall_at_5": dev_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)
    
    append_leaderboard("Phase 1.5", config.PHASE_DIRECT_LLM_MODEL, dev_recall,
                       note="Direct LLM Selection")
    
    print(f"\n✅ Phase 1.5 完成! 輸出目錄: {output_dir}")


if __name__ == "__main__":
    main()
