"""
INLP HW3 - Phase 4: SOTA 策略
HyDE (Query Expansion) + Cross-Encoder Reranking
"""
import json
import os
import re
from typing import Dict, List

from tqdm import tqdm
import torch
import numpy as np

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict,
)
from evaluation import recall_at_k, error_analysis
from submission_utils import (
    generate_submission, save_metrics, append_leaderboard,
    save_rankings, load_rankings,
)
from run_phase3_hybrid import find_latest_rankings

config.print_config()


# ══════════════════════════════════════════════════════
#  HyDE: Query Expansion
# ══════════════════════════════════════════════════════

HYDE_PROMPT_TEMPLATE = """You are a document expert. Given the following question about a document, write a short paragraph (3-5 sentences) that could serve as the answer or relevant evidence. Include specific details, numbers, and terminology that might appear in the actual document.

Document: {doc_name}
Domain: {domain}
Question: {question}

Hypothetical answer:"""


def load_expansion_model():
    """載入 HyDE 用的 LLM"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = config.PHASE_4_EXPANSION_MODEL
    print(f"📦 Loading expansion model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def generate_hyde(
    question: str,
    doc_name: str,
    domain: str,
    model,
    tokenizer,
) -> str:
    """用 LLM 生成假設性回答 (HyDE)"""
    prompt = HYDE_PROMPT_TEMPLATE.format(
        doc_name=doc_name, domain=domain, question=question
    )
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def expand_queries(
    data: List[dict],
    model,
    tokenizer,
) -> Dict[int, str]:
    """對所有問題生成 HyDE 擴展查詢"""
    expanded = {}
    for sample in tqdm(data, desc="HyDE expansion"):
        q_id = sample["q_id"]
        question = sample["question"]
        doc_name = sample.get("doc_name", "")
        domain = sample.get("domain", "")
        
        try:
            hyde_text = generate_hyde(question, doc_name, domain, model, tokenizer)
            # 拼接原問題 + 假設回答
            expanded[q_id] = f"{question} {hyde_text}"
        except Exception as e:
            print(f"  ⚠️ q_id={q_id} HyDE failed: {e}")
            expanded[q_id] = question
    
    return expanded


# ══════════════════════════════════════════════════════
#  Cross-Encoder Reranking
# ══════════════════════════════════════════════════════

def load_reranker():
    """載入 Cross-Encoder reranker"""
    model_name = config.PHASE_4_RERANKER_MODEL
    print(f"📦 Loading reranker: {model_name}")
    
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker(model_name, use_fp16=True)
    return reranker


def rerank_per_question(
    query: str,
    candidates: List[dict],
    candidate_ids: List[str],
    reranker,
    top_k: int = 5,
) -> List[str]:
    """
    用 Cross-Encoder 重排序候選。
    
    Args:
        query: 查詢文字 (可能已經過 HyDE 擴展)
        candidates: 完整候選池 [{quote_id, modality, text_for_retrieval}]
        candidate_ids: 要重排的 quote_id 列表 (來自 Phase 3 的 Top-100)
        reranker: FlagReranker instance
        top_k: 輸出前幾名
    """
    if not candidate_ids:
        return []
    
    # 建立 id -> text 映射
    id_to_text = {c["quote_id"]: c["text_for_retrieval"] for c in candidates}
    
    # 組合 pairs
    pairs = []
    valid_ids = []
    for cid in candidate_ids:
        text = id_to_text.get(cid, "")
        if text:
            pairs.append([query, text])
            valid_ids.append(cid)
    
    if not pairs:
        return candidate_ids[:top_k]
    
    # 計算分數
    scores = reranker.compute_score(pairs)
    if isinstance(scores, (int, float)):
        scores = [scores]
    
    # 排序
    scored = list(zip(valid_ids, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [qid for qid, _ in scored[:top_k]]


def run_reranking(
    data: List[dict],
    initial_rankings: Dict[int, List[str]],
    reranker,
    expanded_queries: Dict[int, str] = None,
    top_k: int = 5,
    rerank_depth: int = 100,
) -> Dict[int, List[str]]:
    """對整個 dataset 做 Cross-Encoder reranking"""
    results = {}
    
    for sample in tqdm(data, desc="Reranking"):
        q_id = sample["q_id"]
        
        # 使用擴展後的查詢，若無則用原始問題
        if expanded_queries and q_id in expanded_queries:
            query = expanded_queries[q_id]
        else:
            query = sample["question"]
        
        candidates = build_candidates(sample)
        candidate_ids = initial_rankings.get(q_id, [])[:rerank_depth]
        
        ranked = rerank_per_question(query, candidates, candidate_ids, reranker, top_k)
        results[q_id] = ranked
    
    return results


def main():
    # ── 載入資料 ──────────────────────────────────────
    train_data = load_train()
    test_data = load_test()
    train_subset, dev_subset = split_train_dev(train_data)
    
    print(f"📊 Train: {len(train_subset)}, Dev: {len(dev_subset)}, Test: {len(test_data)}")
    
    # ── 尋找 Phase 3 的 rankings ─────────────────────
    hybrid_test_path = find_latest_rankings("phase_3", "rankings_top100.json")
    hybrid_dev_path = find_latest_rankings("phase_3", "dev_rankings_top100.json")
    
    # 如果 Phase 3 不存在，嘗試用 Phase 2 的 rankings
    if not hybrid_test_path:
        print("⚠️ Phase 3 rankings 不存在，嘗試用 Phase 2...")
        hybrid_test_path = find_latest_rankings("phase_2", "rankings_top100.json")
        hybrid_dev_path = find_latest_rankings("phase_2", "dev_rankings_top100.json")
    
    if not hybrid_test_path:
        print("❌ 需要先跑 Phase 2 或 Phase 3!")
        return
    
    print(f"📂 Initial rankings (test): {hybrid_test_path}")
    
    test_rankings = load_rankings(hybrid_test_path)
    dev_rankings = load_rankings(hybrid_dev_path) if hybrid_dev_path else {}
    
    # ── 輸出目錄 ──────────────────────────────────────
    output_dir = config.get_output_dir("phase_4", "hyde_reranker")
    
    # ── HyDE Query Expansion ─────────────────────────
    # 儲存 HyDE prompt 模板
    prompts_dir = os.path.join(config.PROJECT_ROOT, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    with open(os.path.join(prompts_dir, "hyde.txt"), "w") as f:
        f.write(HYDE_PROMPT_TEMPLATE)
    
    print("\n📝 HyDE Query Expansion...")
    expansion_model, expansion_tokenizer = load_expansion_model()
    
    dev_expanded = expand_queries(dev_subset, expansion_model, expansion_tokenizer)
    test_expanded = expand_queries(test_data, expansion_model, expansion_tokenizer)
    
    # 釋放 LLM 記憶體
    del expansion_model, expansion_tokenizer
    torch.cuda.empty_cache()
    
    # ── 用擴展查詢重跑 Dense retrieval 取 Top-100 ─────
    # 如果有 Phase 3 的 rankings 就直接用
    print("\n🔍 Loading reranker...")
    reranker = load_reranker()
    
    # ── Dev Reranking ─────────────────────────────────
    if dev_rankings:
        print("\n🔍 Reranking dev set...")
        dev_preds = run_reranking(
            dev_subset, dev_rankings, reranker,
            expanded_queries=dev_expanded,
            top_k=config.RERANKER_TOP_K,
        )
        
        dev_golds = get_gold_quotes_dict(dev_subset)
        dev_recall = recall_at_k(dev_preds, dev_golds, k=5)
        print(f"\n📈 Dev Recall@5 (HyDE + Reranker) = {dev_recall:.4f}")
        
        ea = error_analysis(dev_preds, dev_golds, k=5)
        print(f"  Perfect: {ea['summary']['perfect_count']}")
        print(f"  Partial: {ea['summary']['partial_count']}")
        print(f"  Zero:    {ea['summary']['zero_count']}")
    else:
        dev_recall = -1.0
        ea = None
    
    # ── Test Reranking ────────────────────────────────
    print("\n🔍 Reranking test set...")
    test_preds = run_reranking(
        test_data, test_rankings, reranker,
        expanded_queries=test_expanded,
        top_k=config.RERANKER_TOP_K,
    )
    
    # ── 儲存結果 ──────────────────────────────────────
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds, test_data, submission_path)
    
    # 儲存 HyDE 擴展結果
    hyde_path = os.path.join(output_dir, "hyde_queries.json")
    with open(hyde_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in test_expanded.items()}, f, indent=2, ensure_ascii=False)
    
    metrics = {
        "phase": "phase_4",
        "expansion_model": config.PHASE_4_EXPANSION_MODEL,
        "reranker_model": config.PHASE_4_RERANKER_MODEL,
        "initial_rankings_source": hybrid_test_path,
        "dev_recall_at_5": dev_recall,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
    }
    if ea:
        metrics["error_analysis_summary"] = ea["summary"]
    save_metrics(metrics, output_dir)
    
    if dev_recall > 0:
        append_leaderboard(
            "Phase 4",
            f"HyDE({config.PHASE_4_EXPANSION_MODEL}) + {config.PHASE_4_RERANKER_MODEL}",
            dev_recall,
            note="Query Expansion + Reranker",
        )
    
    print(f"\n✅ Phase 4 完成! 輸出目錄: {output_dir}")


if __name__ == "__main__":
    main()
