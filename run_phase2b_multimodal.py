"""
INLP HW3 - Phase 2b: Multimodal Embedding 檢索
對比 (a) 直接嵌入原始圖片 vs (b) 圖→文字描述後純文字檢索
"""
import json
import os
import time
import numpy as np
from typing import Dict, List

from tqdm import tqdm
import torch
from PIL import Image

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict, get_img_path,
)
from evaluation import recall_at_k, recall_at_k_per_sample, error_analysis
from submission_utils import (
    generate_submission, save_metrics, append_leaderboard, save_rankings,
)

config.print_config()


def load_mm_model():
    """載入多模態 embedding model (SigLIP 或 CLIP)"""
    model_name = config.PHASE_2B_MM_MODEL
    print(f"📦 Loading multimodal model: {model_name}")
    
    if "siglip" in model_name.lower():
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return model, processor, "siglip"
    else:
        # CLIP
        from transformers import CLIPProcessor, CLIPModel
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return model, processor, "clip"


def encode_texts_mm(model, processor, texts: List[str], model_type: str, batch_size: int = 32) -> np.ndarray:
    """用多模態模型的 text tower 編碼文字"""
    device = next(model.parameters()).device
    all_embs = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            if model_type == "siglip":
                text_embs = model.get_text_features(**inputs)
            else:
                text_embs = model.get_text_features(**inputs)
        
        all_embs.append(text_embs.cpu().float().numpy())
    
    return np.vstack(all_embs)


def encode_images_mm(model, processor, img_paths: List[str], model_type: str, batch_size: int = 16) -> np.ndarray:
    """用多模態模型的 image tower 編碼圖片"""
    device = next(model.parameters()).device
    all_embs = []
    
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                # 用空白圖代替
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
        
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            if model_type == "siglip":
                img_embs = model.get_image_features(**inputs)
            else:
                img_embs = model.get_image_features(**inputs)
        
        all_embs.append(img_embs.cpu().float().numpy())
    
    return np.vstack(all_embs)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (d,), b: (n, d) -> (n,)"""
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_n @ a_n


def mm_retrieve_per_question(
    question: str,
    candidates: List[dict],
    model, processor, model_type: str,
    top_k: int = 5,
) -> List[str]:
    """
    多模態檢索：
    - text_quotes: 用 text tower 編碼文字
    - img_quotes: 用 image tower 編碼原始圖片
    - question: 用 text tower 編碼
    用 RRF 合併不同模態的分數
    """
    if not candidates:
        return []
    
    device = next(model.parameters()).device
    
    # 編碼 query
    q_inputs = processor(text=[question], return_tensors="pt", padding=True, truncation=True, max_length=256)
    q_inputs = {k: v.to(device) for k, v in q_inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        q_emb = model.get_text_features(**q_inputs).cpu().float().numpy()[0]
    
    scored = []
    
    for c in candidates:
        if c["modality"] == "text":
            # 用 text tower
            t_inputs = processor(
                text=[c["text_for_retrieval"][:500]],
                return_tensors="pt", padding=True, truncation=True, max_length=256,
            )
            t_inputs = {k: v.to(device) for k, v in t_inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                c_emb = model.get_text_features(**t_inputs).cpu().float().numpy()[0]
        else:
            # 用 image tower
            img_path = get_img_path(c.get("img_path", ""))
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            
            i_inputs = processor(images=[img], return_tensors="pt", padding=True)
            i_inputs = {k: v.to(device) for k, v in i_inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                c_emb = model.get_image_features(**i_inputs).cpu().float().numpy()[0]
        
        sim = float(cosine_sim(q_emb, c_emb.reshape(1, -1))[0])
        scored.append((c["quote_id"], sim))
    
    # Z-score 正規化後排序
    scores = np.array([s for _, s in scored])
    mean_s = scores.mean()
    std_s = scores.std() + 1e-10
    normalized = (scores - mean_s) / std_s
    
    indexed = list(zip([qid for qid, _ in scored], normalized))
    indexed.sort(key=lambda x: x[1], reverse=True)
    
    return [qid for qid, _ in indexed[:top_k]]


def run_mm_retrieval(
    data: List[dict],
    model, processor, model_type: str,
    top_k: int = 5,
) -> Dict[int, List[str]]:
    """對整個 dataset 做多模態檢索"""
    results = {}
    for sample in tqdm(data, desc="MM retrieval"):
        q_id = sample["q_id"]
        candidates = build_candidates(sample)
        ranked = mm_retrieve_per_question(
            sample["question"], candidates, model, processor, model_type, top_k
        )
        results[q_id] = ranked
    return results


def main():
    # ── 載入資料 ──────────────────────────────────────
    train_data = load_train()
    test_data = load_test()
    train_subset, dev_subset = split_train_dev(train_data)
    
    print(f"📊 Train: {len(train_subset)}, Dev: {len(dev_subset)}, Test: {len(test_data)}")
    
    # ── 載入模型 ──────────────────────────────────────
    model, processor, model_type = load_mm_model()
    
    # ── 輸出目錄 ──────────────────────────────────────
    output_dir = config.get_output_dir("phase_2b", config.PHASE_2B_MM_MODEL)
    
    # ── Dev 評估 (with timing) ────────────────────────
    print("\n🔍 Running Multimodal Retrieval on dev set...")
    t0 = time.time()
    dev_preds = run_mm_retrieval(dev_subset, model, processor, model_type, top_k=5)
    dev_time = time.time() - t0
    
    dev_golds = get_gold_quotes_dict(dev_subset)
    dev_recall = recall_at_k(dev_preds, dev_golds, k=5)
    print(f"\n📈 Dev Recall@5 = {dev_recall:.4f} (time: {dev_time:.1f}s)")
    
    ea = error_analysis(dev_preds, dev_golds, k=5)
    
    # ── Modality 命中率分析 ───────────────────────────
    text_hits = 0
    img_hits = 0
    text_total = 0
    img_total = 0
    
    for sample in dev_subset:
        q_id = sample["q_id"]
        golds = set(sample.get("gold_quotes", []))
        preds = set(dev_preds.get(q_id, [])[:5])
        
        for g in golds:
            if g.startswith("text"):
                text_total += 1
                if g in preds:
                    text_hits += 1
            elif g.startswith("image"):
                img_total += 1
                if g in preds:
                    img_hits += 1
    
    text_hit_rate = text_hits / max(text_total, 1)
    img_hit_rate = img_hits / max(img_total, 1)
    
    print(f"  Text modality hit rate:  {text_hits}/{text_total} = {text_hit_rate:.4f}")
    print(f"  Image modality hit rate: {img_hits}/{img_total} = {img_hit_rate:.4f}")
    
    # ── Test 預測 ─────────────────────────────────────
    print("\n🔍 Running Multimodal Retrieval on test set...")
    test_preds = run_mm_retrieval(test_data, model, processor, model_type, top_k=5)
    
    # ── 儲存結果 ──────────────────────────────────────
    submission_path = os.path.join(output_dir, "submission.csv")
    generate_submission(test_preds, test_data, submission_path)
    
    metrics = {
        "phase": "phase_2b",
        "model": config.PHASE_2B_MM_MODEL,
        "dev_recall_at_5": dev_recall,
        "dev_time_seconds": dev_time,
        "text_modality_hit_rate": text_hit_rate,
        "image_modality_hit_rate": img_hit_rate,
        "dev_samples": len(dev_subset),
        "test_samples": len(test_data),
        "error_analysis_summary": ea["summary"],
    }
    save_metrics(metrics, output_dir)
    
    # ── 產生 Q3 對照數據 (comparison.json) ────────────
    comparison = {
        "multimodal_embedding": {
            "model": config.PHASE_2B_MM_MODEL,
            "dev_recall_at_5": dev_recall,
            "dev_latency_seconds": dev_time,
            "text_modality_hit_rate": text_hit_rate,
            "image_modality_hit_rate": img_hit_rate,
        },
        "text_description": {
            "model": config.PHASE_2_EMBED_MODEL,
            "note": "Run Phase 2 first to fill this data",
        },
    }
    comp_path = os.path.join(output_dir, "comparison.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"✅ Comparison saved: {comp_path}")
    
    append_leaderboard("Phase 2b", config.PHASE_2B_MM_MODEL, dev_recall,
                       note="Multimodal Embedding")
    
    print(f"\n✅ Phase 2b 完成! 輸出目錄: {output_dir}")


if __name__ == "__main__":
    main()
