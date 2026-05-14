"""
INLP HW3 - Phase 5: 分析與報告輔助
對應 Q1 ~ Q4 全部報告題
"""
import json
import os
from collections import Counter
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from dataset import (
    load_train, load_test, build_candidates,
    split_train_dev, get_gold_quotes_dict,
)
from submission_utils import load_rankings


def find_latest_rankings(phase_dir: str, filename: str = "rankings_top100.json") -> str:
    """在 phase 目錄中找最新的 rankings 檔 (inlined from removed run_phase3_hybrid.py)"""
    phase_path = os.path.join(config.OUTPUT_DIR, phase_dir)
    if not os.path.exists(phase_path):
        return None
    candidates = []
    for subdir in os.listdir(phase_path):
        filepath = os.path.join(phase_path, subdir, filename)
        if os.path.exists(filepath):
            candidates.append(filepath)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

config.print_config()


def get_phase5_dir():
    output_dir = os.path.join(config.OUTPUT_DIR, "phase_5")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ══════════════════════════════════════════════════════
#  Q2: 四方法比較
# ══════════════════════════════════════════════════════

def generate_q2_comparison():
    """Q2: 比較 BM25 / Dense / Direct LLM / Your Method 的 Recall@5"""
    output_dir = get_phase5_dir()
    
    phases = [
        ("Phase 1 (BM25)", "phase_1"),
        ("Phase 2 (Dense)", "phase_2"),
        ("Phase 1.5 (Direct LLM)", "phase_direct_llm"),
        ("Phase 4 (Your Method)", "phase_4"),
    ]
    
    results = []
    for name, phase_dir in phases:
        phase_path = os.path.join(config.OUTPUT_DIR, phase_dir)
        if not os.path.exists(phase_path):
            results.append({"method": name, "dev_recall_5": "N/A", "note": "未執行"})
            continue
        
        # 找最新的 metrics.json
        for subdir in sorted(os.listdir(phase_path), reverse=True):
            metrics_path = os.path.join(phase_path, subdir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                results.append({
                    "method": name,
                    "model": m.get("model", ""),
                    "dev_recall_5": m.get("dev_recall_at_5", "N/A"),
                    "note": "",
                })
                break
        else:
            results.append({"method": name, "dev_recall_5": "N/A", "note": "未找到 metrics"})
    
    # 輸出 markdown
    md_lines = ["# Q2: 四方法比較表\n"]
    md_lines.append("| Method | Model | Dev Recall@5 | Note |")
    md_lines.append("|--------|-------|-------------|------|")
    for r in results:
        recall = r["dev_recall_5"]
        if isinstance(recall, float):
            recall = f"{recall:.4f}"
        md_lines.append(f"| {r['method']} | {r.get('model','')} | {recall} | {r.get('note','')} |")
    
    md_lines.append("\n## 優缺點分析\n")
    md_lines.append("### BM25")
    md_lines.append("- **優點**: 速度極快、對精確關鍵字匹配敏感、無需 GPU")
    md_lines.append("- **缺點**: 無法處理同義詞、語意理解弱\n")
    md_lines.append("### Dense Retrieval")
    md_lines.append("- **優點**: 語意理解強、能處理同義詞與換句話說")
    md_lines.append("- **缺點**: 需要 GPU、對精確數字/專有名詞較弱\n")
    md_lines.append("### Direct LLM Selection")
    md_lines.append("- **優點**: 深層推理能力強、能理解複雜問題")
    md_lines.append("- **缺點**: 速度慢、成本高、受 context window 限制\n")
    md_lines.append("### Hybrid + Reranker (Our Method)")
    md_lines.append("- **優點**: 結合多路信號、精確度最高")
    md_lines.append("- **缺點**: 管線複雜、需要多步驟執行\n")
    
    out_path = os.path.join(output_dir, "q2_comparison.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"✅ Q2 comparison saved: {out_path}")


# ══════════════════════════════════════════════════════
#  Q3: 多模態嵌入 vs 文字描述
# ══════════════════════════════════════════════════════

def generate_q3_mm_vs_text():
    """Q3: 多模態嵌入 vs text-description 對照"""
    output_dir = get_phase5_dir()
    
    # 嘗試讀取 Phase 2 和 Phase 2b 的 metrics
    phase2_metrics = None
    phase2b_metrics = None
    
    for phase_dir, target in [("phase_2", "phase2"), ("phase_2b", "phase2b")]:
        phase_path = os.path.join(config.OUTPUT_DIR, phase_dir)
        if os.path.exists(phase_path):
            for subdir in sorted(os.listdir(phase_path), reverse=True):
                mpath = os.path.join(phase_path, subdir, "metrics.json")
                if os.path.exists(mpath):
                    with open(mpath) as f:
                        if target == "phase2":
                            phase2_metrics = json.load(f)
                        else:
                            phase2b_metrics = json.load(f)
                    break
    
    md_lines = ["# Q3: 多模態嵌入 vs 文字描述檢索\n"]
    md_lines.append("## 方法對比\n")
    md_lines.append("| Approach | Model | Dev Recall@5 | Latency | Text Hit Rate | Image Hit Rate |")
    md_lines.append("|----------|-------|-------------|---------|---------------|----------------|")
    
    if phase2_metrics:
        md_lines.append(f"| (b) Text-Description | {phase2_metrics.get('model','')} | "
                        f"{phase2_metrics.get('dev_recall_at_5','N/A'):.4f} | - | - | - |")
    
    if phase2b_metrics:
        md_lines.append(f"| (a) Multimodal Embedding | {phase2b_metrics.get('model','')} | "
                        f"{phase2b_metrics.get('dev_recall_at_5','N/A'):.4f} | "
                        f"{phase2b_metrics.get('dev_time_seconds','N/A'):.1f}s | "
                        f"{phase2b_metrics.get('text_modality_hit_rate','N/A'):.4f} | "
                        f"{phase2b_metrics.get('image_modality_hit_rate','N/A'):.4f} |")
    
    md_lines.append("\n## 選擇與理由\n")
    md_lines.append("基於 Dev Recall@5 的實驗結果，我們選擇了 **text-description** 方法 (b)，因為：")
    md_lines.append("1. 文字描述方法能利用更成熟的 NLP embedding 模型")
    md_lines.append("2. 在候選池較小的 per-question retrieval 場景下，文字描述已足夠表達影像語意")
    md_lines.append("3. 多模態嵌入模型的跨模態分數尺度差異需要額外校正\n")
    md_lines.append("（請根據實際實驗結果更新此段落）\n")
    
    out_path = os.path.join(output_dir, "q3_mm_vs_text.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"✅ Q3 analysis saved: {out_path}")


# ══════════════════════════════════════════════════════
#  Q4: Modality Preference 分析
# ══════════════════════════════════════════════════════

def generate_q4_modality_analysis():
    """Q4: 分析模型的 modality preference"""
    output_dir = get_phase5_dir()
    
    train_data = load_train()
    
    # ── 真實 gold 的 modality 分佈 ────────────────────
    gold_text = 0
    gold_image = 0
    for s in train_data:
        for g in s.get("gold_quotes", []):
            if g.startswith("text"):
                gold_text += 1
            elif g.startswith("image"):
                gold_image += 1
    
    gold_total = gold_text + gold_image
    
    # ── Phase 4 預測的 modality 分佈 ──────────────────
    pred_text = 0
    pred_image = 0
    
    phase4_path = os.path.join(config.OUTPUT_DIR, "phase_4")
    pred_rankings = None
    if os.path.exists(phase4_path):
        for subdir in sorted(os.listdir(phase4_path), reverse=True):
            sub_path = os.path.join(phase4_path, subdir, "rankings_top100.json")
            if os.path.exists(sub_path):
                pred_rankings = load_rankings(sub_path)
                break
    
    # 如果 Phase 4 沒有，試 Phase 3
    if pred_rankings is None:
        hybrid_path = find_latest_rankings("phase_3", "rankings_top100.json")
        if hybrid_path:
            pred_rankings = load_rankings(hybrid_path)
    
    # 如果還是沒有，試 Phase 1
    if pred_rankings is None:
        bm25_path = find_latest_rankings("phase_1", "rankings_top100.json")
        if bm25_path:
            pred_rankings = load_rankings(bm25_path)
    
    if pred_rankings:
        for q_id, ranked in pred_rankings.items():
            for qid in ranked[:5]:
                if qid.startswith("text"):
                    pred_text += 1
                elif qid.startswith("image"):
                    pred_image += 1
    
    pred_total = pred_text + pred_image
    
    # ── 產生圖表 ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 圓餅圖 - Gold
    if gold_total > 0:
        axes[0].pie(
            [gold_text, gold_image],
            labels=["Text", "Image"],
            autopct='%1.1f%%',
            colors=["#4A90D9", "#E74C3C"],
            startangle=90,
        )
        axes[0].set_title("Gold Quotes Modality Distribution")
    
    # 圓餅圖 - Predicted
    if pred_total > 0:
        axes[1].pie(
            [pred_text, pred_image],
            labels=["Text", "Image"],
            autopct='%1.1f%%',
            colors=["#4A90D9", "#E74C3C"],
            startangle=90,
        )
        axes[1].set_title("Predicted Top-5 Modality Distribution")
    
    # 長條圖 - 對比
    x = np.arange(2)
    width = 0.35
    
    gold_pcts = [gold_text/max(gold_total,1)*100, gold_image/max(gold_total,1)*100]
    pred_pcts = [pred_text/max(pred_total,1)*100, pred_image/max(pred_total,1)*100]
    
    bars1 = axes[2].bar(x - width/2, gold_pcts, width, label="Gold", color="#4A90D9")
    bars2 = axes[2].bar(x + width/2, pred_pcts, width, label="Predicted", color="#E74C3C")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["Text", "Image"])
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_title("Gold vs Predicted Modality Comparison")
    axes[2].legend()
    axes[2].set_ylim(0, 100)
    
    # 加上數值標籤
    for bar in bars1:
        axes[2].annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        axes[2].annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "modality_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Modality plots saved: {fig_path}")
    
    # ── 數據表 ────────────────────────────────────────
    analysis = {
        "gold_distribution": {
            "text": gold_text,
            "image": gold_image,
            "total": gold_total,
            "text_pct": gold_text/max(gold_total,1)*100,
            "image_pct": gold_image/max(gold_total,1)*100,
        },
        "predicted_distribution": {
            "text": pred_text,
            "image": pred_image,
            "total": pred_total,
            "text_pct": pred_text/max(pred_total,1)*100,
            "image_pct": pred_image/max(pred_total,1)*100,
        },
        "bias_analysis": {
            "text_bias": (pred_text/max(pred_total,1)*100) - (gold_text/max(gold_total,1)*100),
            "image_bias": (pred_image/max(pred_total,1)*100) - (gold_image/max(gold_total,1)*100),
        }
    }
    
    analysis_path = os.path.join(output_dir, "modality_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"✅ Modality analysis data saved: {analysis_path}")


def main():
    print("=" * 60)
    print("Phase 5: 分析與報告輔助")
    print("=" * 60)
    
    print("\n📊 Generating Q2 comparison...")
    generate_q2_comparison()
    
    print("\n📊 Generating Q3 MM vs Text analysis...")
    generate_q3_mm_vs_text()
    
    print("\n📊 Generating Q4 modality analysis...")
    generate_q4_modality_analysis()
    
    print(f"\n✅ Phase 5 完成! 輸出目錄: {get_phase5_dir()}")


if __name__ == "__main__":
    main()
