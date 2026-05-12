"""
INLP HW3 - Phase 0: 基礎建設與資料探索 (EDA)
任務 3 & 4: 統計分析 + 影像可讀性驗證
"""
import json
import os
import random
from collections import Counter

import config
from dataset import load_train, load_test, parse_sample, build_candidates, get_img_path

config.print_config()

random.seed(config.SEED)


def run_eda():
    train_data = load_train()
    test_data = load_test()
    
    print(f"\n📊 載入完成: train={len(train_data)}, test={len(test_data)}")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("INLP HW3 - Phase 0: EDA Report")
    report_lines.append("=" * 70)
    
    # ── 基本統計 ──────────────────────────────────────
    report_lines.append(f"\n📌 資料集概況")
    report_lines.append(f"  Train samples: {len(train_data)}")
    report_lines.append(f"  Test samples:  {len(test_data)}")
    
    # ── 問題長度統計 ──────────────────────────────────
    train_q_lens = [len(s["question"]) for s in train_data]
    test_q_lens = [len(s["question"]) for s in test_data]
    
    report_lines.append(f"\n📌 問題長度 (字元數)")
    report_lines.append(f"  Train: min={min(train_q_lens)}, max={max(train_q_lens)}, "
                        f"mean={sum(train_q_lens)/len(train_q_lens):.1f}")
    report_lines.append(f"  Test:  min={min(test_q_lens)}, max={max(test_q_lens)}, "
                        f"mean={sum(test_q_lens)/len(test_q_lens):.1f}")
    
    # ── text_quotes / img_quotes 數量分佈 ─────────────
    train_text_counts = [len(s.get("text_quotes", [])) for s in train_data]
    train_img_counts = [len(s.get("img_quotes", [])) for s in train_data]
    test_text_counts = [len(s.get("text_quotes", [])) for s in test_data]
    test_img_counts = [len(s.get("img_quotes", [])) for s in test_data]
    
    report_lines.append(f"\n📌 候選池大小分佈")
    report_lines.append(f"  Train text_quotes: min={min(train_text_counts)}, max={max(train_text_counts)}, "
                        f"mean={sum(train_text_counts)/len(train_text_counts):.1f}")
    report_lines.append(f"  Train img_quotes:  min={min(train_img_counts)}, max={max(train_img_counts)}, "
                        f"mean={sum(train_img_counts)/len(train_img_counts):.1f}")
    report_lines.append(f"  Train total candidates: "
                        f"mean={sum(a+b for a,b in zip(train_text_counts, train_img_counts))/len(train_data):.1f}")
    report_lines.append(f"  Test text_quotes:  min={min(test_text_counts)}, max={max(test_text_counts)}, "
                        f"mean={sum(test_text_counts)/len(test_text_counts):.1f}")
    report_lines.append(f"  Test img_quotes:   min={min(test_img_counts)}, max={max(test_img_counts)}, "
                        f"mean={sum(test_img_counts)/len(test_img_counts):.1f}")
    
    # ── gold_quotes modality 分析 ─────────────────────
    report_lines.append(f"\n📌 Gold Quotes Modality 分佈 (Train only)")
    gold_text_count = 0
    gold_img_count = 0
    gold_per_sample_counts = []
    
    for s in train_data:
        golds = s.get("gold_quotes", [])
        gold_per_sample_counts.append(len(golds))
        for g in golds:
            if g.startswith("text"):
                gold_text_count += 1
            elif g.startswith("image"):
                gold_img_count += 1
    
    total_golds = gold_text_count + gold_img_count
    report_lines.append(f"  Total gold quotes: {total_golds}")
    report_lines.append(f"  Text golds: {gold_text_count} ({gold_text_count/total_golds*100:.1f}%)")
    report_lines.append(f"  Image golds: {gold_img_count} ({gold_img_count/total_golds*100:.1f}%)")
    report_lines.append(f"  Gold quotes per sample: min={min(gold_per_sample_counts)}, "
                        f"max={max(gold_per_sample_counts)}, "
                        f"mean={sum(gold_per_sample_counts)/len(gold_per_sample_counts):.2f}")
    
    # gold 數量分佈
    gold_count_dist = Counter(gold_per_sample_counts)
    report_lines.append(f"  Gold count distribution:")
    for k in sorted(gold_count_dist.keys()):
        report_lines.append(f"    {k} golds: {gold_count_dist[k]} samples")
    
    # ── evidence_modality_type 分佈 ───────────────────
    report_lines.append(f"\n📌 Evidence Modality Type 分佈")
    # evidence_modality_type 可能是 list，需要轉成可 hash 的 tuple
    modality_counter = Counter()
    for s in train_data:
        emt = s.get("evidence_modality_type", "N/A")
        if isinstance(emt, list):
            modality_counter[tuple(sorted(emt))] += 1
        else:
            modality_counter[str(emt)] += 1
    for mod, cnt in modality_counter.most_common():
        report_lines.append(f"  {mod}: {cnt} ({cnt/len(train_data)*100:.1f}%)")
    
    # ── Domain 分佈 ───────────────────────────────────
    report_lines.append(f"\n📌 Domain 分佈")
    train_domains = Counter(s.get("domain", "N/A") for s in train_data)
    test_domains = Counter(s.get("domain", "N/A") for s in test_data)
    
    report_lines.append(f"  Train domains:")
    for d, c in train_domains.most_common():
        report_lines.append(f"    {d}: {c} ({c/len(train_data)*100:.1f}%)")
    report_lines.append(f"  Test domains:")
    for d, c in test_domains.most_common():
        report_lines.append(f"    {d}: {c} ({c/len(test_data)*100:.1f}%)")
    
    # ── question_type 分佈 ────────────────────────────
    report_lines.append(f"\n📌 Question Type 分佈")
    qt_counter = Counter(s.get("question_type", "N/A") for s in train_data)
    for qt, cnt in qt_counter.most_common():
        report_lines.append(f"  {qt}: {cnt} ({cnt/len(train_data)*100:.1f}%)")
    
    # ── 任務 4：驗證影像可讀 ──────────────────────────
    report_lines.append(f"\n📌 影像可讀性驗證 (隨機抽 20 張)")
    all_img_paths = []
    for s in train_data + test_data:
        for iq in s.get("img_quotes", []):
            all_img_paths.append(iq.get("img_path", ""))
    
    report_lines.append(f"  資料集中共有 {len(all_img_paths)} 筆 img_path 引用")
    unique_imgs = list(set(all_img_paths))
    report_lines.append(f"  不重複圖片: {len(unique_imgs)}")
    
    # 隨機抽樣
    sample_imgs = random.sample(unique_imgs, min(20, len(unique_imgs)))
    success = 0
    fail = 0
    
    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False
        report_lines.append("  ⚠️ Pillow 未安裝，無法驗證影像可讀性")
    
    if has_pil:
        for img_rel in sample_imgs:
            img_abs = get_img_path(img_rel)
            try:
                img = Image.open(img_abs)
                img.verify()  # 驗證可讀
                success += 1
                report_lines.append(f"  ✅ {img_rel} -> OK ({img.size})")
            except Exception as e:
                fail += 1
                report_lines.append(f"  ❌ {img_rel} -> FAIL: {e}")
        
        report_lines.append(f"\n  影像驗證結果: {success}/{success+fail} 成功")
    
    # ── 輸出報告 ──────────────────────────────────────
    output_dir = os.path.join(config.OUTPUT_DIR, "phase_0")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eda_report.txt")
    
    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ EDA report saved: {report_path}")


if __name__ == "__main__":
    run_eda()
