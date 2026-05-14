// =====================================================================
// NLP HW3 — Multimodal Retrieval-Augmented Generation
// Author: <學號> (請於提交前替換)
// Build:  typst compile document/HW3_report.typ document/HW3_<學號>.pdf
// =====================================================================

#set document(title: "INLP HW3 — Multimodal RAG Report")
#set page(paper: "a4", margin: (x: 2.0cm, y: 2.2cm), numbering: "1")
#set text(font: "Noto Serif CJK TC", size: 10.5pt, lang: "zh", region: "tw")
#set par(justify: true, leading: 0.62em)
#show heading.where(level: 1): it => block(below: 0.8em)[
  #set text(weight: "bold", size: 14pt)
  #it
]
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.5em)[
  #set text(weight: "bold", size: 11.5pt)
  #it
]
#show raw.where(block: false): box.with(
  fill: luma(240), inset: (x: 3pt, y: 0pt), outset: (y: 3pt), radius: 2pt,
)
#show raw.where(block: true): block.with(
  fill: luma(245), inset: 8pt, radius: 3pt, width: 100%,
)

// ─── Title block ──────────────────────────────────────────────────
#align(center)[
  #text(size: 18pt, weight: "bold")[INLP 2026 — HW3 Report]
  #v(2pt)
  #text(size: 12pt)[111550132 張家睿]
]

#v(0.5em)

= Q1. Method Description (5%)

== 整體 Pipeline

最終提交的 pipeline 不使用 retriever，而是把每題*完整 15 個 candidates 直接交給 27B 開放權重 LLM*，由 LLM 一次性挑出 top-5 evidence。流程：

#block(stroke: 0.5pt + luma(180), inset: 8pt, radius: 3pt)[
  *問題 (q)* + *15 candidates* (text\_quotes + img\_quotes 的 `img_description`)
  → 套用 prompt template (含 candidate 截斷 char\_limit=1500)
  → Qwen3.6-27B Q6\_K GGUF 生成 5 個 `quote_id`
  → 依輸出順序作為 ranked top-5 提交。
]

== 主要 idea

任務有兩個關鍵特性，讓 retrieval 反而是「不必要的瓶頸」：

1. *每題候選池極小 (≈ 15)*，遠小於一般 RAG 的數萬 chunks。把 15 個原文塞入 ≤ 8K 的 context 完全可行 (15 × 1500 char ≈ 6K tokens)。
2. *evidence selection 依賴細節推理*（比較數字、找跨段落 references 等），這正是 LLM 比 dense embedding 拿手的場景。

因此採用 *"retrieve-then-rank" 改為 "LLM-as-selector"*：跳過 retriever，由 LLM 直接做 relevance ranking。

== Evidence preprocessing

- *文本 candidates*：原樣傳入，僅截斷至 1500 字元（依 dev 細調，1500 比 800/1200/2000 都好，見 Q1 結尾的 ablation 表）。
- *圖片 candidates*：採用 dataset 已附的 `img_description`（不重新跑 VLM）。`img_description` 已是高品質自然語言描述，混在 text candidates 中以同等形式呈現給 LLM，模型用 ID prefix (`text*` / `image*`) 自然區辨 modality。
- *Prompt*：固定 system-like instruction，列出所有 candidates，要求輸出「space-separated 5 個 IDs，不解釋」。
  ```
  Pick exactly 5 candidates most useful for answering the question.
  Output ONLY 5 IDs separated by spaces, in order of relevance.
  Example: text3 image2 text5 text1 image4
  ```

== Ranking procedure

LLM 輸出的 token 序列即為 ranked list；用 regex `(text|image)\d+` 抓取，過濾掉不在 allowed set 的 hallucinated IDs；若數量不足 5，補上 candidate pool 前若干個作 fallback。Temperature = 0 確保可重現。

== 額外技術

+ *Reasoning bypass*：Qwen3.6-27B 預設 chat template 會於 assistant 段落起始處插入 `<think>\n` token，迫使模型先進行 reasoning。`max_tokens=64` 之預算不足以涵蓋完整 `<think>` 區塊，導致 reasoning token 佔用全部 output budget，最終輸出被截斷 (truncated) 且不含實際答案 (早期版本 LB=0.32049 即源於此)。解法為*手動組裝 chat template，預先填入空的 `<think>\n\n</think>\n\n`*，使模型直接略過 reasoning 階段。

  ```python
  prompt = (
      "<|im_start|>user\n" + user + "<|im_end|>\n"
      "<|im_start|>assistant\n"
      "<think>\n\n</think>\n\n"   // ← 略過 reasoning
  )
  ```

+ *Char-limit sweep*：依 dev 結果，CHAR\_LIMIT\_PER\_CANDIDATE = 1500 為最佳值。

  #align(center)[
    #table(
      columns: 4, align: center, stroke: 0.5pt + luma(150),
      [*CHAR\_LIMIT*], [*Dev Recall\@5*], [*Public LB*], [*Note*],
      [800],  [0.8659], [0.81278], [初版],
      [1500], [*0.8988*], [*0.81895*], [#text(rgb("#0a7a3b"))[*best*]],
      [2000], [0.8977], [0.81818], [略遜於 1500],
    )
  ]

+ *Local GGUF inference*：用 `llama-cpp-python` (CUDA build) 載入 Q6\_K 量化 (22 GB)，單卡 RTX 4090 全 layers 上 GPU，`n_ctx=8192, n_batch=512`。Test 1798 題約 27 分鐘完成（≈ 1.0 s/sample）。

+ *Resumable cache*：以 `q_id` 為 key 存 JSON，每 25 題自動寫盤，crash 後可從中斷處 resume。dev 與 test 共用同份 cache 但以 split set 過濾避免 q\_id 撞號。

= Q2. Comparison of Retrieval Methods (5%)

== 數據

#align(center)[
  #table(
    columns: 5, align: (left, left, center, center, left),
    stroke: 0.5pt + luma(150),
    [*Method*], [*Model*], [*Dev R\@5*], [*Public LB*], [*Note*],
    [*BM25*], [rank\_bm25 (Okapi)], [0.7364], [—], [baseline],
    [*Dense*], [BAAI/bge-m3 (dense)], [0.7389], [—], [single-vec],
    [*direct LLM*], [Gemma-4-31B-it (NIM)], [0.8848], [0.80970], [純 LLM 15→5],
    [*Ours*], [Qwen3.6-27B Q6\_K (local)], [*0.8988*], [*0.81895*], [#text(rgb("#0a7a3b"))[*best*]],
  )
]

(另含 Hybrid RRF 等 retrieval-中心 pipeline 之 ablation：4-way BGE-M3 + BM25 RRF + img\_boost 達 Dev 0.8081 / LB 0.72342；加 bge-reranker-v2-gemma 為 0.8068 / 0.72650。皆遠低於 direct-LLM。)

== 強弱分析

*BM25*：詞袋 TF-IDF + 長度正規化。優：CPU 即可、對精確 keyword 命中極快；缺：無法處理同義詞、跨語言、數值/單位推理 — 候選若採用同義表述，將發生 false negatives，無法成功召回 (recall)。

*Dense (BGE-M3)*：以 dual-encoder 將問題與候選嵌入至同一向量空間，依 cosine similarity 排序。優：捕捉語意相似性、對 paraphrase robust；缺：對細微差異（特定數字、實體名）區辨力弱；img\_description 中常見的 templated 描述使分數鑑別度降低，產生大量 false positives。

*Direct LLM selection (Gemma-4-31B)*：將 15 候選與問題一同輸入 prompt，由 LLM 一次選 5。優：深層推理、跨 candidate 比較、modality 自然融合；缺：依賴 API/GPU、單題 latency 高（NIM 平均 \~ 3.5 s/題）。

*Ours (Qwen3.6-27B Q6\_K local)*：在 direct-LLM 上施加四項優化 — (i) 開放權重本地推理（符合 ≤ 80B 規定，無 API quota）；(ii) reasoning bypass 解決因 reasoning token 佔用導致輸出被截斷 (truncated) 的問題；(iii) CHAR\_LIMIT 自 800 提升至 1500，每 candidate 訊息量提升約 2×；(iv) 確定性解碼搭配可恢復 cache。

== 為何 LLM-based 大幅領先 retrieval

+ *候選池只有 15*：retrieval 的高 recall 在大型 corpus 才重要；此處 LLM 已能在 15 中做全比較。
+ *跨 modality 比較*：retrieval 對「該選 text 還是 image evidence」沒有顯式機制；LLM 在 prompt 內看到兩種 modality 描述，可直接權衡。
+ *組合推理*：許多題需要「合併兩段 evidence 才能回答」(e.g. 引用一個表 + 一句敘述)；retrieval 各自打分難以捕捉這種 *jointly informative*；LLM 一次性 ranking 比較合理。
+ *Domain coverage*：BM25/Dense 在 Academic / Tutorial 等 long-form domain 表現顯著弱於 Financial（domain skew）；LLM 不受此影響。

= Q3. Multimodal Embedding vs. Text-Description Retrieval for Image Evidence (10%)

== 兩種方法

+ *(a) Multimodal Embedding* — 用一個跨模態 encoder（這裡選 `google/siglip-so400m-patch14-384`，1B 參數，符合 ≤ 80B）的 *image tower* 編碼原始圖檔，與問題用 *text tower* 嵌入後 cosine ranking；text candidates 仍走 text tower。
+ *(b) Text-Description Retrieval* — 把 `img_description` 當成純文字 evidence，所有 candidates 統一用 text-only encoder（BAAI/bge-m3，dense single-vec）做檢索；不接觸原始像素。

== 實驗結果（Dev set, 200 題）

#align(center)[
  #table(
    columns: 5, align: (left, left, center, center, center),
    stroke: 0.5pt + luma(150),
    [*Approach*], [*Model*], [*Dev R\@5*], [*Text hit-rate*], [*Image hit-rate*],
    [(a) Multimodal Embed], [SigLIP-so400m (1B)], [*0.1660*], [113/199 = 0.568], [#text(fill: rgb("#b03030"))[*0/307 = 0.000*]],
    [(b) Text Description], [BAAI/bge-m3 (567M)], [*0.7389*], [—], [—],
  )
]

#text(size: 9pt, fill: luma(80))[(Phase 2b 評估範圍 Dev 200 題；Image hit-rate = 0 並非隨機，是跨模態 cosine 分布錯位造成的系統性偏差，分析見下。)]

== 我們選 (b)，原因

+ *對齊現有最佳 pipeline*：最終的 LLM-selection (Q1) 統一吃 text，img\_description 直接當 text evidence 餵入；用 (b) 與整套上下游 consistent。
+ *品質充足*：dataset 提供的 `img_description` 已是 caption-quality 的自然語言，能傳達圖表中的數值、軸標、趨勢，這些是 cosine-similarity 圖嵌入較弱的維度。
+ *Scale 一致*：(a) 跨模態 cosine 的 image-vs-text 分數常需 calibration（image-image 與 text-text 分數分布不同）；(b) 全 text 統一尺度，rank fusion 更穩。
+ *計算成本低*：text-only 不需處理 14826 張圖片，跑 dev 200 題只需 \~ 30s vs SigLIP 數百秒。

== 兩者性能差異原因（(a) 較 (b) 表現落後 0.57）

+ *跨模態 cosine 分布錯位，image 系統性排序末端*：SigLIP 的 *image-text cosine* 與 *text-text cosine* 兩個分布之 mean/std 顯著不同（SigLIP 預訓練即採用 sigmoid loss，未強制兩 modality 同尺度）。對 candidates 做 z-score normalize 後排序時，*所有 image candidates 系統性落於分布末端*，導致 top-5 幾乎完全排除 image。實測 image hit-rate = 0/307。此即文獻所稱之 "cross-modal scale gap"，需 per-modality calibration 方能改善，但對下游影響範圍有限。

+ *圖表類圖片屬於 out-of-distribution (OOD)*：HW3 大量出現 charts/tables，SigLIP 預訓練分布以自然影像 (web image-text pairs) 為主，對「金融折線圖」「演算法流程圖」等 chart-heavy 內容之特徵抽取能力較弱。

+ *文字描述已壓縮 question-relevant signal*：dataset 之 `img_description` 多由強模型生成，已抽取圖中 key facts (數值、實體、結論)；text encoder 對此類 keyword 之命中率高。

+ *Text encoder 對長 query 容忍度較高*：SigLIP text tower max\_position\_embeddings 僅 *64 tokens*（實驗中已驗證此 hard limit，超出即 raise `ValueError`），多數 HW3 問題 ≫ 64 tokens 必須 truncate，遺失關鍵 token；BGE-M3 支援 8192 tokens 無此限制。
+ *Text hit-rate 僅 0.568* 顯示即便排除 image 崩潰之影響，SigLIP text tower 的語意檢索能力仍不及專為 retrieval 訓練的 BGE-M3（後者 Dev R\@5 = 0.7389，文本與 img\_description 一律以 text 形式處理）。

= Q4. Modality Preference Analysis (10%)

== Top-5 modality 分布（最佳 pipeline，Dev 200 題）

#align(center)[
  #table(
    columns: 4, align: center, stroke: 0.5pt + luma(150),
    [*Source*], [*Text count (%)*], [*Image count (%)*], [*Total*],
    [Predicted Top-5], [652 (*65.2%*)], [348 (*34.8%*)], [1000],
    [Gold answers],   [199 (39.3%)],   [307 (60.7%)],   [506],
    [Bias (pred − gold)], [#text(fill: rgb("#b03030"))[*+25.9 pp*]], [#text(fill: rgb("#b03030"))[*−25.9 pp*]], [—],
  )
]

== 結論：模型 *過度偏好 text*

LLM-selector top-5 中 65% 為 text，但 ground-truth 僅 39% 為 text；image evidence 受到 *系統性 under-retrieval*，差距約 26 個百分點。

== Per-modality hit-rate （更細的視角）

#align(center)[
  #table(
    columns: 4, align: (left, center, center, center),
    stroke: 0.5pt + luma(150),
    [*Modality*], [*Gold count*], [*Hits in top-5*], [*Hit-rate*],
    [text],  [199], [153], [0.7688],
    [image], [307], [270], [*0.8795*],
  )
]

值得注意的現象：*當 LLM 選擇 image candidates 時，per-modality precision 反而較高 (0.88 > 0.77)*；惟其選取 image 之數量偏低（僅佔 top-5 的 35%），整體 image evidence 仍有 12% (37/307) 之 false-negative 漏失。

== 形成原因分析

+ *Image candidates 訊號量受 char\_limit 削弱*：`img_description` 通常較短（中位數 \~ 250 字元），被 char\_limit=1500 截斷之機率極低；text\_quotes 中位數 \~ 1100 字元，於 prompt 中之 token 佔比約 4×。LLM 接收之 text token 顯著多於 image token，attention 分布因而偏向 text candidates。

+ *Img\_description 風格趨於 templated*：許多 image 描述以「This figure shows…」「The chart illustrates…」開頭，LLM 將其視為敘述性 (descriptive) 而非事實性 (factual) 內容；在 picking task 中傾向選擇含具體數字或引文之 text snippet，而非 chart description。

+ *Question 含 textual anchor 觸發 shortcut*：HW3 問題常出現「In the section about X…」「According to Table 3…」等 textual anchor；LLM 過度依賴文字錨點 (textual anchors) 進行表面特徵匹配，造成 text-favored shortcut。

+ *Image per-modality precision 較高之解讀*：當 image candidate 與問題語意明確對齊（例如問「該圖顯示什麼」），其被選中之機率極高；hit-rate 較高之原因在於 LLM 對 image 候選採取較保守的選擇策略 (high-precision, low-recall behavior)。Text 因候選數量多、訊號雜訊比較低，false-positive 數量相對較高。

== 改進方向（未實作）

- *Length-normalized prompt*：將 text candidates 壓縮至與 img\_description 相近之長度（例如統一 char\_limit=400），有助於使 modality 分布趨於中性。
- *Explicit modality budget*：於 prompt 中加入「expected mix \~ 40% text / 60% image」之顯式指示，使 LLM 進行 modality balancing。
- *Two-pass selection*：第一階段於 text candidates 中選出 top-3，第二階段於剩餘 image candidates 中補足 2 項。

= 附錄 A：完整 Leaderboard

#text(size: 9pt)[
#align(center)[
  #table(
    columns: 4, align: (left, left, center, center),
    stroke: 0.5pt + luma(180),
    [*Phase*], [*Method (短描述)*], [*Dev R\@5*], [*Public LB*],
    [Phase 1], [BM25], [0.7364], [—],
    [Phase 2], [BGE-M3 dense], [0.7389], [—],
    [Phase 3], [BM25+Dense RRF], [0.7713], [0.68875],
    [Phase 3b], [BGE-M3 4-way (dense+sparse+colbert)+BM25+img\_boost], [0.8081], [0.72342],
    [Phase 3c], [Phase3b + VLM caption concat], [0.8108], [0.72342],
    [Phase 4b], [Phase3b + bge-reranker-v2-gemma 融合], [0.8068], [0.72650],
    [Phase 4c], [Phase3c + reranker (overfit)], [0.8162], [0.72033],
    [Phase 5b], [Ensemble RRF 3b/3c/4b/4c], [0.8152], [0.72650],
    [Phase 6 (NIM)], [Gemma-4-31B direct 15→5], [0.8848], [0.80970],
    [*Phase 6 local*], [*Qwen3.6-27B Q6\_K, CL=1500 (best)*], [*0.8988*], [*0.81895*],
    [Phase 6 long-ctx], [Gemma CL=2000], [0.8986], [0.79506],
    [Phase 6 (35B-A3B)], [Qwen3.6-35B-A3B Q4\_K\_M MoE], [0.8255], [— (skip)],
    [Phase 6 hybrid], [Phase 3b top-10 → Qwen3.6-27B], [0.8759], [0.79583],
  )
]
]

= 附錄 B：執行與重現

```
# 環境
conda activate NLP2
pip install llama-cpp-python  # 需 CUDA build

# Reproduce 最佳結果 (Phase 6 local, LB 0.81895)
export CUDA_VISIBLE_DEVICES=0
N_CTX=8192 CHAR_LIMIT_PER_CANDIDATE=1500 \
  python run_phase6_local.py
# → outputs/phase_6_local/<timestamp>/submission.csv

# Q3 對照：Multimodal embedding (a)
CUDA_VISIBLE_DEVICES=0 python run_phase2b_multimodal.py
# → outputs/phase_2b/google_siglip-so400m-patch14-384_<ts>/

# 評估
python tools/compute_modality.py  # Q4 modality 分析
```

源碼結構（提交至 E3）：
- `run_phase6_local.py`：最終最佳 pipeline (主程式)。
- `run_phase6_direct_llm.py`：NIM Gemma 版本 (Q2 對照)。
- `run_phase1_bm25.py` / `run_phase2_dense.py` / `run_phase2b_multimodal.py`：Q2、Q3 baseline。
- `dataset.py` / `evaluation.py` / `submission_utils.py`：共用工具。
- `config.py`：所有超參數中央化設定。
