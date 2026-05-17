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

最終提交的 pipeline 不使用 retriever，而是把每題*完整 15 個 candidates 直接交給 31B 開放權重 LLM*，由 LLM 一次性挑出 top-5 evidence。流程：

#block(stroke: 0.5pt + luma(180), inset: 8pt, radius: 3pt)[
  *question prompt* + *15 candidates* (text\_quotes + img\_description)
  → 套用 prompt template
  → Gemma-4-31B-it 生成 5 個 `quote_id`
  → 解析輸出，依輸出順序作為 ranked top-5 提交。
]

== 主要 idea

任務有兩個關鍵特性，讓 retrieval 反而是「不必要的瓶頸」：

1. *每題候選池極小 (≈ 15)*，遠小於一般 RAG 的數萬 chunks。即便不截斷文本，整題 prompt 仍可在 LLM context window 範圍內。
2. *evidence selection 依賴細節推理*（比較數字、找跨段落 references 等），這正是 LLM 比 dense embedding 拿手的場景。

因此採用 *「retrieve-then-rank」改為「LLM-as-selector」*：跳過 retriever，由 LLM 直接做 relevance ranking。

== Evidence preprocessing

- *文本 candidates*：*原樣傳入不截斷* (CHAR\_LIMIT=0)，使 LLM 取得完整上下文。
- *圖片 candidates*：採用 dataset 已附的 `img_description`（不重新跑 VLM）。`img_description` 已是高品質自然語言描述，混在 text candidates 中以同等形式呈現給 LLM，模型用 ID prefix (`text*` / `image*`) 自然區辨 modality。Evidence 統一以 `[quote_id]: content` 格式列出。
- *Prompt*：強制要求模型輸出 EXACTLY 5 個 IDs：

  ```
  You are an expert retrieval assistant... extract the top 5 most
  important evidence IDs that best answer the question.
  ...
  You MUST extract and rank EXACTLY 5 evidence IDs in descending order.
  Even if you think fewer than 5 items are relevant, you MUST fill all 5
  spots with your best guesses. DO NOT output fewer than 5 IDs.
  Output ONLY the 5 IDs separated by commas (e.g., text1, image3, ...).
  ```

== Ranking procedure

LLM 輸出的 token 序列即為 ranked list；用正規表示式 `(text|image)[\s\-_]*(\d+)`抓取，能容忍 `text 1` / `text_1` 等變體。過濾不在 allowed set 的 hallucinated IDs；*不做 padding* — 若 LLM 給少於 5 個 ID 即原樣交出（Kaggle 允許 fewer than top-k，亦比隨機補位更穩健）。

== 額外技術

+ *enable\_thinking=False*：透過 NIM 的 `chat_template_kwargs` 顯式關閉 Gemma 內建 reasoning，避免 `max_tokens` 預算被 `<think>` token 佔據。

+ *Sampling 設定 (NIM)*：temperature = 0.1 + top\_p = 0.95、max\_tokens = 2048，給足輸出空間並保留輕微 sampling 自由度。

+ *單執行緒 + 2 s 間隔*：對 NIM 公開 endpoint 採低速率呼叫，避免並行造成的 429 限流大規模 fallback。

= Q2. Comparison of Retrieval Methods (5%)

== 數據

#align(center)[
  #table(
    columns: 5, align: (left, left, center, center, left),
    stroke: 0.5pt + luma(150),
    [*Method*], [*Model*], [*Dev R\@5*], [*Public LB*], [*Note*],
    [*BM25*], [rank\_bm25 (Okapi)], [0.7364], [—], [baseline],
    [*Dense*], [BAAI/bge-m3 (dense)], [0.7389], [—], [single-vec],
    [*LLM (API)*], [Gemma-4-31B-it], [0.8854], [*0.82203*], [#text(rgb("#0a7a3b"))[*best*] — full optim],
    [*LLM (local)*], [Qwen3.6-27B Q6\_K (local)], [*0.8988*], [0.81895], [開放權重本地, 無 API],
  )
]

(另含 Hybrid RRF 等 retrieval-中心 pipeline 之 ablation：4-way BGE-M3 + BM25 RRF + img\_boost 達 Dev 0.8081 / LB 0.72342；加 bge-reranker-v2-gemma 為 0.8068 / 0.72650。皆遠低於 direct-LLM 路線。)

== 強弱分析

*BM25*：詞袋 TF-IDF + 長度正規化。優：CPU 即可、對精確 keyword 命中極快；缺：無法處理同義詞、跨語言、數值/單位推理 — 候選若採用同義表述，將發生 false negatives，無法成功召回 (recall)。

*Dense (BGE-M3)*：以 dual-encoder 將問題與候選嵌入至同一向量空間，依 cosine similarity 排序。優：捕捉語意相似性、對 paraphrase robust；缺：對細微差異（特定數字、實體名）區辨力弱；img\_description 中常見的 templated 描述使分數鑑別度降低，產生大量 false positives。

*Gemma-4-31B*：在 direct LLM 上層層套疊四組優化 — (i) *MUST-5 嚴格指令* + 不截斷候選文本，使 Gemma 能完整看到 evidence；(ii) 關閉 reasoning，避免 token budget 被 `<think>` 區塊佔用；(iii) temp=0.1 + top\_p=0.95，引入輕微 sampling 噪聲；(iv) permissive regex + 無 padding + max\_tokens=2048，能解析「text 1」「text\_1」等變體，且不為了湊滿 5 個 ID 亂猜。

*Qwen3.6-27B Q6\_K*：開放權重本地 GGUF 推理，符合 ≤ 80B 規定且不依賴 API。對 CHAR\_LIMIT (1500 最佳) 與 temperature (0.0 最佳) 做專屬調校。LB 0.81895 略低於 Gemma-4-31B。

== 為何 LLM-based 大幅領先 retrieval

+ *候選池只有 15*：retrieval 的高 recall 在大型 corpus 才重要；此處 LLM 已能在 15 中做全比較。
+ *跨 modality 比較*：retrieval 對「該選 text 還是 image evidence」沒有顯式機制；LLM 在 prompt 內看到兩種 modality 描述，可直接權衡。
+ *組合推理*：許多題需要「合併兩段 evidence 才能回答」(e.g. 引用一個表 + 一句敘述)；retrieval 各自打分難以捕捉這種 *jointly informative*；LLM 一次性 ranking 比較合理。
+ *Domain coverage*：BM25/Dense 在 Academic / Tutorial 等 long-form domain 表現顯著弱於 Financial（domain skew）；LLM 不受此影響。

= Q3. Multimodal Embedding vs. Text-Description Retrieval for Image Evidence (10%)

== 兩種方法

+ *(a) Multimodal Embedding:* 用一個跨模態 encoder（這裡選 `google/siglip-so400m-patch14-384`，1B 參數，符合 ≤ 80B）的 *image tower* 編碼原始圖檔，與問題用 *text tower* 嵌入後 cosine ranking；text candidates 仍走 text tower。
+ *(b) Text-Description Retrieval:* 把 `img_description` 當成純文字 evidence，所有 candidates 統一用 text-only encoder（BAAI/bge-m3，dense single-vec）做檢索；不接觸原始像素。

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

== 選 (b)，因為

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

(以下統計取自 Qwen3.6-27B local 路線之 Dev 200 與 large\_dev 899 預測。Gemma-4-31B 在 dev 上 modality 分布為 text 61.3%/img 38.7%，整體趨勢類似，下方結論皆適用兩條路線。)

== Top-5 modality 分布（Qwen local LB-best，Dev 200 題）

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
    [*Phase / 版本*], [*Method*], [*Dev R\@5*], [*Public LB*],
    [Phase 1], [BM25], [0.7364], [—],
    [Phase 2], [BGE-M3 dense], [0.7389], [—],
    [Phase 2b], [SigLIP-so400m multimodal embedding], [0.1660], [—],
    [Phase 3], [BM25+Dense RRF], [0.7713], [0.68875],
    [Phase 3b], [BGE-M3 4-way (dense+sparse+colbert)+BM25+img\_boost], [0.8081], [0.72342],
    [Phase 3c], [Phase3b + VLM caption concat], [0.8108], [0.72342],
    [Phase 4b], [Phase3b + bge-reranker-v2-gemma 融合], [0.8068], [0.72650],
    [Phase 4c], [Phase3c + reranker (overfit)], [0.8162], [0.72033],
    [Phase 5b], [Ensemble RRF 3b/3c/4b/4c], [0.8152], [0.72650],
    [NIM v1 Gemma], [Gemma-4-31B direct 15→5, soft prompt, CL=800], [0.8848], [0.80970],
    [NIM v2 Gemma], [+ MUST-5 + CL=0 + enable\_thinking=False], [0.8907], [0.82126],
    [NIM v3 Gemma], [v2 + temp=0.1 + top\_p=0.95], [0.8967], [0.81972],
    [*NIM v4 Gemma*], [*v3 + permissive regex + no padding + max\_tokens=2048*], [0.8854], [*0.82203*],
    [Local v1 Qwen Q6\_K], [CL=1500, n\_ctx=8192, temp=0, soft prompt], [*0.8988*], [0.81895],
    [Local v3 Qwen Q6\_K], [+ temp=0.1 + top\_p=0.95 (CL=0)], [0.8794], [0.81355],
    [Local v4 Qwen Q6\_K], [CL=1500 + temp=0.1 + top\_p=0.95 + comma prompt], [0.8881], [0.81587],
    [Local v6 Qwen Q6\_K], [Qwen 上 mirror NIM v4 全部設定], [0.8827], [0.81741],
    [Phase 6 hybrid], [Phase 3b top-10 → Qwen3.6-27B], [0.8759], [0.79583],
  )
]
]

= 附錄 B：執行與重現

```
# 環境
conda activate NLP2
pip install llama-cpp-python  # 本地路線需 CUDA build of llama-cpp-python

# ── Reproduce LB-best (NIM v4 Gemma, LB 0.82203) ──
python run_phase6_direct_llm.py
# → outputs/phase_6/gemma_4_31b_direct_v4_<ts>/submission.csv

# ── Reproduce 本地路線 (Local v1 Qwen, LB 0.81895) ──
export CUDA_VISIBLE_DEVICES=0
N_CTX=8192 CHAR_LIMIT_PER_CANDIDATE=1500 TEMPERATURE=0.0 TOP_P=1.0 \
    CACHE_SUFFIX=_v1 python run_phase6_local.py
# → outputs/phase_6_local/<ts>/submission.csv

# Q3 對照：Multimodal embedding (a)
CUDA_VISIBLE_DEVICES=0 python run_phase2b_multimodal.py

# 評估
python tools/compute_modality.py        # Q4 modality 分析
python tools/eval_large_dev.py          # large_dev (n=899) 重新評估
```

程式碼結構：
- `run_phase6_direct_llm.py`：*LB-best Gemma 主程式*。
- `run_phase6_local.py`：本地 Qwen 替代路線。
- `run_phase1_bm25.py` / `run_phase2_dense.py` / `run_phase2b_multimodal.py`：Q2、Q3 baseline。
- `dataset.py` / `evaluation.py` / `submission_utils.py`：共用工具 (含 `pad_short=False` 控制不補位)。
- `config.py`：所有超參數中央化設定。
- `tools/eval_large_dev.py` / `compute_modality.py` / `build_large_dev.py`：診斷與 large\_dev 評估腳本。
