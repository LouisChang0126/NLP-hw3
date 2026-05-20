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

最終提交的 pipeline 不使用 retriever，而是把每題*完整 15 個 candidates 直接交給開放權重 LLM*，由 LLM 一次性挑出 top-5 evidence。所用模型為 Gemma-4-31B-it (31B 參數；呼叫 NVIDIA NIM API，模型為開源模型)。

#block(stroke: 0.5pt + luma(180), inset: 8pt, radius: 3pt)[
  *question* + *15 candidates* (`text_quotes` + `img_description`)
  → 套用 prompt template
  → Gemma-4-31B-it 生成 5 個 `quote_id`
  → 正規表示式抽取 ID，依輸出順序作為 ranked top-5 提交。
]

== 主要 idea

任務有兩個關鍵特性，讓 retrieval 反而是「不必要的瓶頸」：

1. *每題候選池極小 (≈ 15)*，遠小於一般 RAG 的數萬 chunks。即便不截斷文本，整題 prompt 仍可在 LLM context window 範圍內。
2. *evidence selection 依賴細節推理*（比較數字、找跨段落 references 等），這正是 LLM 比 dense embedding 拿手的場景。

因此採用 *「retrieve-then-rank」改為「LLM-as-selector」*：跳過 retriever，由 LLM 直接做 relevance ranking。

== Evidence preprocessing

- *文本 candidates*：*原樣傳入不截斷*，使 LLM 取得完整上下文。
- *圖片 candidates*：採用 dataset 已附的 `img_description`（不重新跑 VLM）。`img_description` 已是高品質自然語言描述，混在 text candidates 中以同等形式呈現給 LLM，模型用 ID prefix (`text*` / `image*`) 自然區辨 modality。Evidence 統一以 `[quote_id]: content` 格式列出。

== Prompt

```
You are an expert retrieval assistant. I will provide you with a question
and a list of evidence items. Your task is to analyze the evidence and
extract the top 5 most important evidence IDs that best answer the question.

Question: {question}

Evidence Items:
{candidates}

You must extract and rank exactly 5 evidence IDs in descending order of
importance. Even if you think fewer than 5 items are relevant, you MUST
fill all 5 spots with your best guesses. DO NOT output fewer than 5 IDs.
Output ONLY the 5 IDs separated by spaces
(e.g., text1 image3 text5 image2 text10).
```

== Ranking procedure

LLM 輸出的 token 序列即為 ranked list；用正規表示式 `(text|image)[\s\-_]*(\d+)` 抓取，能容忍 `text 1` / `text_1` 等格式變體。過濾不在 allowed set 的 hallucinated IDs；少於 5 個即重新調用API。

== 額外技術

+ *Deterministic decoding*：`temperature = 0.0` + `top_p = 0.1`，使輸出穩定可重現，避免 sampling 造成的 LB 隨機飄移。

+ *enable\_thinking=False*：透過 NIM 的 `chat_template_kwargs` 顯式關閉 Gemma 內建 reasoning，避免 `max_tokens=2048` 之輸出預算被 `<think>` token 佔據。

+ *Min-quotes retry loop*：若單次回應解析出之有效 ID 數 < 5 (`min_quotes=5`)，自動重試最多 12 次並指數型 backoff，最終取所有 attempts 中解析到最多 ID 之回應。此設計能補救偶發 API timeout / 空回應 / 格式錯誤造成的 evidence 漏失。

+ *Evidence modality 過濾*：觀察 train 後得到硬規則 — `evidence_modality_type` 只含 `text` 之題目，gold 100 % 落在 `text*`；只含 `table / figure / chart` 之題目，gold 100 % 落在 `image*`。據此於送入 LLM 前 *prune 掉另一型態之 candidates*：
  - `evidence_modality_type` 含 text 不含 image-like → 移除全部 image candidates
  - `evidence_modality_type` 含 image-like 不含 text → 移除全部 text candidates
  - 兼有兩者或為空 → 不過濾（fallback）

= Q2. Comparison of Retrieval Methods (5%)

== 數據

#align(center)[
  #table(
    columns: 5, align: (left, left, center, center, left),
    stroke: 0.5pt + luma(150),
    [*Method*], [*Model*], [*Dev R\@5*], [*Public LB*], [*Note*],
    [*BM25*], [rank\_bm25 (Okapi)], [0.7364], [—], [baseline],
    [*Dense*], [BAAI/bge-m3 (dense)], [0.7389], [—], [single-vec dual encoder],
    [*Direct LLM*], [Gemma-4-31B-it (baseline)], [0.8848], [0.80970], [Simple prompt, default sampling],
    [*Own method*], [Gemma-4-31B-it (refined)], [0.9057], [#text(rgb("#0a7a3b"))[*0.88597*]], [#text(rgb("#0a7a3b"))[*best*] — deterministic + retry + modality filter],
  )
]

== 強弱分析

*BM25*：詞袋 TF-IDF + 長度正規化。優：CPU 即可、對精確 keyword 命中極快；缺：無法處理同義詞、跨語言、數值/單位推理 — 候選若採用同義表述，將發生 false negatives，無法成功召回。

*Dense (BGE-M3)*：以 dual-encoder 將問題與候選嵌入至同一向量空間，依 cosine similarity 排序。優：捕捉語意相似性、對 paraphrase robust；缺：對細微差異（特定數字、實體名）區辨力弱；`img_description` 中常見的 templated 描述使分數鑑別度降低，產生大量 false positives。

*Direct LLM (baseline)*：直接把 15 個 candidates 餵給 Gemma-4-31B-it 並用簡單 prompt 要求挑出 top-5。優：跳過 retriever 之 false negative，將 ranking 交由具推理能力之 LLM；缺：在 default sampling (temp=0.7, top\_p≈1.0) 下，相同設定多次跑分結果 LB 飄移可達 ±0.5 pp；偶發回應只給 < 5 IDs 之 case 無補救機制。

*Own method (refined)*：在 Direct LLM 之上層層套疊以下優化 — (i) *deterministic decoding* (temp=0, top\_p=0.1) 鎖死採樣使結果可重現；(ii) *關閉 reasoning* 避免 token budget 浪費；(iii) *min-quotes retry loop* (max 12 attempts) 補救少數 < 4 IDs 之 case；(iv) *permissive regex + 無 padding* 容忍輸出格式變體且不亂猜補位；(v) *evidence modality 過濾* (依 `evidence_modality_type` prune 對立 modality candidates) — 單此一項即 +1.39 pp LB (0.87211 → 0.88597)。Dev R\@5 略升 (0.8858 → 0.9057)，*LB 共提升 7.63 pp*。

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

#text(size: 9pt, fill: luma(80))[(評估範圍 Dev 200 題；Image hit-rate = 0 並非隨機，是跨模態 cosine 分布錯位造成的系統性偏差，分析見下。)]

== 選 (b)，因為

+ *對齊現有最佳 pipeline*：最終的 LLM-selection (Q1) 統一吃 text，`img_description` 直接當 text evidence 餵入；用 (b) 與整套上下游 consistent。
+ *品質充足*：dataset 提供的 `img_description` 已是 caption-quality 的自然語言，能傳達圖表中的數值、軸標、趨勢，這些是 cosine-similarity 圖嵌入較弱的維度。
+ *Scale 一致*：(a) 跨模態 cosine 的 image-vs-text 分數常需 calibration（image-image 與 text-text 分數分布不同）；(b) 全 text 統一尺度，rank fusion 更穩。
+ *計算成本低*：text-only 不需處理 14826 張圖片，跑 dev 200 題只需 \~ 30 s vs SigLIP 數百秒。

== 兩者性能差異原因（(a) 較 (b) 表現落後 0.57）

+ *跨模態 cosine 分布錯位，image 系統性排序末端*：SigLIP 的 *image-text cosine* 與 *text-text cosine* 兩個分布之 mean/std 顯著不同（SigLIP 預訓練即採用 sigmoid loss，未強制兩 modality 同尺度）。對 candidates 做 z-score normalize 後排序時，*所有 image candidates 系統性落於分布末端*，導致 top-5 幾乎完全排除 image。實測 image hit-rate = 0/307。此即文獻所稱之 "cross-modal scale gap"，需 per-modality calibration 方能改善。

+ *圖表類圖片屬於 out-of-distribution (OOD)*：HW3 大量出現 charts/tables，SigLIP 預訓練分布以自然影像 (web image-text pairs) 為主，對「金融折線圖」「演算法流程圖」等 chart-heavy 內容之特徵抽取能力較弱。

+ *文字描述已壓縮 question-relevant signal*：dataset 之 `img_description` 多由強模型生成，已抽取圖中 key facts (數值、實體、結論)；text encoder 對此類 keyword 之命中率高。

+ *Text encoder 對長 query 容忍度較高*：SigLIP text tower max\_position\_embeddings 僅 *64 tokens*（實驗中已驗證此 hard limit，超出即 raise `ValueError`），多數 HW3 問題 ≫ 64 tokens 必須 truncate，遺失關鍵 token；BGE-M3 支援 8192 tokens 無此限制。

+ *Text hit-rate 僅 0.568* 顯示即便排除 image 崩潰之影響，SigLIP text tower 的語意檢索能力仍不及專為 retrieval 訓練之 BGE-M3。

= Q4. Modality Preference Analysis (10%)

== 觀察一：未加 modality filter 之原始 LLM 行為偏向 text

未啟用 Q1 之 *evidence modality 過濾* 時（即直接把 15 個 candidates 不過濾餵入 LLM），dev 200 題之 top-5 modality 分布如下：

#align(center)[
  #table(
    columns: 4, align: center, stroke: 0.5pt + luma(150),
    [*Source*], [*Text count (%)*], [*Image count (%)*], [*Total*],
    [Predicted Top-5 (no filter)], [602 (*60.7%*)], [389 (*39.3%*)], [991],
    [Gold answers], [199 (39.3%)], [307 (60.7%)], [506],
    [Bias (pred − gold)], [#text(fill: rgb("#b03030"))[*+21.4 pp*]], [#text(fill: rgb("#b03030"))[*−21.4 pp*]], [—],
  )
]

LLM 在無 prior 之 retrieval 中 *系統性偏好 text*：top-5 有 60.7% 為 text，但 gold 只有 39.3%；image evidence 因此 under-retrieved 約 21.4 pp。

== 偏好成因

+ *Token 佔比不對稱*：`text_quotes` 中位數 \~ 1100 字元；`img_description` 中位數 \~ 250 字元。LLM 在 prompt 中接收到的 text token 量約為 image 的 4×，attention 分布因而偏向 text candidates。

+ *Img\_description 風格較 templated*：image 描述多以「This figure shows…」「The chart illustrates…」起頭，LLM 將之視為敘述性 (descriptive) 而非事實性 (factual)；在 picking task 中傾向選擇含具體數字或引文之 text snippet。

== 觀察二：加上 modality filter 後分布修正至近乎中性

啟用 Q1 之 modality filter 後，dev 200 題之分布變為：

#align(center)[
  #table(
    columns: 4, align: center, stroke: 0.5pt + luma(150),
    [*Source*], [*Text count (%)*], [*Image count (%)*], [*Total*],
    [Predicted Top-5 (w/ filter)], [372 (*37.2%*)], [627 (*62.8%*)], [999],
    [Gold answers], [199 (39.3%)], [307 (60.7%)], [506],
    [Bias (pred − gold)], [#text(fill: rgb("#0a7a3b"))[*−2.1 pp*]], [#text(fill: rgb("#0a7a3b"))[*+2.1 pp*]], [—],
  )
]

Bias 從 ±21.4 pp 收斂至 ±2.1 pp，幾乎完全對齊 gold 分布。

== Per-modality hit-rate （filter on）

#align(center)[
  #table(
    columns: 4, align: (left, center, center, center),
    stroke: 0.5pt + luma(150),
    [*Modality*], [*Gold count*], [*Hits in top-5*], [*Hit-rate*],
    [text],  [199], [154], [0.7739],
    [image], [307], [269], [*0.8762*],
  )
]

Image per-modality hit-rate (0.8762) 高於 text (0.7739)，反映 *當 LLM 真的選 image 時挑得相當準*；filter 之主要作用即在於 *允許 LLM 把原本被 text 擠掉的 image candidates 提到 top-5 之內*。

== 為何 modality filter 同時提升 LB

純單一型態 (text-only 或 image-only) 之題目佔 train 比例頗高 (image-only 樣本占 40.8%)。對這些題目，將對立 modality 之 distractor 整批移除，縮小 candidate pool (15 → 5 或 10)，降低 LLM 比較負擔

實驗驗證：Public LB 由 0.87211 → 0.88597 (+1.39 pp)，與 dev 上之 bias 修正方向一致。

= 附錄 A：執行與重現

```
# ── Reproduce LB-best (Own method, LB 0.88597) ──
# 先把 NIM API key 放到 api_key.txt
python HW3_111550132.py --input test.jsonl --output submission.csv \
               --backup gemma_answering_results.json \
               --min-quotes 4 --max-attempts 12

# ── 對照 baselines (Q2 / Q3) ──
python run_phase1_bm25.py            # BM25
python run_phase2_dense.py           # BGE-M3 dense
python run_phase2b_multimodal.py     # SigLIP (Q3)

# ── 評估與診斷 ──
python compute_modality.py     # Q4 modality 分析
python evaluation.py                 # Dev Recall@5
```

== 程式碼結構

- `HW3_111550132.py`：Gemma-4-31B-it direct 15→5 with deterministic decoding + min-quotes retry。
- `run_phase1_bm25.py` / `run_phase2_dense.py` / `run_phase2b_multimodal.py`：Q2、Q3 baseline。
- `dataset.py` / `evaluation.py` / `submission_utils.py`：共用工具。
- `config.py`：超參數中央化設定。
- `compute_modality.py`：Q4 modality 分析腳本。
