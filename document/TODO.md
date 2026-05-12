INLP HW3 Multimodal RAG - Vibe Coding TODO

這是一份給 AI 輔助開發工具的任務清單。請依照階段（Phase）依序執行。每個階段完成後，請確保程式碼能順利運行，並將結果存入對應的輸出資料夾。

專案架構說明

原始資料目錄：data/ (Kaggle 下載：`train.jsonl` 2055 筆含 `gold_quotes`、`test.jsonl` 1798 筆無標籤、`image/` 14826 張、`sample_submission.csv`)

輸出目錄：outputs/phase_{i}/{model}_{MMDDHHMM}/

設定檔：config.py (控管所有路徑與各階段選用的開源模型)

⚠️ 重要：資料結構與檢索範圍
- 每一筆 sample 自帶該題的 `text_quotes` 與 `img_quotes` 候選清單，**檢索範圍是「該題候選池」（per-question retrieval），不是跨題的全域 corpus**。
- 所有 retriever（BM25 / Dense / Hybrid / Reranker / Direct LLM）都對 **單題的 `text_quotes ∪ img_quotes`** 進行排序，輸出該題 Top-5 `quote_id`。
- 約束：模型總參數 ≤ 80B、open-weight only。禁用 GPT-4/Claude 等閉源 API。

Setup Checklist (動工前先做)

[ ] 在 Kaggle 將 team name 改為 `<student ID>`（不改扣 5%）。
[ ] 建立 `requirements.txt`：`rank_bm25`、`sentence-transformers`、`FlagEmbedding`、`faiss-cpu`、`transformers`、`torch`、`pandas`、`tqdm`、`matplotlib`、`scikit-learn`。
[ ] 固定亂數種子 `SEED=42`（dataset split / sampling / 任何 shuffle 都套用）。
[ ] 在每次 run 開頭印出 `config.py` 快照進 log，便於追溯 outputs 目錄對應的設定。
[ ] deadline：E3 繳交 2026-05-26 23:59（無遲交）；Kaggle 每天最多 5 次提交。

## Phase 0: 基礎建設與資料探索 (EDA)

[ ] 任務 1：資料讀取。撰寫 `dataset.py`，從 `data/` 目錄讀取 `train.jsonl` (2055) 與 `test.jsonl` (1798)。每筆 sample 解析 `q_id`、`doc_name`、`domain`、`question`、`evidence_modality_type`、`text_quotes`（list of {quote_id, text}）、`img_quotes`（list of {quote_id, img_path, img_description}）、以及 train 才有的 `gold_quotes`。

[ ] 任務 2：候選池建構（per-question）。**不要做全域 corpus**。為每題建立一個候選清單 `candidates[q_id] = [{quote_id, modality, text_for_retrieval}]`，其中：
    - `text_quotes` 的 `text_for_retrieval = text`、`modality='text'`
    - `img_quotes` 的 `text_for_retrieval = img_description`、`modality='image'`
    - 保留每筆 `quote_id`，確保最終輸出時可正確還原。

[ ] 任務 3：EDA。統計問題長度、`text_quotes` / `img_quotes` 的數量分佈、`gold_quotes` 的 modality 真實比例、跨 domain 的分佈，並將觀察寫入 `outputs/phase_0/eda_report.txt`。

[ ] 任務 4：驗證影像可讀。隨機抽 20 張 `img_path` 嘗試載入（之後 Phase 2b 會用到），結果寫進 EDA report。

## Phase 0.5: 本地驗證框架 (Local Validation)

Kaggle 每天最多 5 次提交、public LB 只佔 30%，**所有實驗都先在本地 dev set 上驗證再上傳**。

[ ] 任務 1：dev split。從 `train.jsonl` (2055 筆都有 `gold_quotes`) 固定切出 hold-out dev（建議用 `seed=42` 取最後 200 筆，或 stratify by `domain`）。輸出 `data/splits/dev_ids.json`。

[ ] 任務 2：實作 `evaluation.py`：`recall_at_k(preds: dict[q_id -> list[quote_id]], golds: dict[q_id -> list[quote_id]], k=5) -> float`。同時提供 per-sample 結果以便錯誤分析。

[ ] 任務 3：每個 Phase 跑完後，**強制**在 dev 上跑 Recall@5 並寫進該 phase output 目錄 `metrics.json`。

[ ] 任務 4：維護 `outputs/leaderboard.md`，欄位：phase、model、dev_recall@5、public_LB、submissions_today、備註。每次跑完手動 append。

## Phase 1: Simple Baseline (純文字稀疏檢索 BM25)

[ ] 任務 1：建立 BM25 檢索器（per-question）。使用 `rank_bm25`，對 **每題自己的候選池** 即時建索引（候選池小，無需 ElasticSearch）。

[ ] 任務 2：執行檢索。對該題的候選做 BM25 排序，取 Top-5（並保留 Top-10 / Top-100 給後續 RRF 用）。

[ ] 任務 3：產生提交檔，**精確** 對齊官方格式：
    - CSV 兩欄、欄名 **必須** 為 `q_id, gold_quotes`
    - `gold_quotes` 是 **單一空白** 分隔的 5 個 `quote_id`
    - 行數要等於 `test.jsonl` 筆數，`q_id` 一一對應
    - 在腳本內加 assert 驗證欄名 / 每列 token 數 ≤ 5
    - 寫入 `outputs/phase_1/bm25_MMDDHHMM/submission.csv`

[ ] 任務 4：dev 評估。對 dev set 跑同一支 retriever，記 Recall@5 到 `metrics.json`，append 到 `outputs/leaderboard.md`。

Phase 1.5: Direct LLM Selection Baseline (Q2 必考)

報告 Q2 要比較 BM25 / Dense / **Direct LLM Selection** / 自己的方法。此 phase 提供「直接讓 LLM 挑 Top-5」的對照組。

[ ] 任務 1：選模型。`config.py` 加上 `PHASE_DIRECT_LLM_MODEL`（建議 `Qwen2.5-7B-Instruct` 或 `Llama-3.1-8B-Instruct`，受顯存允許可上 70B；嚴守 ≤80B + open-weight）。

[ ] 任務 2：Prompt 設計。將該題 `question` + 全部候選（每筆含 `quote_id` 與 `text` 或 `img_description`）塞進 prompt，要求模型輸出 JSON `{"top5": ["quote_id_a", ...]}`；若候選超出 context 上限，採分批 pointwise 評分後合併排序。

[ ] 任務 3：解析 + 驗證。對輸出做 schema 驗證；若解析失敗或數量不足，fallback 用 BM25 Top-5 補位。

[ ] 任務 4：輸出 `outputs/phase_direct_llm/{model}_{MMDDHHMM}/submission.csv` 並在 dev 上算 Recall@5、寫 `metrics.json` 和 `leaderboard.md`。

## Phase 2: 稠密檢索 (Dense Retrieval)

[ ] 任務 1：模型載入。根據 `config.py` 中的 `PHASE_2_EMBED_MODEL`（建議 `BAAI/bge-m3` 或 `intfloat/multilingual-e5-large`）載入 HuggingFace 模型。

[ ] 任務 2：建立向量索引（per-question）。對每題的候選池（text + img_description）算 embedding，存到記憶體 / 該題自己的小 FAISS index。也可以批次預算後 cache 成 `outputs/phase_2/embeddings.npy` 加速後續 phase 重用。

[ ] 任務 3：執行純向量檢索。Query 也做 embedding，cosine similarity 取 Top-5（保留 Top-100 給 Phase 3 / Phase 4）。

[ ] 任務 4：dev 評估 + 提交檔。同 Phase 1 任務 3 的格式驗證流程，輸出 `outputs/phase_2/{model}_{MMDDHHMM}/submission.csv` + `metrics.json`，append 到 `leaderboard.md`。

## Phase 2b: Multimodal Embedding 檢索 (Q3 必考實驗，10%)

報告 Q3 要求對比 **(a) 直接嵌入原始圖片** vs **(b) 圖→文字描述後純文字檢索**。Phase 2 已完成 (b)；本 phase 補齊 (a)。

[ ] 任務 1：模型選型。`config.py` 增加 `PHASE_2B_MM_MODEL`，建議 `openai/clip-vit-large-patch14` 或 `google/siglip-so400m-patch14-384`；進階可試 `vidore/colqwen2.5-v0.2`（多向量、≈3B，注意顯存）。

[ ] 任務 2：對 `img_quotes` 用 image encoder 編碼原始圖檔；對 `question` 與 `text_quotes` 用同模型的 text tower 編碼（保證共享向量空間）。

[ ] 任務 3：per-question 計算 query 對候選的相似度，產生 image / text 同表的分數。**注意 modality 分數尺度差異**：建議改用 RRF 或 z-score 正規化後再合併。

[ ] 任務 4：dev 評估 + submission。同格式輸出 `outputs/phase_2b/{model}_{MMDDHHMM}/submission.csv`。

[ ] 任務 5：產生 Q3 對照數據。將 (a) 多模態嵌入 vs (b) text-description (Phase 2) 的 dev Recall@5、latency、modality 命中率寫進 `outputs/phase_2b/comparison.json`，並挑 3~5 筆兩者預測不同的案例存到 `outputs/phase_2b/case_study.md`。

[ ] (可選加分) 任務 6：用較強 VLM（如 `Qwen2.5-VL-7B-Instruct`，≤80B）對所有 `img_path` 重新生成更精細的 caption，存成 `data/img_description_v2.jsonl`。重跑 Phase 1 / Phase 2 比較原版 vs v2 描述對 Recall@5 的影響。

## Phase 3: 混合檢索 (Hybrid Search: BM25 + Dense) - 目標超越 Strong Baseline

[ ] 任務 1：實作 Reciprocal Rank Fusion (RRF)。讀取 Phase 1 (BM25) 與 Phase 2 (Dense) 的 per-question Top-100 排名。`RRF_score(d) = Σ 1/(k + rank_i(d))`，常數 `k=60`。

[ ] 任務 2：分數融合。對該題候選池套 RRF，可選擇對 image 候選加上權重（modality bias 校正）。也可加 Phase 2b 的多模態分數做三路融合。

[ ] 任務 3：dev 評估 + 提交檔。取 RRF 後 Top-5，沿用 Phase 1 任務 3 的格式驗證，輸出 `outputs/phase_3/{name}_{MMDDHHMM}/submission.csv` 與 `metrics.json`。

## Phase 4: SOTA 策略 (Query Expansion + Cross-Encoder Reranking) - 衝擊 Leaderboard 前段班

[ ] 任務 1：HyDE (Query Expansion)。使用 `config.py` 中的 `PHASE_4_EXPANSION_MODEL`（建議 `Llama-3.1-8B-Instruct` 起跳，資源夠用可上 70B）。將 HyDE prompt 模板（注入 `doc_name` / `domain` / `question`）存到 `prompts/hyde.txt`，供 Q1 報告引用。輸出：`{原 Question} + {假想回答}` 作為新 query，重跑 Phase 3 混合檢索取 Top-100 候選。

[ ] 任務 2：Cross-Encoder 重排序。`PHASE_4_RERANKER_MODEL` 預設 `BAAI/bge-reranker-v2-m3`（568M，單卡可跑）；fallback 較小：`BAAI/bge-reranker-base`；進階：`Qwen3-Reranker-8B`（≤80B、顯存要 ~16GB）。在 `config.py` 內留 model id list 並標註顯存需求。將 Top-100 的 `(query, candidate_text)` 配對輸入模型取得相關性分數。

[ ] 任務 3：dev 評估 + 提交檔。Reranker 分數降冪取 Top-5，沿用格式驗證輸出 `outputs/phase_4/{name}_{MMDDHHMM}/submission.csv` 與 `metrics.json`，append `leaderboard.md`。

## Phase 5: 分析與報告輔助 (對應 Q1 ~ Q4 全部報告題)

報告佔 30%（Q1 5%、Q2 5%、Q3 10%、Q4 10%）。本 phase 把前面所有實驗結果彙整成報告素材，最後打包繳交。

[ ] 任務 1 (Q1 5%)：方法描述素材。把最終 pipeline（retriever / preprocessing / ranking / 額外技巧）畫成流程圖（Mermaid 或 draw.io）存 `outputs/phase_5/q1_pipeline.png`。列出每階段選用的模型 id、超參、prompt 連結。

[ ] 任務 2 (Q2 5%)：四方法比較表。從 `outputs/phase_1` (BM25)、`outputs/phase_2` (Dense)、`outputs/phase_direct_llm` (Direct LLM)、`outputs/phase_4` (your method) 抓 dev Recall@5 與 public LB，輸出 `outputs/phase_5/q2_comparison.md`（含每方法的優缺點、推測差異原因）。

[ ] 任務 3 (Q3 10%)：多模態嵌入 vs 文字描述對照。從 Phase 2b 拉 (a) vs (b) 的 dev Recall@5、retrieval latency、3~5 筆兩者預測不同的 case study，輸出 `outputs/phase_5/q3_mm_vs_text.md`。明確寫出「你選了哪一條 + 為什麼」。

[ ] 任務 4 (Q4 10%)：modality preference 分析。
    - 對 **Phase 4 最終 Top-5** 統計 `modality=='text'` vs `modality=='image'` 的比例。
    - 對 **train 的 `gold_quotes`** 做同樣統計，比較模型輸出分佈 vs 真實分佈差距。
    - 繪製圓餅圖 + 長條圖存成 `.png`，數據表存 `outputs/phase_5/modality_analysis.json`。
    - 在報告中分析偏好成因（embedding modality gap、`img_description` 品質、resample 策略等）。

[ ] 任務 5：報告 PDF 撰寫。把任務 1~4 素材整合成 `HW3_<studentID>.pdf`，每題給足細節。

[ ] 任務 6：E3 打包繳交（沿用 plan 中要求，併入本 phase 末尾）。
    - 程式碼整理為 `HW3_<studentID>.py` 或 `HW3_<studentID>.ipynb`（可重現 Phase 1 ~ Phase 4 流程）。
    - 連同 `HW3_<studentID>.pdf` 一起壓成 `HW3_<studentID>.zip`。
    - deadline：**2026-05-26 23:59**（無遲交）。
    - Kaggle 端確認 team name 已改為 `<student ID>`（避免 -5%），並選定 3 個 submission 作為 private LB 計分。