這份作業（INLP HW3）的核心是**多模態檢索（Multimodal Retrieval）**，評估指標為 **Recall@5**。由於不要求生成答案，純粹是一個 Information Retrieval (IR) 與 Ranking 的任務。

回顧近年的 Kaggle 競賽（例如 *Multimodal Document Retrieval Challenge*、*Kaggle LLM Science Exam* 等），要在這類 RAG 檢索賽道中拿下第一名（1st Place），單靠傳統的 Dense Retriever（如直接把文本轉 Embedding 算 Cosine Similarity）是絕對不夠的。Kaggle 頂尖選手的獲勝關鍵通常在於**多階段檢索（Multi-stage Retrieval）**、**強大的特徵組合（Ensemble）**以及**資料預處理的魔法**。

考量到作業限制（**參數量 $\le$ 80B 的開源模型**，禁用 GPT-4/Claude API），以下我為你整理出最有可能在 Leaderboard 霸榜的第一名解法策略：

---

### 🏆 奪冠架構：四階段多模態檢索系統 (The 1st-Place Pipeline)

要達到最高的 Recall@5，你需要建立一個「召回率極高（寬鬆）」到「精準度極高（嚴格）」的漏斗型架構。

#### 第一階段：Query 增強與重寫 (Query Expansion & Rewriting)
很多時候使用者的問題很簡短，直接算 Embedding 容易找不到對應的隱含文本。
* **HyDE (Hypothetical Document Embeddings)：** 這是 Kaggle RAG 賽道的常勝軍技巧。使用頂尖的開源 70B 模型（如 `Llama-3.3-70B-Instruct` 或 `Qwen2.5-72B-Instruct`，符合 $\le$ 80B 規定），讓它「假裝」回答這個問題，生成一段虛擬答案。
* **作法：** 將原始 Query 與生成的虛擬答案拼接在一起（`Query + HyDE Answer`）再去做 Embedding，這樣能大幅增加與正確 Evidence 的語意重疊率。
* **意圖辨識 (Intent Detection)：** 讓 LLM 判斷使用者的問題是針對「文字」、「表格（Table）」還是「圖片（Figure）」。若判斷出是問表格，後續 Rerank 時可以直接對圖片證據（`img_quotes`）加上權重加分。

#### 第二階段：多模態特徵工程（勝負的關鍵分水嶺）
作業提供了 `img_description`，但官方給的描述通常偏向字面、不夠深入。這也是你的報告 Q3 要你討論的重點。
* **降維打擊（VLM 重新描述）：** 既然允許用開源模型，可以直接拿目前地表最強的開源視覺語言模型（例如 **`Qwen2-VL-72B-Instruct`** 或 **`InternVL-2.5-78B`**）對所有 `image/` 目錄下的圖片重新生成 Caption。
    * *Prompt 技巧：* 告訴 VLM 這是為了 RAG 檢索，請它詳細提取圖片中的「數據」、「圖表趨勢」、「實體文字（OCR）」等資訊。將這個超強大的 Caption 作為該圖片的文本特徵，會比原版的描述準確非常多。
* **前沿技術 (Vision Retriever - ColPali)：** 傳統作法是把圖片轉文字，但 2024 年底最新的 SOTA 技術是 **ColPali**（基於 PaliGemma 的多模態檢索）。它可以直接將文件圖片映射到與 Query 相同的向量空間中，捕捉文字描述容易遺漏的視覺排版資訊。如果能結合視覺檢索的分數，肯定能與其他對手拉開巨大差距。

#### 第三階段：雙路混合召回 (Hybrid Retrieval)
Kaggle 冠軍隊伍從不依賴單一檢索器，他們會用 **Sparse + Dense** 雙路召回，取聯集（抓取 Top-50 到 Top-100）。
1.  **Sparse Retriever (BM25)：** 對於長尾詞、專有名詞（例如特定的型號、人名、年份）非常有效。
2.  **Dense Retriever：** 推薦使用目前霸榜的開源 Embedding 模型，如 **`BGE-M3`**（支援多語言、長文本、稀疏與稠密向量）或 **`E5-Mistral-71B`**。
3.  **分數融合 (Reciprocal Rank Fusion, RRF)：** 將 BM25 的排名與 Dense 模型的排名透過數學公式結合：
    $RRF\_Score = \frac{1}{k + Rank_{BM25}} + \frac{1}{k + Rank_{Dense}}$
    （通常常數 $k$ 設為 60）。根據這套公式，選出綜合分數最高的前 50 名 Candidate。

#### 第四階段：Cross-Encoder 與 LLM 終極重排 (Reranking)
作業要求最終選出 Top-5，因此從召回的 Top-50 中挑出最精華的 5 個是決定分數的最後一步。
* **第一層 Rerank (Cross-Encoder)：** 使用如 `bge-reranker-v2.5-gemma2-lightweight` 等開源重排模型。這類模型會把 `[Query, Candidate]` 一起輸入神經網路，計算出極為精準的相關性分數。
* **第二層 Rerank (LLM-as-a-Judge) [可選/殺手鐧]：** 若想榨乾分數，可以對 Cross-Encoder 排出來的前 10 名，再呼叫一次 70B 的 LLM，Prompt 寫道：「問題是 X，請評估以下候選文檔 Y 是否能直接作為證據回答問題，請給出 0.0 到 1.0 的相關性分數。」根據這個終極分數做最後的 Top-5 排序。

---

### 💡 Kaggle 競賽專屬的提分小技巧 (Hacks)

除了強大的演算法架構，Kaggle 選手還會用這些細節 Trick 偷分數：
1.  **Metadata 過濾器 (Hard Filtering)：**
    觀察資料中的 `doc_name` 與 `domain`。如果能用 NER (命名實體辨識) 或字串匹配找出 Query 是在問哪一篇特定的 Document，就把其他 Document 的候選者直接過濾掉（分數歸零），這能大幅減少模型被其他文章的相似文本誤導的機率。
2.  **Modality 權重微調：**
    正如作業報告 Q4 所暗示的，資料集可能有 Modality Preference。你可以根據 `dev_15.jsonl` 的真實標籤（`gold_quotes`）統計出真實的比例。如果在 Validation 時發現你的模型過度偏好文字證據，可以在最終綜合分數對圖片證據（Image Candidate）乘上一個係數（例如 1.1x）來做校準。
3.  **Test-Time Augmentation (TTA)：**
    將使用者的 Query 用 LLM 換個說法（Paraphrase）變成 3 個不同的 Query，分別進行檢索，最後將這 3 次檢索的候選項目分數加總。這能提高對不同問法的魯棒性。

### 🚀 實作路徑建議

若要以最高效率取得第一名，建議你的開發順序如下：
1. **Baseline（求穩）：** 先用現成的 `img_description`，做出 **BM25 + BGE-M3 的雙路召回與 RRF 融合**。這個基準線通常已經能輕易打敗助教的 Strong Baseline。
2. **衝刺 Leaderboard（奪冠）：** 引入 **VLM 對圖片重新生成 Caption**（使用 Qwen2-VL 替換掉原本粗糙的描述），並在檢索後加上 **BGE-Reranker** 進行最後排序。
3. **報告拿滿分：** 實作這套系統後，你在回答作業 Q2（四種方法比較）與 Q3（多模態檢索策略比較）時，手邊將會擁有完整的對照組數據與深入的洞察，這絕對能讓你的 Report 拿到滿分。
