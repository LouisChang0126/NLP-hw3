"""
INLP HW3 Multimodal RAG — 全域設定檔

集中管理所有路徑、模型、超參數。預設值都來自實驗驗證:
  - Phase 3b/4b 的 LB-best config (current ceiling 0.72650)
  - VLM caption 的最佳呼叫設定 (enable_thinking=False 等)
"""
import os
from datetime import datetime

# ══════════════════════════════════════════════════════
#  路徑
# ══════════════════════════════════════════════════════
SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images", "images")
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, "sample_submission.csv")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Phase 中產生的副產物
IMG_CAPTIONS_VLM_FILE = os.path.join(DATA_DIR, "img_captions_vlm.json")
STRATIFIED_DEV_IDS_FILE = os.path.join(SPLITS_DIR, "stratified_dev_ids.json")


# ══════════════════════════════════════════════════════
#  Dev split
# ══════════════════════════════════════════════════════
DEV_SIZE = 200                  # NOTE: 200 太小, stratified 後每 domain ~67, variance 大
DEV_SPLIT_STRATEGY = "random"   # "random" | "stratified"


# ══════════════════════════════════════════════════════
#  共用檢索參數
# ══════════════════════════════════════════════════════
TOP_K_SUBMIT = 5
CANDIDATE_POOL_DEPTH = 100      # 每階段保留的 Top-N 候選池深度
RRF_K = 60                      # Reciprocal Rank Fusion 平滑常數


# ══════════════════════════════════════════════════════
#  Phase 2: Dense Retrieval
# ══════════════════════════════════════════════════════
PHASE_2_EMBED_MODEL = "BAAI/bge-m3"


# ══════════════════════════════════════════════════════
#  Phase 3b: 4-way RRF (BM25 + BGE-M3 dense/sparse/colbert) + image boost
#  → Dev 0.8081  LB 0.72342  (P2 多向量證實有用)
# ══════════════════════════════════════════════════════
PHASE_3B_USE_BM25 = True
PHASE_3B_USE_DENSE = True
PHASE_3B_USE_SPARSE = True     # +1.15 pp on strat dev
PHASE_3B_USE_COLBERT = True    # strat dev 看似 -0.92 pp, 但 LB 證實 +1.16 pp (Phase 3d 驗證)
PHASE_3B_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
PHASE_3B_IMAGE_BOOST = 0.004   # gold image:text=60:40, 此值最接近


# ══════════════════════════════════════════════════════
#  Phase 4b: bge-reranker-v2-gemma + score fusion with Phase 3b RRF
#  → Dev 0.8068  LB 0.72650 (current best)
#  注意: 用 transformers 直接調用, FlagEmbedding 與 transformers 5.5 不相容
# ══════════════════════════════════════════════════════
PHASE_4B_RERANKER_MODEL = "BAAI/bge-reranker-v2-gemma"
PHASE_4B_RERANK_MAX_LEN = 512
PHASE_4B_RERANK_BATCH = 16
PHASE_4B_FUSION_ALPHA = 2.0    # final = z(rerank) + alpha * z(rrf_prior)
PHASE_4B_IMAGE_BOOST_Z = 0.30  # 套在 z-scored fused score 上


# ══════════════════════════════════════════════════════
#  VLM image re-captioning (NVIDIA NIM)
# ══════════════════════════════════════════════════════
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
# API key 可從環境變數覆寫; 預設留空, 請執行前 export
NIM_API_KEY = os.environ.get(
    "NIM_API_KEY",
    "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",
)
VLM_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
VLM_MAX_TOKENS = 1024
VLM_TEMPERATURE = 0.2
VLM_ENABLE_THINKING = False    # 關閉節省 ~8x output tokens (實驗證實)
VLM_NUM_WORKERS = 6
VLM_MIN_INTERVAL_SEC = 0.5
VLM_SAVE_EVERY = 25
VLM_MAX_RETRIES = 4


# ══════════════════════════════════════════════════════
#  Submission
# ══════════════════════════════════════════════════════
SUBMISSION_COLUMNS = ["q_id", "gold_quotes"]


# ══════════════════════════════════════════════════════
#  Legacy / 報告用 (未在當前主 pipeline 使用)
# ══════════════════════════════════════════════════════
# Phase 1.5 - Direct LLM selection (Q2 報告四方法之一)
PHASE_DIRECT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Phase 2b - Multimodal embedding (Q3 報告 a vs b 對照)
PHASE_2B_MM_MODEL = "google/siglip-so400m-patch14-384"

# Phase 4 (original) - HyDE 已實驗證實退步 (Dev -13 pp), 保留但不建議使用
PHASE_4_EXPANSION_MODEL = None  # 設 None 代表跳過 HyDE
PHASE_4_RERANKER_MODEL = PHASE_4B_RERANKER_MODEL  # 後續腳本仍可能引用


# ══════════════════════════════════════════════════════
#  向後相容 — 舊腳本可能用到的別名
# ══════════════════════════════════════════════════════
BM25_TOP_K = TOP_K_SUBMIT
BM25_TOP_K_LARGE = CANDIDATE_POOL_DEPTH
DENSE_TOP_K = TOP_K_SUBMIT
DENSE_TOP_K_LARGE = CANDIDATE_POOL_DEPTH
RERANKER_TOP_K = TOP_K_SUBMIT


# ══════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════
def get_output_dir(phase: str, model_name: str = "") -> str:
    """產生帶時間戳記的輸出目錄"""
    timestamp = datetime.now().strftime("%m%d%H%M")
    if model_name:
        safe = model_name.replace("/", "_").replace("\\", "_")
        dirname = f"{safe}_{timestamp}"
    else:
        dirname = timestamp
    path = os.path.join(OUTPUT_DIR, phase, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def print_config():
    """印出當前設定快照"""
    import types
    print("=" * 60)
    print("CONFIG SNAPSHOT")
    print("=" * 60)
    module = __import__(__name__)
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        val = getattr(module, name)
        if callable(val) or isinstance(val, types.ModuleType):
            continue
        # API key 不印明文
        if "API_KEY" in name and isinstance(val, str) and len(val) > 12:
            val = val[:8] + "..." + val[-4:]
        print(f"  {name} = {val!r}")
    print("=" * 60)
