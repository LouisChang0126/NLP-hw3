"""
INLP HW3 Multimodal RAG - 全域設定檔
所有路徑、模型選擇、超參數集中管理
"""
import os
from datetime import datetime

# ── 亂數種子 ──────────────────────────────────────────
SEED = 42

# ── 路徑 ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images", "images")  # 實際圖片位置
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, "sample_submission.csv")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ── Dev split ─────────────────────────────────────────
DEV_SIZE = 200  # hold-out dev set 筆數

# ── Phase 1: BM25 ────────────────────────────────────
BM25_TOP_K = 5
BM25_TOP_K_LARGE = 100  # 保留給後續 RRF / Reranker

# ── Phase 1.5: Direct LLM Selection ──────────────────
PHASE_DIRECT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── Phase 2: Dense Retrieval ─────────────────────────
PHASE_2_EMBED_MODEL = "BAAI/bge-m3"
DENSE_TOP_K = 5
DENSE_TOP_K_LARGE = 100

# ── Phase 2b: Multimodal Embedding ───────────────────
PHASE_2B_MM_MODEL = "google/siglip-so400m-patch14-384"

# ── Phase 3: Hybrid (RRF) ────────────────────────────
RRF_K = 60  # RRF 平滑常數

# ── Phase 4: Query Expansion + Cross-Encoder Reranking ─
PHASE_4_EXPANSION_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # HyDE 用
# Reranker 模型清單 (按顯存需求由低到高)
PHASE_4_RERANKER_MODELS = [
    {"id": "BAAI/bge-reranker-v2-m3", "params": "568M", "vram": "~2GB"},
    {"id": "BAAI/bge-reranker-base", "params": "278M", "vram": "~1GB"},
    # {"id": "Qwen3-Reranker-8B", "params": "8B", "vram": "~16GB"},
]
PHASE_4_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 預設
RERANKER_TOP_K = 5

# ── 提交格式 ──────────────────────────────────────────
SUBMISSION_COLUMNS = ["q_id", "gold_quotes"]
TOP_K_SUBMIT = 5


def get_output_dir(phase: str, model_name: str = "") -> str:
    """產生帶時間戳記的輸出目錄"""
    timestamp = datetime.now().strftime("%m%d%H%M")
    if model_name:
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        dirname = f"{safe_name}_{timestamp}"
    else:
        dirname = timestamp
    path = os.path.join(OUTPUT_DIR, phase, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def print_config():
    """印出當前設定快照"""
    import inspect
    print("=" * 60)
    print("CONFIG SNAPSHOT")
    print("=" * 60)
    module = __import__(__name__)
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        val = getattr(module, name)
        if not callable(val):
            print(f"  {name} = {val!r}")
    print("=" * 60)
