"""
INLP HW3 Multimodal RAG — 全域設定檔

精簡版: 只保留當前 active pipeline (Phase 6 純 LLM 15→5) 所需設定
        以及報告 Q2/Q3 對照組 (BM25, Dense, Multimodal embedding) 用的 model id.
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

# VLM caption 副產物 (Q3 報告用)
IMG_CAPTIONS_VLM_FILE = os.path.join(DATA_DIR, "img_captions_vlm.json")


# ══════════════════════════════════════════════════════
#  Dev split
# ══════════════════════════════════════════════════════
DEV_SIZE = 200


# ══════════════════════════════════════════════════════
#  共用參數
# ══════════════════════════════════════════════════════
TOP_K_SUBMIT = 5
CANDIDATE_POOL_DEPTH = 100


# ══════════════════════════════════════════════════════
#  Phase 1: BM25 (Q2 報告對照)
# ══════════════════════════════════════════════════════
BM25_TOP_K = TOP_K_SUBMIT
BM25_TOP_K_LARGE = CANDIDATE_POOL_DEPTH


# ══════════════════════════════════════════════════════
#  Phase 2: Dense retrieval (Q2 報告對照)
# ══════════════════════════════════════════════════════
PHASE_2_EMBED_MODEL = "BAAI/bge-m3"
DENSE_TOP_K = TOP_K_SUBMIT
DENSE_TOP_K_LARGE = CANDIDATE_POOL_DEPTH


# ══════════════════════════════════════════════════════
#  Phase 2b: Multimodal embedding (Q3 報告 a vs b 對照)
# ══════════════════════════════════════════════════════
PHASE_2B_MM_MODEL = "google/siglip-so400m-patch14-384"


# ══════════════════════════════════════════════════════
#  Phase 6 (active LB-best pipeline): 純 LLM 直接 15→5
#  → Dev 0.8988  LB 0.81895 (Qwen3.6-27B Q6_K GGUF, CL=1500)
# ══════════════════════════════════════════════════════

# NIM API (Gemma-4-31B 用; Q2 報告 direct LLM selection)
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_API_KEY = os.environ.get(
    "NIM_API_KEY",
    "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",
)
DIRECT_LLM_NIM_MODEL = "google/gemma-4-31b-it"

# 本地 GGUF (Phase 6 local 主要 pipeline)
LOCAL_LLM_REPO = "unsloth/Qwen3.6-27B-GGUF"
LOCAL_LLM_GGUF = "Qwen3.6-27B-Q6_K.gguf"


# ══════════════════════════════════════════════════════
#  VLM image re-captioning (Q3 報告: VLM 重生成 caption 對照)
# ══════════════════════════════════════════════════════
VLM_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
VLM_MAX_TOKENS = 1024
VLM_TEMPERATURE = 0.2
VLM_ENABLE_THINKING = False
VLM_NUM_WORKERS = 6
VLM_MIN_INTERVAL_SEC = 0.5
VLM_SAVE_EVERY = 25
VLM_MAX_RETRIES = 4


# ══════════════════════════════════════════════════════
#  Submission
# ══════════════════════════════════════════════════════
SUBMISSION_COLUMNS = ["q_id", "gold_quotes"]


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
        if "API_KEY" in name and isinstance(val, str) and len(val) > 12:
            val = val[:8] + "..." + val[-4:]
        print(f"  {name} = {val!r}")
    print("=" * 60)
