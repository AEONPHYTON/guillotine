"""
config.py

Central configuration for the Guillotine batch runner.
Paths and parameters can be overridden via environment variables (.env) -
see .env.example. No credentials in here: local Ollama does not need any.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Project paths -------------------------------------------------------
DATA_DIR = Path(os.getenv("GHIGLIOTTINA_DATA_DIR", r"D:\guillotine\guillotine\source"))

PROMPTS_JSON = DATA_DIR / os.getenv("PROMPTS_JSON_NAME", "guillotine_prompts.json")
RESULTS_XLSX = DATA_DIR / os.getenv("RESULTS_XLSX_NAME", "ghigliottina_results.xlsx")
LOG_FILE = DATA_DIR / "batch_runner.log"

# --- Ollama -------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# 120s timeout for non-reasoning models
# for reasoning models, set to 600s. If it hasn't finished in 10 minutes, the task is a hard one
OLLAMA_GENERATE_TIMEOUT_S = int(os.getenv("OLLAMA_GENERATE_TIMEOUT_S", "600"))
OLLAMA_LOAD_TIMEOUT_S = int(os.getenv("OLLAMA_LOAD_TIMEOUT_S", "300"))
OLLAMA_KEEP_ALIVE_DURING_MODEL = os.getenv("OLLAMA_KEEP_ALIVE", "5m")

# --- Sampling -------------------------------------------------------------
TEMPERATURES = [0.0, 0.3, 0.7]
TOP_K = 40
TOP_P = 0.9

SHEET_BY_TEMPERATURE = {
    0.0: "T_0.0",
    0.3: "T_0.3",
    0.7: "T_0.7",
}

# --- Local models -------------------------------------------------------
# 17 original + 5 added after a VRAM check (16GB, margin under 13.5GB).
# Tags verified against the actual output of "ollama list" on 2026-08-04.
# mistral-small:22b, qwen2.5:14b, starcoder2:15b, granite3-dense:8b had not
# been pulled yet as of that date - if you haven't pulled them yet, they
# stay in the list and are skipped with a warning (they don't block the
# batch), see resolve_available_models in batch_runner.py.
LOCAL_MODELS = [
    "nemotron-3-nano:4b",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "tinyllama:1.1b",
    "phi3:mini",
    "phi3.5:latest",
    "nous-hermes2:latest",
    "zephyr:7b",
    "mistral:7b",
    "vicuna:13b-q4_K_M",
    "qwen2.5:7b",
    "qwen3.5:9b",
    "llama3.2:3b",
    "llama3.1:8b",
    "gemma2:2b",
    "gemma2:9b",
    "gemma4:e4b",
    "mistral-small:22b",
    "qwen2.5:14b",
    "deepseek-r1:14b",
    "starcoder2:15b", # doesn't pick up info from the prompt
    "granite3-dense:8b",
    "gpt-oss:20b",       # 13GB - right at the threshold, verify empirically before the full batch
    "gemma3:12b",        # 8.1GB
    "gemma4:12b",        # 7.6GB
    "qwen3:14b",         # 9.3GB
    "Hudson/gpt2-instruct:345m-q8_0",
    "stablelm2:1.6b-chat",
    "orca-mini:3b",
]

# Used to flag the is_reasoning_model column in logs and to warn before a
# known-slow model. Base: known architecture name (deepseek-r1, qwq,
# gpt-oss). Added after analyzing the real timings of Session 1
# (2026-08-05/06): qwen3.5:9b (stdev 30.2s, 18 timeouts out of 89),
# gemma4:12b (stdev 25.6s, 25 timeouts out of 89) and qwen3:14b (stdev
# 7.4s, 0 timeouts but variance clearly outside the flat ~2.3s+-0.1s
# pattern of non-reasoning models) empirically behave like reasoning
# models even though their tag name doesn't label them as such.
# gemma4:12b is an exact match (not "gemma4"): gemma4:e4b instead has
# flat variance (stdev 3.0s) and should not be included.
REASONING_MODEL_MARKERS = ["deepseek-r1",
                           "qwq",
                           "gpt-oss",
                           "qwen3.5",
                           "gemma4:12b",
                           "qwen3:14b",
                           "starcoder2:15b-instruct"]

# --- Prompt template -------------------------------------------------------
PROMPT_TEMPLATE = (
    "You are playing a word-association game. You will be given 5 clue "
    "words. Find the single word that connects all five (through common "
    "phrases, idioms, or shared concepts). Respond with ONLY the final "
    "word, in uppercase, with no explanation. "
    "Clues: {word1}, {word2}, {word3}, {word4}, {word5}."
)

# --- Scoring -------------------------------------------------------------
LEVENSHTEIN_SIMILARITY_THRESHOLD = 0.5

# --- Output columns (order = column order in the Excel sheet) -------------
RESULT_COLUMNS = [
    "model_name", "prompt_id", "word1", "word2", "word3", "word4", "word5",
    "expected_solution", "raw_response", "predicted_word", "format_compliant",
    "exact_match", "same_root", "levenshtein_normalized", "is_similar_lev",
    "is_clue_echo", "synonym_manual", "coherence_manual", "timestamp",
    "difficulty_note", "duration_s", "is_first_call", "status",
]

# --- OpenRouter -------------------------------------------------------------
# No hardcoded credentials: the key lives only in .env, never in this file
# or anywhere else in the repository.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_TIMEOUT_S = int(os.getenv("OPENROUTER_TIMEOUT_S", "600"))
# 20 requests/minute = 1 every 3s on the free models - 3.5s margin for
# jitter. See the rate limit hit with z-ai/glm-5.2 (2026-08-20).
OPENROUTER_MIN_INTERVAL_S = float(os.getenv("OPENROUTER_MIN_INTERVAL_S", "3.5"))
# Optional, only for attribution in OpenRouter's public leaderboard - not
# required for the code to work.
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "ghigliottina-benchmark")

# Populated with the exact IDs as they are verified on the OpenRouter site
# (model page -> API snippet -> "model" field). A wrong ID fails with a
# clear error, not silently, see resolve_available_models in
# openrouter_runner.py.
OPENROUTER_MODELS: list[str] = [
        # free
        "nvidia/nemotron-nano-9b-v2:free",
        "z-ai/glm-5.2:free",
        "nvidia/nemotron-3.5-lightning:free",
        "openai/gpt-oss-20b:free",
        "google/gemma-4-26b-a4b-it:free",

        # payment
        "inclusionai/ling-2.6-flash",
        "inclusionai/ling-3.0-flash",
        "ibm-granite/granite-4.0-h-micro",
        "nex-agi/nex-n2-mini",
        "upstage/solar-pro4",
        "qwen/qwen3.7-flash",
        "openai/gpt-oss-120b",
        "sao10k/l3-lunaris-8b",
        "meta-llama/llama-3.2-1b-instruct",
        "amazon/nova-micro-v1",
        "cohere/command-r7b-12-2024",
        "mistralai/mistral-small-24b-instruct-2501",
        "ibm-granite/granite-4.1-8b",
        "qwen/qwen3-30b-a3b-instruct-2507",
        "deepseek/deepseek-v4-flash-0731",
        "z-ai/glm-4.7-flash",
        "bytedance-seed/seed-1.6-flash",
        "qwen/qwen3-32b",
        "stepfun/step-3.5-flash",
        "meta-llama/llama-4-scout",
        "meta-llama/llama-3.3-70b-instruct",
        "inclusionai/ring-2.6-1t",
        "inclusionai/ling-2.6-1t",
        "xiaomi/mimo-v2.5",
        "nousresearch/hermes-4-70b",
        "tencent/hy3",
        "tencent/hunyuan-a13b-instruct",
        "allenai/olmo-3-32b-think",
        "upstage/solar-pro-3",
        "cohere/command-r-08-2024",
        "openai/gpt-4o-mini",
        "z-ai/glm-4.5-air",
        "deepseek/deepseek-v3.2",
        "mistralai/mistral-saba",
        "meta-llama/llama-4-maverick",
        "qwen/qwen3-next-80b-a3b-thinking",
        "arcee-ai/trinity-large-thinking",
        "stepfun/step-3.7-flash",
        "nex-agi/nex-n2-pro",
        "xiaomi/mimo-v2.5-pro",
        "z-ai/glm-4.6v",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen3.7-plus",
        "amazon/nova-2-lite-v1",
        "z-ai/glm-4.7",
        "mistralai/mistral-medium-3.1",
        "openai/gpt-3.5-turbo",
        "moonshotai/kimi-k2.5",
        "deepseek/deepseek-r1-0528",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "bytedance-seed/seed-2-1-turbo",
        "z-ai/glm-5",
        "z-ai/glm-5.2",
        "aion-labs/aion-3.0-mini",
        "arcee-ai/virtuoso-large",
        "nousresearch/hermes-3-llama-3.1-405b",
        "qwen/qwen3-235b-a22b-2507",
        "qwen/qwen3.5-122b-a10b",
        "qwen/qwen3-235b-a22b-thinking-2507",
        "cohere/command-r-plus-08-2024",
        "mistralai/mixtral-8x22b-instruct", # partial issues at temp 0
        "deepseek/deepseek-v4-flash-vision-exp", # issues... vision model
        "thinkingmachines/inkling-small"
]

# Raw backup of the paid calls (JSONL, one file per session, not one file
# per model/call). See openrouter_client.py.
JSON_BACKUP_DIR = DATA_DIR / "raw_logs"

# True "commercial" models (high cost per call, a single run with no
# control over sampling parameters - same principle as answers collected
# by hand via copy-paste in chat, RQ4 does not apply to these). Kept
# separate from OPENROUTER_MODELS on purpose: that list stays at
# controlled temperature (3 runs), this one gets a single run for cost
# reasons, not because it's technically impossible to control the
# parameters.

OPENROUTER_COMMERCIAL: list[str] = [
    "anthropic/claude-opus-5",
    "x-ai/grok-4.6",
    "openai/gpt-5.6-sol",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3",
    "qwen/qwen3.8-max",
    "meta/muse-spark-1.2",
    "openai/gpt-5.5",
    "google/gemini-3.7-flash",
    "deepseek/deepseek-v4-pro-0813",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "cohere/command-a",
    "inception/mercury-2",
    "anthropic/claude-sonnet-5",
]

COMMERCIAL_SHEET_NAME = "Commercial"
