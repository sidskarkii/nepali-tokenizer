"""Tokenizer registry, paths, and benchmark constants."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
RESULTS_DIR = ROOT / "results"
TOKENIZERS_DIR = ROOT / "tokenizers"

DEVANAGARI_RANGE = (0x0900, 0x097F)

CLEANING = {
    "min_doc_chars": 200,
    "min_devanagari_ratio": 0.5,
    "min_latin_ratio": 0.8,
    "books_min_chars": 500,
}

BENCHMARK = {
    "n_cc100_nepali_docs": 2000,
    "n_cc100_english_docs": 2000,
    "books_max_per_category": 30,
    "n_bootstrap": 10_000,
    "seed": 42,
}

# Top Nepali grapheme clusters for coverage analysis
NEPALI_GRAPHEMES = [
    # Vowels
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ",
    # Common consonants
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व", "श", "ष", "स", "ह",
    # Common conjuncts
    "क्ष", "त्र", "ज्ञ", "श्र",
    "क्र", "प्र", "ग्र", "द्र",
    "स्त", "स्थ", "न्त", "न्द",
    "म्ब", "ङ्ग", "ण्ड",
]

TOKENIZER_REGISTRY = [
    # Latest (2025-2026)
    {"name": "Qwen 3.5",        "local": "qwen3.5",        "family": "Qwen",     "gen": "2026", "model_id": "Qwen/Qwen3.5-27B"},
    {"name": "Gemma 4",         "local": "gemma4",          "family": "Gemma",    "gen": "2026", "model_id": "google/gemma-4-E2B-it"},
    {"name": "Mistral Small 4", "local": "mistral-small-4", "family": "Mistral",  "gen": "2026", "model_id": "mistralai/Mistral-Small-4-119B-2603"},
    {"name": "DeepSeek V4",     "local": "deepseek-v4",     "family": "DeepSeek", "gen": "2026", "model_id": "deepseek-ai/DeepSeek-V4-Flash"},
    {"name": "GLM-5",           "local": "glm5",            "family": "GLM",      "gen": "2026", "model_id": "zai-org/GLM-5.1"},
    {"name": "Kimi K2.6",       "local": "kimi-k2.6",       "family": "Kimi",     "gen": "2026", "model_id": "moonshotai/Kimi-K2.6"},
    {"name": "Phi-4",           "local": "phi4",            "family": "Phi",      "gen": "2025", "model_id": "microsoft/phi-4"},
    {"name": "LLaMA 4",         "local": "llama4",          "family": "LLaMA",    "gen": "2025", "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
    {"name": "LLaMA 3",         "local": "llama3",          "family": "LLaMA",    "gen": "2024", "model_id": "meta-llama/Llama-3.2-1B"},
    {"name": "GPT-4o",          "local": None,              "family": "OpenAI",   "gen": "2024", "model_id": "o200k_base"},
    # Previous gen (for family comparison)
    {"name": "Qwen 3",          "local": "qwen3",           "family": "Qwen",     "gen": "2025", "model_id": "Qwen/Qwen3-0.6B"},
    {"name": "Gemma 2",         "local": "gemma2",          "family": "Gemma",    "gen": "2024", "model_id": "google/gemma-2-2b"},
    {"name": "Mistral v0.3",    "local": "mistral-v0.3",    "family": "Mistral",  "gen": "2024", "model_id": "mistralai/Mistral-7B-v0.3"},
    {"name": "DeepSeek V3",     "local": "deepseek-v3",     "family": "DeepSeek", "gen": "2025", "model_id": "deepseek-ai/DeepSeek-V3"},
    {"name": "GLM-4",           "local": "glm4",            "family": "GLM",      "gen": "2024", "model_id": "THUDM/glm-4-9b"},
    {"name": "Kimi K2",         "local": "kimi-k2",         "family": "Kimi",     "gen": "2025", "model_id": "moonshotai/Kimi-K2-Instruct"},
    {"name": "Phi-3.5",         "local": "phi3.5",          "family": "Phi",      "gen": "2024", "model_id": "microsoft/Phi-3.5-mini-instruct"},
]
