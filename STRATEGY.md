# Nepali Tokenizer Benchmark — Strategy Brief

## Goal
Comprehensive benchmark of ALL major 2026 LLM tokenizers on Nepali (Devanagari) text. Publishable results for a blog post. Then: train one SentencePiece tokenizer + extend the worst models.

## Benchmark Design

### Metric: "Nepali Tax"
- `tax = (nepali_tokens / nepali_words) / (english_tokens / english_words)`
- Tax of 3.0x = Nepali costs 3x more tokens per word than English
- Directly maps to: API cost multiplier, context window reduction, latency increase

### Test Corpus
- **Nepali:** 200 docs from CC-100 (data/raw/ne.txt.xz), filtered to >200 chars
- **English:** Synthetic paragraph repeated to match word count
- ~10K-20K words each side for statistical stability

### Concerns with current approach
1. **English text is synthetic** — a single paragraph repeated 20x. Not representative of real English. Should we use real English text (e.g., from CC-100 English or similar)?
2. **200 docs may not be enough** — should we test on 1000+ docs for tighter confidence intervals?
3. **"Words" metric is fuzzy** — Nepali doesn't always space-delimit the same way English does. `split()` is a rough proxy. Is this good enough or do we need something more principled?
4. **Should we also report:** tokens-per-character ratio? Fertility (tokens per Unicode codepoint)? Compression ratio vs UTF-8 bytes?
5. **Missing analysis:** What specific Nepali/Devanagari tokens exist in each vocab? Coverage of common Nepali syllables?

### Models to Test (14 + tiktoken)

**2026 models:**
| Local Dir | Model ID | Vocab |
|---|---|---|
| qwen3.5 | Qwen/Qwen3.5-27B | 248K |
| gemma4 | google/gemma-4-E2B-it | 262K |
| mistral-small-4 | mistralai/Mistral-Small-4-119B-2603 | 131K |
| deepseek-v4 | deepseek-ai/DeepSeek-V4-Flash | 128K |
| phi4 | microsoft/phi-4 | 100K |
| glm4 | THUDM/glm-4-9b | 151K |
| kimi-k2 | moonshotai/Kimi-K2-Instruct | 164K |
| llama4 | meta-llama/Llama-4-Scout-17B-16E (pending) | 202K |

**Older baselines:**
| Local Dir | Model ID | Vocab |
|---|---|---|
| llama3 | meta-llama/Llama-3.2-1B | 128K |
| qwen2.5 | Qwen/Qwen2.5-0.5B | 151K |
| gemma2 | google/gemma-2-2b | 256K |
| mistral-v0.3 | mistralai/Mistral-7B-v0.3 | 32K |
| deepseek-v2 | deepseek-ai/DeepSeek-V2-Lite | 100K |
| phi3 | microsoft/phi-3-mini-4k-instruct | 32K |

**Proprietary:**
| Name | Method | Vocab |
|---|---|---|
| GPT-4o | tiktoken o200k_base | 200K |

### Output Format
- Results table sorted by tax (worst first)
- Group by model family to show old→new improvement
- JSON export for programmatic use
- Per-family Devanagari token count in vocab

## Questions for Review

1. Is the "tax" metric the right primary metric, or should we lead with something else?
2. Is the synthetic English comparison text a problem? If so, what's a good real English source that's unambiguously available?
3. Are there other 2026 models we're missing? (Command R+, Falcon, Yi, etc.)
4. For the blog post: what framing resonates — "cost fairness", "infrastructure gap", "multilingual readiness"?
5. Any methodological red flags in how the existing script works?

## Post-Benchmark: Tokenizer Training Plan

After benchmark establishes which models are worst:
1. Train SentencePiece BPE (32K-64K) on: CulturaX ne, ai4bharat/sangraha, CC-100, books corpus
2. Extend 3-4 worst model tokenizers with learned Nepali merges
3. Publish extended tokenizers on HuggingFace
