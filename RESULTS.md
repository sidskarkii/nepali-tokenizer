# Benchmark Results

## Metric

```
nepali_tax = nepali_tokens_per_word / english_tokens_per_word
```

A 3.0x tax means Nepali costs 3x more tokens than English for the same content. That's 3x the API bill and 1/3 the context window.

Tokenizers called with `add_special_tokens=False`. Tiktoken (GPT-4o) doesn't add them by default.

## Corpus

**Nepali** — 2,054 docs, ~2.78M words
- CC-100 web text: 2,000 docs (~445K words). Filtered to >200 chars, >50% Devanagari. NFC normalized, SHA-256 deduplicated.
- Books: 54 docs (~2.33M words). NRB publications, Vedas, Muna Madan, dictionaries. Filtered to >500 chars, >50% Devanagari. Legal capped at 30 (seed=42).

**English** — CC-100: 2,000 docs (~107K words). Filtered to >200 chars, >80% Latin.

Books are much larger than CC-100 docs (avg 43K vs 222 words). Tax is corpus-weighted (total tokens / total words), so books dominate. This is intentional.

## Results (worst → best)

```
Model                Gen      Vocab │ NE tok/word EN tok/word │    Tax │ Dev Tokens  Dev%
====================================================================================================
Phi-4                2025   100,352 │        7.17        1.25 │   5.7x │         27  0.0%
GLM-4                2024   151,343 │        6.57        1.24 │   5.3x │         27  0.0%
Mistral v0.3         2024    32,768 │        6.79        1.35 │   5.0x │         44  0.1%
Phi-3.5              2024    32,011 │        7.05        1.40 │   5.0x │         39  0.1%
Qwen 3               2025   151,669 │        6.10        1.25 │   4.9x │         71  0.1%
GLM-5                2026   154,856 │        5.89        1.24 │   4.8x │        138  0.1%
Qwen 3.5             2026   248,077 │        4.65        1.25 │   3.7x │        959  0.4%
DeepSeek V4          2026   129,280 │        4.10        1.24 │   3.3x │        303  0.2%
Kimi K2.6            2026   163,840 │        3.99        1.23 │   3.3x │        318  0.2%
Kimi K2              2025   163,840 │        3.99        1.23 │   3.3x │        318  0.2%
LLaMA 3              2024   128,256 │        3.72        1.24 │   3.0x │      1,018  0.8%
Mistral Small 4      2026   131,072 │        3.30        1.27 │   2.6x │      1,569  1.2%
Gemma 2              2024   256,000 │        3.21        1.25 │   2.6x │      1,516  0.6%
LLaMA 4              2025   201,135 │        2.99        1.23 │   2.4x │      2,696  1.3%
GPT-4o               2024   200,019 │        2.68        1.22 │   2.2x │      3,985  2.0%
Gemma 4              2026   262,144 │        2.52        1.26 │   2.0x │     13,754  5.2%
DeepSeek V3*         2025   128,815 │  DROPS TEXT        1.41 │    N/A │          0  0.0%
```

## Family Progression

```
DeepSeek: V3 (2025): drops text via AutoTokenizer* → V4 (2026): 3.3x
Gemma:    2 (2024): 2.6x → 4 (2026): 2.0x
GLM:      4 (2024): 5.3x → 5 (2026): 4.8x
Kimi:     K2 (2025): 3.3x → K2.6 (2026): 3.3x  (same tokenizer)
LLaMA:    3 (2024): 3.0x → 4 (2025): 2.4x
Mistral:  v0.3 (2024): 5.0x → Small 4 (2026): 2.6x
Phi:      3.5 (2024): 5.0x → 4 (2025): 5.7x  (regression — better English, same bad Nepali)
Qwen:     3 (2025): 4.9x → 3.5 (2026): 3.7x
```

## Findings

**Phi-4 is worst and regressed.** 27 Devanagari tokens in 100K vocab. Microsoft expanded from 32K to 100K without adding Devanagari. The tax went up because English got more efficient while Nepali stayed the same.

**DeepSeek V3 tokenizer drops non-Latin text via `AutoTokenizer`.** When loaded with `transformers` v5, `LlamaTokenizerFast` silently overwrites the ByteLevel pre-tokenizer with Metaspace, causing all non-Latin scripts to encode to zero tokens. This is a `transformers` loading regression, not a DeepSeek vocab issue — loading with `PreTrainedTokenizerFast` produces correct output. Affects V3, R1, and related models. See [transformers #45488](https://github.com/huggingface/transformers/issues/45488).

**Gemma 4 is best** with 13,754 Devanagari tokens (5.2% of vocab). GPT-4o is second at 2.2x with 3,985 tokens.

**Devanagari token count predicts efficiency.** <50 tokens → ~5.3x tax. 50-300 → ~4.2x. 300-2000 → ~2.7x. 2000+ → ~2.1x.

**Kimi K2 and K2.6 share an identical tokenizer.** Same vocab, same token counts. K2.6 was a weights update only.

**Mistral improved most.** 5.0x → 2.6x, adding 1,525 Devanagari tokens.

## Tokenizer Extension Results

Added high-value Nepali tokens (pieces the base tokenizer splits into 3+ subtokens) from our 32K SentencePiece model:

| Target | Before | After | Reduction | Tokens Added |
|---|---:|---:|---:|---:|
| Phi-4 | 7.10 | 3.41 | 51.9% | 15,257 |
| Qwen 3.5 | 4.49 | 2.50 | 44.3% | 15,194 |
| DeepSeek V4 | 4.01 | 2.52 | 37.3% | 15,287 |
| Kimi K2.6 | 3.89 | 2.51 | 35.5% | 15,291 |

Extension only changes tokenization. The model still needs CPT to learn the new embeddings.

## Limitations

1. `str.split()` word boundaries differ across languages. Tokens-per-byte in the JSON avoids this but is less intuitive.
2. CC-100 sampling is sequential (first 2,000 qualifying docs), not random.
3. Books dominate the weighted average due to size skew (avg 43K vs 222 words/doc).
4. English CC-100 is streamed from HuggingFace — exact docs depend on dataset version. Use cached JSONL for reproducibility.
5. Some book texts have OCR noise from DjVu scans.

## Reproducing

```bash
pip install -r requirements.txt
python prep_data.py      # downloads CC-100 English, ~5 min
python benchmark.py      # runs benchmark, ~5 min
# results in results/benchmark_results.json
```

Full per-document metrics, vocabulary analysis, and bootstrap CIs are in the JSON output.

\* DeepSeek V3 text dropping is caused by a `transformers` v5 loading regression, not a DeepSeek tokenizer defect. `LlamaTokenizerFast` silently overwrites the tokenizer's ByteLevel pre-tokenizer with Metaspace on load. Loading with `PreTrainedTokenizerFast` produces correct results. See [transformers #45488](https://github.com/huggingface/transformers/issues/45488).
