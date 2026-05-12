# Devanagari Tokenizer Benchmark: Measuring and Reducing the Nepali Token Tax Across 17 LLM Families

Comprehensive benchmark of tokenizer efficiency for the Nepali language (Devanagari script) across 17 large language model tokenizers from 9 model families including GPT-4o, Gemma 4, LLaMA 4, Qwen 3, DeepSeek V4, Phi-4, Mistral, Kimi K2, and GLM. Measures the "token tax" that Nepali speakers pay compared to English: from 2.0x (Gemma 4) to 5.7x (Phi-4).

Includes a full remediation pipeline: extended tokenizers for 4 models (up to 52% token reduction), a standalone 32K Nepali BPE tokenizer trained on 7.49GB of cleaned corpus, and an end-to-end CPT+SFT experiment on Qwen3-4B achieving 48% token count reduction with BPC improvement from 1.96 to 1.07. Also documents a **DeepSeek V3 tokenizer bug** where non-Latin text silently drops to zero tokens.

**Write-up**: [siddhantskarki.com/case-studies/nepali-tokenizer](https://www.siddhantskarki.com/case-studies/nepali-tokenizer) | **Models**: [huggingface.co/sidskarki](https://huggingface.co/sidskarki)

## Results

Full 17-model benchmark in [RESULTS.md](RESULTS.md). Highlights:

| Tokenizer | Nepali tax | Nepali tok/word |
|---|---:|---:|
| Phi-4 (worst) | 5.7x | 7.17 |
| Qwen 3.5 | 3.7x | 4.65 |
| DeepSeek V3 | N/A | drops text |
| GPT-4o | 2.2x | 2.68 |
| Gemma 4 (best) | 2.0x | 2.52 |

After extension:

| Target | Before | After | Reduction |
|---|---:|---:|---:|
| Phi-4 | 7.10 | 3.41 | 51.9% |
| Qwen 3.5 | 4.49 | 2.50 | 44.3% |
| DeepSeek V4 | 4.01 | 2.52 | 37.3% |

The standalone 32K Nepali BPE tokenizer reaches 1.34 tok/word (near English-level), trained on a 7.49GB cleaned corpus from CulturaX, CC-100, sangraha, and Nepali books.

## Repository Layout

```
benchmark.py           Tokenizer benchmark (17 models, 2 strata)
config.py              Tokenizer registry and benchmark constants
prep_data.py           Benchmark data preparation
build_corpus.py        Tokenizer training corpus assembly
train_tokenizer.py     SentencePiece BPE training and sweep
extend_tokenizer.py    Add Nepali tokens to existing model tokenizers
cpt_train.py           Qwen3-4B continued pretraining with LoRA
sft_train.py           Nepali instruction fine-tuning
eval_model.py          Base vs tuned model comparison
notebooks/             Reproducible tokenizer benchmark notebook
results/               Benchmark and extension result JSON
tokenizers/            Selected tokenizer artifacts
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the tokenizer benchmark
python prep_data.py
python benchmark.py

# Or the notebook (uses only committed tokenizer files, no downloads)
jupyter notebook notebooks/nepali_tokenizer_benchmark.ipynb
```

## What's Where

**This repo** — benchmark code, training scripts, results, selected tokenizer artifacts.

**HuggingFace** — model adapters (CPT + SFT), extended tokenizers.

**Not committed** — the 7.49GB training corpus (recipe is in `build_corpus.py`), downloaded base tokenizer caches, model checkpoints.

## License

Community Use License v1.0 — free for individuals, researchers, non-profits, and organizations under $100K annual revenue. Attribution required. Above-threshold entities must obtain a separate written license. See [LICENSE.md](LICENSE.md).
