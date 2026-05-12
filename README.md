# Nepali Tokenizer Infrastructure

Most LLM tokenizers need 2x–6x more tokens for Nepali than English. This repo measures that gap across 17 models, trains a Nepali-optimized tokenizer, and extends existing model tokenizers to close the gap.

The end-to-end experiment extends the Qwen3-4B tokenizer with 15K Nepali tokens, then runs LoRA continued pretraining and instruction fine-tuning. The extended tokenizer cuts Nepali token count by 48% on a 2,000-doc benchmark split.

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
