"""Train and evaluate standalone Nepali SentencePiece BPE tokenizers.

This script:
1. Trains SentencePiece BPE models at 32K, 48K, and 64K vocab sizes.
2. Evaluates fertility (tokens per word) on `data/clean/nepali_cc100.jsonl`.
3. Compares against the target model tokenizers and Aananda-giri/NepaliBPE.
4. Saves models, metadata, and evaluation summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from config import CLEAN_DIR, RESULTS_DIR, TOKENIZERS_DIR

CORPUS_PATH = Path("data/corpus/nepali_training_corpus.txt")
BENCHMARK_PATH = CLEAN_DIR / "nepali_cc100.jsonl"
OUTPUT_ROOT = TOKENIZERS_DIR / "sentencepiece_nepali_bpe"
RESULTS_ROOT = RESULTS_DIR / "tokenizer_training"

DEFAULT_VOCAB_SIZES = (32_000, 48_000, 64_000)
DEFAULT_MODEL_PREFIX = "nepali_bpe"

COMPARISON_TOKENIZERS = [
    {
        "name": "Aananda-giri/NepaliBPE",
        "kind": "hf",
        "local_dir": None,
        "model_id": "Aananda-giri/NepaliBPE",
    },
    {
        "name": "Phi-4",
        "kind": "hf",
        "local_dir": TOKENIZERS_DIR / "phi4",
        "model_id": "microsoft/phi-4",
    },
    {
        "name": "Qwen 3.5",
        "kind": "hf",
        "local_dir": TOKENIZERS_DIR / "qwen3.5",
        "model_id": "Qwen/Qwen3.5-27B",
    },
    {
        "name": "DeepSeek V4",
        "kind": "hf",
        "local_dir": TOKENIZERS_DIR / "deepseek-v4",
        "model_id": "deepseek-ai/DeepSeek-V4-Flash",
    },
    {
        "name": "Kimi K2.6",
        "kind": "hf",
        "local_dir": TOKENIZERS_DIR / "kimi-k2.6",
        "model_id": "moonshotai/Kimi-K2.6",
    },
]


@dataclass
class EvalResult:
    name: str
    source: str
    vocab_size: int
    docs: int
    failed_docs: int
    total_tokens: int
    total_words: int
    total_bytes: int
    tokens_per_word: float
    bytes_per_token: float
    chars_per_token: float
    seconds: float
    model_path: str | None = None
    notes: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help="Path to the training corpus text file.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=BENCHMARK_PATH,
        help="Path to the Nepali benchmark JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for trained SentencePiece models.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_ROOT,
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_VOCAB_SIZES),
        help="Vocabulary sizes to train.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="SentencePiece training thread count.",
    )
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=10_000_000,
        help="SentencePiece reservoir sample size.",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="SentencePiece character coverage.",
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=16_384,
        help="SentencePiece max sentence length in bytes.",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default=DEFAULT_MODEL_PREFIX,
        help="Base prefix for trained models.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and evaluate existing local SentencePiece models only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if the model files already exist.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    docs: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def benchmark_docs(path: Path) -> list[dict]:
    docs = load_jsonl(path)
    if not docs:
        raise ValueError(f"No benchmark documents found in {path}")
    return docs


def sentencepiece_training_args(
    corpus_path: Path,
    model_prefix: Path,
    vocab_size: int,
    args: argparse.Namespace,
) -> dict:
    return {
        "input": str(corpus_path),
        "model_prefix": str(model_prefix),
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "character_coverage": args.character_coverage,
        "byte_fallback": True,
        "split_digits": False,
        "split_by_number": True,
        "split_by_unicode_script": True,
        "split_by_whitespace": True,
        "normalization_rule_name": "identity",
        "remove_extra_whitespaces": True,
        "shuffle_input_sentence": True,
        "input_sentence_size": args.input_sentence_size,
        "max_sentence_length": args.max_sentence_length,
        "max_sentencepiece_length": 24,
        "num_threads": args.num_threads,
        "train_extremely_large_corpus": True,
        "hard_vocab_limit": False,
        "unk_id": 0,
        "bos_id": -1,
        "eos_id": -1,
        "pad_id": -1,
    }


def train_sentencepiece_model(
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int,
    args: argparse.Namespace,
) -> Path:
    import sentencepiece as spm

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"{args.model_prefix}_{vocab_size // 1000}k"
    model_path = prefix.with_suffix(".model")

    if model_path.exists() and not args.force:
        print(f"[train] Reusing existing model: {model_path}")
        return model_path

    train_args = sentencepiece_training_args(corpus_path, prefix, vocab_size, args)
    print(f"[train] Training {vocab_size:,} vocab -> {model_path}")
    start = time.time()
    spm.SentencePieceTrainer.train(**train_args)
    elapsed = time.time() - start

    meta = {
        "trained_at_unix": int(time.time()),
        "elapsed_seconds": round(elapsed, 2),
        "corpus": str(corpus_path),
        "model": str(model_path),
        "trainer_args": train_args,
    }
    with prefix.with_suffix(".train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    return model_path


def encode_with_sentencepiece(model_path: Path) -> tuple[Callable[[str], list[int]], int]:
    import sentencepiece as spm

    proc = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(text: str) -> list[int]:
        return proc.encode(text, out_type=int)

    return encode, proc.get_piece_size()


def load_hf_tokenizer(local_dir: Path | None, model_id: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    candidate = local_dir if local_dir and local_dir.exists() else model_id

    try:
        tok = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
        return tok
    except Exception as first_error:
        if local_dir:
            tokenizer_json = local_dir / "tokenizer.json"
            if tokenizer_json.exists():
                try:
                    tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
                    return tok
                except Exception:
                    pass
        raise RuntimeError(f"failed to load tokenizer from {candidate}: {first_error}") from first_error


def encode_with_hf(local_dir: Path | None, model_id: str) -> tuple[Callable[[str], list[int]], int, str]:
    tok = load_hf_tokenizer(local_dir, model_id)

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    source = str(local_dir) if local_dir and local_dir.exists() else model_id
    return encode, len(tok), source


def evaluate_tokenizer(
    name: str,
    source: str,
    encode_fn: Callable[[str], list[int]],
    vocab_size: int,
    docs: list[dict],
    *,
    model_path: Path | None = None,
    notes: str | None = None,
) -> EvalResult:
    start = time.time()
    total_tokens = 0
    total_words = 0
    total_bytes = 0
    total_chars = 0
    failed_docs = 0

    for doc in docs:
        text = doc["text"]
        word_count = int(doc.get("word_count") or len(text.split()))
        byte_count = int(doc.get("byte_count") or len(text.encode("utf-8")))
        char_count = int(doc.get("char_count") or len(text))
        try:
            token_ids = encode_fn(text)
        except Exception:
            failed_docs += 1
            continue
        if not token_ids:
            failed_docs += 1
            continue
        total_tokens += len(token_ids)
        total_words += word_count
        total_bytes += byte_count
        total_chars += char_count

    docs_ok = len(docs) - failed_docs
    tokens_per_word = total_tokens / total_words if total_words else 0.0
    bytes_per_token = total_bytes / total_tokens if total_tokens else 0.0
    chars_per_token = total_chars / total_tokens if total_tokens else 0.0
    elapsed = time.time() - start

    return EvalResult(
        name=name,
        source=source,
        vocab_size=vocab_size,
        docs=docs_ok,
        failed_docs=failed_docs,
        total_tokens=total_tokens,
        total_words=total_words,
        total_bytes=total_bytes,
        tokens_per_word=tokens_per_word,
        bytes_per_token=bytes_per_token,
        chars_per_token=chars_per_token,
        seconds=elapsed,
        model_path=str(model_path) if model_path else None,
        notes=notes,
    )


def evaluate_sentencepiece_models(
    model_paths: list[Path],
    docs: list[dict],
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for model_path in model_paths:
        encode_fn, vocab_size = encode_with_sentencepiece(model_path)
        result = evaluate_tokenizer(
            name=model_path.stem,
            source="sentencepiece",
            encode_fn=encode_fn,
            vocab_size=vocab_size,
            docs=docs,
            model_path=model_path,
        )
        print(
            f"[eval] {result.name:<24} "
            f"tpw={result.tokens_per_word:.4f} "
            f"bytes/token={result.bytes_per_token:.4f} "
            f"failed={result.failed_docs}"
        )
        results.append(result)
    return results


def evaluate_baselines(docs: list[dict]) -> list[EvalResult]:
    results: list[EvalResult] = []

    for spec in COMPARISON_TOKENIZERS:
        try:
            encode_fn, vocab_size, source = encode_with_hf(spec["local_dir"], spec["model_id"])
            result = evaluate_tokenizer(
                name=spec["name"],
                source=source,
                encode_fn=encode_fn,
                vocab_size=vocab_size,
                docs=docs,
            )
            print(
                f"[eval] {result.name:<24} "
                f"tpw={result.tokens_per_word:.4f} "
                f"bytes/token={result.bytes_per_token:.4f} "
                f"failed={result.failed_docs}"
            )
            results.append(result)
        except Exception as error:
            result = EvalResult(
                name=spec["name"],
                source=spec["model_id"],
                vocab_size=0,
                docs=0,
                failed_docs=len(docs),
                total_tokens=0,
                total_words=0,
                total_bytes=0,
                tokens_per_word=0.0,
                bytes_per_token=0.0,
                chars_per_token=0.0,
                seconds=0.0,
                notes=str(error),
            )
            print(f"[eval] {spec['name']:<24} skipped: {error}")
            results.append(result)

    return results


def save_results(results_dir: Path, results: list[EvalResult], corpus: Path, benchmark: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"sentencepiece_sweep_{timestamp}.json"
    csv_path = results_dir / f"sentencepiece_sweep_{timestamp}.csv"
    latest_json = results_dir / "latest_sentencepiece_sweep.json"
    latest_csv = results_dir / "latest_sentencepiece_sweep.csv"

    payload = {
        "created_at": timestamp,
        "corpus": str(corpus),
        "benchmark": str(benchmark),
        "results": [asdict(r) for r in results],
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with latest_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    fieldnames = list(asdict(results[0]).keys()) if results else list(EvalResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    with latest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    print(f"[save] {json_path}")
    print(f"[save] {csv_path}")


def print_summary(results: list[EvalResult]) -> None:
    ranked = sorted(
        results,
        key=lambda item: (item.tokens_per_word == 0.0, item.tokens_per_word),
    )

    print("\n=== Fertility Summary (lower is better) ===")
    print(
        f"{'Tokenizer':<24} {'Vocab':>8} {'Tok/Word':>10} "
        f"{'Bytes/Tok':>10} {'Docs':>6} {'Fail':>6}"
    )
    for item in ranked:
        print(
            f"{item.name[:24]:<24} "
            f"{item.vocab_size:>8,} "
            f"{item.tokens_per_word:>10.4f} "
            f"{item.bytes_per_token:>10.4f} "
            f"{item.docs:>6} "
            f"{item.failed_docs:>6}"
        )


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.corpus.exists() and not args.skip_train:
        raise FileNotFoundError(f"Training corpus not found: {args.corpus}")
    if not args.benchmark.exists():
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark}")


def main() -> int:
    args = parse_args()
    validate_inputs(args)

    docs = benchmark_docs(args.benchmark)
    print(f"[data] Loaded {len(docs):,} benchmark docs from {args.benchmark}")

    model_paths: list[Path] = []
    for vocab_size in args.vocab_sizes:
        model_path = args.output_dir / f"{args.model_prefix}_{vocab_size // 1000}k.model"
        if args.skip_train:
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model for --skip-train: {model_path}")
        else:
            model_path = train_sentencepiece_model(args.corpus, args.output_dir, vocab_size, args)
        model_paths.append(model_path)

    sp_results = evaluate_sentencepiece_models(model_paths, docs)
    baseline_results = evaluate_baselines(docs)
    all_results = sp_results + baseline_results

    print_summary(all_results)
    save_results(args.results_dir, all_results, args.corpus, args.benchmark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
