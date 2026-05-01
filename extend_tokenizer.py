"""Extend target model tokenizers with Nepali tokens from our SentencePiece model.

For each target model:
1. Load the base tokenizer
2. Load our Nepali SentencePiece tokenizer
3. Extract delta tokens (tokens in ours that the base needs 3+ subtokens to represent)
4. Add delta tokens to the base tokenizer
5. Save the extended tokenizer
6. Evaluate fertility improvement
"""

import json
import sys
import time
from pathlib import Path

import sentencepiece as spm

from config import CLEAN_DIR, RESULTS_DIR, TOKENIZERS_DIR

SP_MODEL = TOKENIZERS_DIR / "sentencepiece_nepali_bpe" / "nepali_bpe_32k.model"
BENCHMARK_PATH = CLEAN_DIR / "nepali_cc100.jsonl"
OUTPUT_DIR = TOKENIZERS_DIR / "extended"
MAX_NEW_TOKENS = 16_000


def load_benchmark():
    docs = []
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def get_nepali_pieces(sp_model_path: Path) -> list[str]:
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    pieces = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece.startswith("▁"):
            clean = piece[1:]
        else:
            clean = piece
        if any(0x0900 <= ord(c) <= 0x097F for c in clean) and len(clean) >= 2:
            pieces.append(clean)
    return pieces


def compute_delta(nepali_pieces: list[str], base_tokenizer, max_tokens: int) -> list[str]:
    delta = []
    for piece in nepali_pieces:
        base_ids = base_tokenizer.encode(piece, add_special_tokens=False)
        if len(base_ids) >= 3:
            delta.append((piece, len(base_ids)))

    delta.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in delta[:max_tokens]]


def evaluate_fertility(encode_fn, docs: list[dict]) -> dict:
    total_tokens = 0
    total_words = 0
    for doc in docs:
        try:
            ids = encode_fn(doc["text"])
            if not ids:
                continue
        except Exception:
            continue
        total_tokens += len(ids)
        total_words += len(doc["text"].split())
    tpw = total_tokens / total_words if total_words else 0
    return {"tokens_per_word": round(tpw, 4), "total_tokens": total_tokens, "total_words": total_words}


def extend_hf_tokenizer(name: str, local_dir: str, nepali_pieces: list[str], docs: list[dict]) -> dict:
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Extending {name}")
    print(f"{'='*60}")

    base_path = TOKENIZERS_DIR / local_dir
    tok = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    base_vocab = len(tok)

    encode_base = lambda text: tok.encode(text, add_special_tokens=False)
    before = evaluate_fertility(encode_base, docs)
    print(f"  Before: {before['tokens_per_word']:.2f} tok/word (vocab={base_vocab:,})")

    delta = compute_delta(nepali_pieces, tok, MAX_NEW_TOKENS)
    print(f"  Delta tokens to add: {len(delta):,}")
    print(f"  Sample: {delta[:10]}")

    num_added = tok.add_tokens(delta)
    new_vocab = len(tok)
    print(f"  Added: {num_added:,} (vocab: {base_vocab:,} → {new_vocab:,})")

    after = evaluate_fertility(encode_base, docs)
    improvement = (1 - after["tokens_per_word"] / before["tokens_per_word"]) * 100
    print(f"  After:  {after['tokens_per_word']:.2f} tok/word ({improvement:.1f}% reduction)")

    out_path = OUTPUT_DIR / f"{local_dir}-nepali"
    out_path.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(out_path))
    print(f"  Saved to {out_path}")

    return {
        "name": name,
        "base_vocab": base_vocab,
        "new_vocab": new_vocab,
        "tokens_added": num_added,
        "before_tpw": before["tokens_per_word"],
        "after_tpw": after["tokens_per_word"],
        "improvement_pct": round(improvement, 1),
        "output_path": str(out_path),
    }


def extend_kimi_tokenizer(nepali_pieces: list[str], docs: list[dict]) -> dict:
    from transformers import AutoTokenizer

    name = "Kimi K2.6"
    print(f"\n{'='*60}")
    print(f"Extending {name}")
    print(f"{'='*60}")

    base_path = TOKENIZERS_DIR / "kimi-k2.6"
    tok = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    base_vocab = len(tok)

    encode_base = lambda text: tok.encode(text, add_special_tokens=False)
    before = evaluate_fertility(encode_base, docs)
    print(f"  Before: {before['tokens_per_word']:.2f} tok/word (vocab={base_vocab:,})")

    delta = compute_delta(nepali_pieces, tok, MAX_NEW_TOKENS)
    print(f"  Delta tokens to add: {len(delta):,}")
    print(f"  Sample: {delta[:10]}")

    num_added = tok.add_tokens(delta)
    new_vocab = len(tok)
    print(f"  Added: {num_added:,} (vocab: {base_vocab:,} → {new_vocab:,})")

    after = evaluate_fertility(encode_base, docs)
    improvement = (1 - after["tokens_per_word"] / before["tokens_per_word"]) * 100
    print(f"  After:  {after['tokens_per_word']:.2f} tok/word ({improvement:.1f}% reduction)")

    out_path = OUTPUT_DIR / "kimi-k2.6-nepali"
    out_path.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(out_path))
    print(f"  Saved to {out_path}")

    return {
        "name": name,
        "base_vocab": base_vocab,
        "new_vocab": new_vocab,
        "tokens_added": num_added,
        "before_tpw": before["tokens_per_word"],
        "after_tpw": after["tokens_per_word"],
        "improvement_pct": round(improvement, 1),
        "output_path": str(out_path),
    }


def main():
    print("Loading Nepali pieces from SentencePiece model...")
    nepali_pieces = get_nepali_pieces(SP_MODEL)
    print(f"  {len(nepali_pieces):,} Nepali pieces extracted")

    print("Loading benchmark data...")
    docs = load_benchmark()
    print(f"  {len(docs):,} docs")

    results = []

    targets = [
        ("Phi-4", "phi4"),
        ("Qwen 3.5", "qwen3.5"),
        ("DeepSeek V4", "deepseek-v4"),
    ]

    for name, local_dir in targets:
        try:
            r = extend_hf_tokenizer(name, local_dir, nepali_pieces, docs)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")

    try:
        r = extend_kimi_tokenizer(nepali_pieces, docs)
        results.append(r)
    except Exception as e:
        print(f"  Kimi FAILED: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<16} {'Before':>8} {'After':>8} {'Δ':>8} {'Added':>8}")
    print("-" * 52)
    for r in results:
        print(f"{r['name']:<16} {r['before_tpw']:>7.2f}x {r['after_tpw']:>7.2f}x {r['improvement_pct']:>7.1f}% {r['tokens_added']:>7,}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "extension_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
