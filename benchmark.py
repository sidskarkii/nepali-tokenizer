"""Nepali tokenizer benchmark: efficiency metrics + vocabulary analysis."""

import json
import time
from pathlib import Path

import numpy as np

from config import (
    BENCHMARK,
    CLEAN_DIR,
    DEVANAGARI_RANGE,
    NEPALI_GRAPHEMES,
    RESULTS_DIR,
    TOKENIZER_REGISTRY,
    TOKENIZERS_DIR,
)


def is_devanagari(cp: int) -> bool:
    return DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]


# ── Data loading ──


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def load_clean_data():
    ne_cc100 = load_jsonl(CLEAN_DIR / "nepali_cc100.jsonl")
    ne_books = load_jsonl(CLEAN_DIR / "nepali_books.jsonl")
    en_cc100 = load_jsonl(CLEAN_DIR / "english_cc100.jsonl")
    return ne_cc100, ne_books, en_cc100


# ── Tokenizer loading ──


def load_tokenizer(entry: dict):
    if entry["local"] is None:
        import tiktoken
        enc = tiktoken.get_encoding(entry["model_id"])
        return enc.encode, enc.n_vocab, "tiktoken"

    path = TOKENIZERS_DIR / entry["local"]
    if not path.exists():
        return None, None, f"not found: {path}"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    encode_fn = lambda text, _tok=tok: _tok.encode(text, add_special_tokens=False)
    return encode_fn, len(tok), "hf"


# ── Per-document metrics ──


def doc_metrics(text: str, token_ids: list[int]) -> dict:
    n_tokens = len(token_ids)
    n_bytes = len(text.encode("utf-8"))
    n_words = len(text.split())
    return {
        "tokens_per_byte": n_tokens / n_bytes if n_bytes else 0,
        "tokens_per_word": n_tokens / n_words if n_words else 0,
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "n_words": n_words,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(BENCHMARK["seed"])
    means = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def aggregate_metrics(all_doc_metrics: list[dict]) -> dict:
    tpb = np.array([m["tokens_per_byte"] for m in all_doc_metrics])
    tpw = np.array([m["tokens_per_word"] for m in all_doc_metrics])
    ci_lo, ci_hi = bootstrap_ci(tpb) if len(tpb) >= 10 else (tpb.mean(), tpb.mean())

    return {
        "n_docs": len(all_doc_metrics),
        "total_tokens": sum(m["n_tokens"] for m in all_doc_metrics),
        "total_bytes": sum(m["n_bytes"] for m in all_doc_metrics),
        "total_words": sum(m["n_words"] for m in all_doc_metrics),
        "tokens_per_byte": {"mean": float(tpb.mean()), "median": float(np.median(tpb)),
                            "p10": float(np.percentile(tpb, 10)), "p90": float(np.percentile(tpb, 90)),
                            "ci_95": [ci_lo, ci_hi]},
        "tokens_per_word": {"mean": float(tpw.mean()), "median": float(np.median(tpw)),
                            "p10": float(np.percentile(tpw, 10)), "p90": float(np.percentile(tpw, 90))},
    }


# ── Vocabulary analysis ──


def analyze_vocab_hf(tok_path: Path) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True)
    vocab_size = len(tok)

    dev_tokens = []
    dev_lengths = []
    for tid in range(vocab_size):
        try:
            decoded = tok.decode([tid])
        except Exception:
            continue
        if any(is_devanagari(ord(c)) for c in decoded):
            dev_tokens.append(decoded)
            dev_lengths.append(len(decoded))

    grapheme_hits = 0
    for g in NEPALI_GRAPHEMES:
        if any(g in t for t in dev_tokens):
            grapheme_hits += 1

    return {
        "vocab_size": vocab_size,
        "devanagari_count": len(dev_tokens),
        "devanagari_pct": round(len(dev_tokens) / vocab_size * 100, 2) if vocab_size else 0,
        "grapheme_coverage": round(grapheme_hits / len(NEPALI_GRAPHEMES) * 100, 1),
        "avg_devanagari_token_len": round(np.mean(dev_lengths), 1) if dev_lengths else 0,
        "top_tokens": sorted(dev_tokens, key=len, reverse=True)[:20],
    }


def analyze_vocab_tiktoken(encoding_name: str) -> dict:
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    vocab_size = enc.n_vocab

    dev_tokens = []
    dev_lengths = []
    for tid in range(vocab_size):
        try:
            decoded = enc.decode([tid])
        except Exception:
            continue
        if any(is_devanagari(ord(c)) for c in decoded):
            dev_tokens.append(decoded)
            dev_lengths.append(len(decoded))

    grapheme_hits = 0
    for g in NEPALI_GRAPHEMES:
        if any(g in t for t in dev_tokens):
            grapheme_hits += 1

    return {
        "vocab_size": vocab_size,
        "devanagari_count": len(dev_tokens),
        "devanagari_pct": round(len(dev_tokens) / vocab_size * 100, 2) if vocab_size else 0,
        "grapheme_coverage": round(grapheme_hits / len(NEPALI_GRAPHEMES) * 100, 1),
        "avg_devanagari_token_len": round(np.mean(dev_lengths), 1) if dev_lengths else 0,
        "top_tokens": sorted(dev_tokens, key=len, reverse=True)[:20],
    }


# ── Main benchmark ──


def run_benchmark():
    ne_cc100, ne_books, en_cc100 = load_clean_data()
    print(f"Loaded: {len(ne_cc100)} NE-CC100, {len(ne_books)} NE-Books, {len(en_cc100)} EN-CC100\n")

    strata = {
        "nepali_cc100": ne_cc100,
        "nepali_books": ne_books,
        "english": en_cc100,
    }

    results = []

    for entry in TOKENIZER_REGISTRY:
        name = entry["name"]
        print(f"Testing {name}...", end=" ", flush=True)
        t0 = time.time()

        encode_fn, vocab_size, status = load_tokenizer(entry)
        if encode_fn is None:
            print(f"SKIPPED ({status})")
            continue

        result = {
            "name": name,
            "family": entry["family"],
            "gen": entry["gen"],
            "model_id": entry["model_id"],
            "vocab_size": vocab_size,
            "strata": {},
        }

        for stratum_name, docs in strata.items():
            metrics_list = []
            for doc in docs:
                try:
                    token_ids = encode_fn(doc["text"])
                    if not token_ids and stratum_name.startswith("nepali"):
                        continue
                except Exception:
                    continue
                metrics_list.append(doc_metrics(doc["text"], token_ids))
            result["strata"][stratum_name] = aggregate_metrics(metrics_list)
            if stratum_name.startswith("nepali") and metrics_list and metrics_list[0]["n_tokens"] == 0:
                result["broken"] = True

        # Nepali tax: combined NE tokens-per-word / EN tokens-per-word
        ne_total_tokens = (result["strata"]["nepali_cc100"]["total_tokens"]
                           + result["strata"]["nepali_books"]["total_tokens"])
        ne_total_words = (result["strata"]["nepali_cc100"]["total_words"]
                          + result["strata"]["nepali_books"]["total_words"])
        en_total_tokens = result["strata"]["english"]["total_tokens"]
        en_total_words = result["strata"]["english"]["total_words"]
        ne_tpw = ne_total_tokens / ne_total_words if ne_total_words else 0
        en_tpw = en_total_tokens / en_total_words if en_total_words else 0
        result["nepali_tax"] = round(ne_tpw / en_tpw, 1) if en_tpw else 0
        result["ne_tokens_per_word"] = round(ne_tpw, 2)
        result["en_tokens_per_word"] = round(en_tpw, 2)
        result["failed_docs"] = sum(
            s["n_docs"] for s in [result["strata"]["nepali_cc100"], result["strata"]["nepali_books"]]
        )
        result["failed_docs"] = (len(ne_cc100) + len(ne_books)) - result["failed_docs"]

        # Vocab analysis
        if entry["local"] is None:
            result["vocab"] = analyze_vocab_tiktoken(entry["model_id"])
        else:
            result["vocab"] = analyze_vocab_hf(TOKENIZERS_DIR / entry["local"])

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s, tax={result['nepali_tax']:.1f}x)")
        results.append(result)

    return results


def print_results(results: list[dict]):
    sorted_results = sorted(results, key=lambda r: r["nepali_tax"], reverse=True)

    print("\n" + "=" * 100)
    print(f"{'Model':<20} {'Gen':<5} {'Vocab':>8} │ {'NE tok/word':>11} {'EN tok/word':>11} │ {'Tax':>6} │ {'Dev Tokens':>10} {'Dev%':>5}")
    print("=" * 100)

    for r in sorted_results:
        v = r["vocab"]
        if r["nepali_tax"] < 0.5:
            print(
                f"{r['name']:<20} {r['gen']:<5} {r['vocab_size']:>8,} │ "
                f"{'DROPS NE':>11} {r['en_tokens_per_word']:>11.2f} │ "
                f"{'  N/A':>6} │ "
                f"{v['devanagari_count']:>10,} {v['devanagari_pct']:>4.1f}%"
            )
        else:
            print(
                f"{r['name']:<20} {r['gen']:<5} {r['vocab_size']:>8,} │ "
                f"{r['ne_tokens_per_word']:>11.2f} {r['en_tokens_per_word']:>11.2f} │ "
                f"{r['nepali_tax']:>5.1f}x │ "
                f"{v['devanagari_count']:>10,} {v['devanagari_pct']:>4.1f}%"
            )

    # Family comparison
    families = {}
    for r in results:
        families.setdefault(r["family"], []).append(r)

    multi_gen = {f: rs for f, rs in families.items() if len(rs) > 1}
    if multi_gen:
        print("\n── Family Progression (Nepali Tax) ──")
        for family, rs in sorted(multi_gen.items()):
            rs_sorted = sorted(rs, key=lambda r: r["gen"])
            taxes = [f"{r['name']} ({r['gen']}): {r['nepali_tax']:.1f}x" for r in rs_sorted]
            print(f"  {family}: {' → '.join(taxes)}")

    print(f"\n'Tax' = Nepali tokens-per-word / English tokens-per-word.")
    print(f"A tax of 3.0x means Nepali costs 3x more tokens = 3x API cost, 1/3 context window.")


def save_results(results: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


def main():
    results = run_benchmark()
    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
