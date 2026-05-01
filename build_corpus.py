"""Build tokenizer training corpus from multiple Nepali sources."""

import hashlib
import lzma
import os
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import DATA_DIR, DEVANAGARI_RANGE

CORPUS_DIR = DATA_DIR / "corpus"
OUTPUT_FILE = CORPUS_DIR / "nepali_training_corpus.txt"

# Target sizes per source (in chars, ~1 byte per char for counting purposes)
# We want ~5-7GB total. Devanagari is 3 bytes/char, so ~2B chars ≈ 6GB
TARGETS = {
    "culturax": 800_000_000,   # ~2.4GB — largest, cleanest web source
    "sangraha": 500_000_000,   # ~1.5GB — verified, high quality
    "cc100": None,             # use all (~1GB uncompressed)
    "books": None,             # use all (~72MB)
}


def is_devanagari(cp: int) -> bool:
    return DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]


def devanagari_ratio(text: str) -> float:
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    return sum(1 for c in non_ws if is_devanagari(ord(c))) / len(non_ws)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        c for c in text
        if c == "\n" or c == "\t"
        or c == "‍"  # ZWJ — used in Devanagari conjuncts
        or c == "‌"  # ZWNJ — used in Devanagari to break conjuncts
        or not unicodedata.category(c).startswith("C")
    )
    lines = text.split("\n")
    cleaned = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(stripped)
            prev_blank = False
    return "\n".join(cleaned).strip()


def stream_culturax(target_chars: int):
    """Stream CulturaX Nepali, yield cleaned documents."""
    from datasets import load_dataset

    print(f"  Streaming CulturaX Nepali (target: {target_chars / 1e9:.1f}B chars)...")
    ds = load_dataset("uonlp/CulturaX", "ne", split="train", streaming=True, trust_remote_code=True)

    total_chars = 0
    kept = 0
    checked = 0
    for row in ds:
        checked += 1
        text = clean_text(row["text"])
        if len(text) < 200 or devanagari_ratio(text) < 0.5:
            continue
        yield text
        kept += 1
        total_chars += len(text)
        if checked % 10000 == 0:
            print(f"    checked {checked:,}, kept {kept:,}, {total_chars / 1e6:.0f}M chars", flush=True)
        if total_chars >= target_chars:
            break

    print(f"    Final: {kept:,} docs, {total_chars / 1e6:.0f}M chars")


def stream_sangraha(target_chars: int):
    """Stream ai4bharat/sangraha verified Nepali."""
    from datasets import load_dataset

    print(f"  Streaming Sangraha verified/nep (target: {target_chars / 1e9:.1f}B chars)...")
    ds = load_dataset("ai4bharat/sangraha", "verified", split="nep", streaming=True, trust_remote_code=True)

    total_chars = 0
    kept = 0
    checked = 0
    for row in ds:
        checked += 1
        text = clean_text(row["text"])
        if len(text) < 200 or devanagari_ratio(text) < 0.5:
            continue
        yield text
        kept += 1
        total_chars += len(text)
        if checked % 10000 == 0:
            print(f"    checked {checked:,}, kept {kept:,}, {total_chars / 1e6:.0f}M chars", flush=True)
        if total_chars >= target_chars:
            break

    print(f"    Final: {kept:,} docs, {total_chars / 1e6:.0f}M chars")


def load_cc100():
    """Load all CC-100 Nepali from local xz file."""
    print("  Loading CC-100 Nepali (full)...")
    xz_path = DATA_DIR / "raw" / "ne.txt.xz"

    docs = []
    total_chars = 0
    with lzma.open(xz_path, "rt", encoding="utf-8") as f:
        current_lines = []
        for line in f:
            if line.strip() == "":
                if current_lines:
                    raw = "\n".join(current_lines)
                    text = clean_text(raw)
                    if len(text) >= 200 and devanagari_ratio(text) >= 0.5:
                        docs.append(text)
                        total_chars += len(text)
                    current_lines = []
            else:
                current_lines.append(line.strip())

        if current_lines:
            raw = "\n".join(current_lines)
            text = clean_text(raw)
            if len(text) >= 200 and devanagari_ratio(text) >= 0.5:
                docs.append(text)
                total_chars += len(text)

    print(f"    {len(docs):,} docs, {total_chars / 1e6:.0f}M chars")
    return docs


def load_books():
    """Load all books from local directory."""
    print("  Loading books...")
    books_dir = DATA_DIR / "books"
    docs = []
    total_chars = 0

    for fpath in sorted(books_dir.rglob("*.txt")):
        try:
            raw = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        text = clean_text(raw)
        if len(text) >= 500 and devanagari_ratio(text) >= 0.5:
            docs.append(text)
            total_chars += len(text)

    print(f"    {len(docs):,} docs, {total_chars / 1e6:.0f}M chars")
    return docs


def deduplicate_file(path: Path):
    """Deduplicate the corpus file by paragraph hashes."""
    print("Deduplicating...")
    seen = set()
    total_lines = 0
    kept_lines = 0
    temp_path = path.with_suffix(".deduped.txt")

    with open(path, encoding="utf-8") as fin, open(temp_path, "w", encoding="utf-8") as fout:
        current_para = []
        for line in fin:
            if line.strip() == "":
                if current_para:
                    para = "\n".join(current_para)
                    h = hashlib.md5(para.encode()).hexdigest()
                    total_lines += 1
                    if h not in seen:
                        seen.add(h)
                        fout.write(para + "\n\n")
                        kept_lines += 1
                    current_para = []
            else:
                current_para.append(line.rstrip())

        if current_para:
            para = "\n".join(current_para)
            h = hashlib.md5(para.encode()).hexdigest()
            total_lines += 1
            if h not in seen:
                seen.add(h)
                fout.write(para + "\n\n")
                kept_lines += 1

    temp_path.rename(path)
    print(f"  {total_lines:,} paragraphs → {kept_lines:,} unique ({100 * kept_lines / total_lines:.1f}%)")


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building tokenizer training corpus...\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Books first (smallest, highest quality)
        for doc in load_books():
            f.write(doc + "\n\n")

        # CC-100 (local, no network)
        for doc in load_cc100():
            f.write(doc + "\n\n")

        # Sangraha (streamed)
        for doc in stream_sangraha(TARGETS["sangraha"]):
            f.write(doc + "\n\n")

        # CulturaX (streamed, largest)
        for doc in stream_culturax(TARGETS["culturax"]):
            f.write(doc + "\n\n")

    # Deduplicate
    deduplicate_file(OUTPUT_FILE)

    # Report
    size_bytes = OUTPUT_FILE.stat().st_size
    line_count = sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8"))
    print(f"\nCorpus ready: {OUTPUT_FILE}")
    print(f"  Size: {size_bytes / 1e9:.2f} GB")
    print(f"  Lines: {line_count:,}")


if __name__ == "__main__":
    main()
