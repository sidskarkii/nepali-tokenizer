"""Load, clean, and cache Nepali + English benchmark data."""

import hashlib
import json
import lzma
import os
import random
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import BENCHMARK, CLEANING, CLEAN_DIR, DATA_DIR, DEVANAGARI_RANGE


def is_devanagari(cp: int) -> bool:
    return DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]


def devanagari_ratio(text: str) -> float:
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    return sum(1 for c in non_ws if is_devanagari(ord(c))) / len(non_ws)


def latin_ratio(text: str) -> float:
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    return sum(1 for c in non_ws if ord(c) < 0x0250) / len(non_ws)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if c == "\n" or c == "\t" or (not unicodedata.category(c).startswith("C")))
    # collapse multiple blank lines
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


def deduplicate(docs: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for doc in docs:
        h = hashlib.sha256(doc["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(doc)
    return out


def load_cc100_nepali() -> list[dict]:
    print("Loading CC-100 Nepali...")
    target = BENCHMARK["n_cc100_nepali_docs"]
    xz_path = DATA_DIR / "raw" / "ne.txt.xz"

    docs = []
    with lzma.open(xz_path, "rt", encoding="utf-8") as f:
        current_lines: list[str] = []
        for line in f:
            if line.strip() == "":
                if current_lines:
                    raw = "\n".join(current_lines)
                    text = clean_text(raw)
                    if (
                        len(text) >= CLEANING["min_doc_chars"]
                        and devanagari_ratio(text) >= CLEANING["min_devanagari_ratio"]
                    ):
                        docs.append({
                            "text": text,
                            "source": "cc100",
                            "category": "web",
                            "word_count": len(text.split()),
                            "char_count": len(text),
                            "byte_count": len(text.encode("utf-8")),
                        })
                    current_lines = []
                    if len(docs) >= target:
                        break
            else:
                current_lines.append(line.strip())

    docs = deduplicate(docs)
    print(f"  {len(docs)} docs, {sum(d['word_count'] for d in docs):,} words")
    return docs


def load_books() -> list[dict]:
    print("Loading books...")
    books_dir = DATA_DIR / "books"
    max_per_cat = BENCHMARK["books_max_per_category"]
    rng = random.Random(BENCHMARK["seed"])

    docs = []
    for cat_dir in sorted(books_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name

        txt_files = sorted(cat_dir.glob("*.txt"))
        if len(txt_files) > max_per_cat:
            txt_files = rng.sample(txt_files, max_per_cat)

        for fpath in txt_files:
            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            text = clean_text(raw)
            if (
                len(text) >= CLEANING["books_min_chars"]
                and devanagari_ratio(text) >= CLEANING["min_devanagari_ratio"]
            ):
                docs.append({
                    "text": text,
                    "source": "books",
                    "category": category,
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "byte_count": len(text.encode("utf-8")),
                })

    docs = deduplicate(docs)

    cat_counts = {}
    for d in docs:
        cat_counts[d["category"]] = cat_counts.get(d["category"], 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} docs")
    print(f"  Total: {len(docs)} docs, {sum(d['word_count'] for d in docs):,} words")
    return docs


def load_cc100_english() -> list[dict]:
    print("Loading CC-100 English (streaming from HuggingFace)...")
    from datasets import load_dataset

    target = BENCHMARK["n_cc100_english_docs"]
    ds = load_dataset("cc100", lang="en", split="train", streaming=True, trust_remote_code=True)

    docs = []
    checked = 0
    for row in ds:
        checked += 1
        raw = row.get("text", "")
        text = clean_text(raw)
        if (
            len(text) >= CLEANING["min_doc_chars"]
            and latin_ratio(text) >= CLEANING["min_latin_ratio"]
        ):
            docs.append({
                "text": text,
                "source": "cc100",
                "category": "english",
                "word_count": len(text.split()),
                "char_count": len(text),
                "byte_count": len(text.encode("utf-8")),
            })
        if len(docs) >= target:
            break
        if checked % 500 == 0:
            print(f"  checked {checked}, kept {len(docs)}...", flush=True)

    docs = deduplicate(docs)
    print(f"  {len(docs)} docs, {sum(d['word_count'] for d in docs):,} words")
    return docs


def save_jsonl(docs: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Saved {len(docs)} docs to {path}")


def main():
    ne_cc100 = load_cc100_nepali()
    ne_books = load_books()
    en_cc100 = load_cc100_english()

    save_jsonl(ne_cc100, CLEAN_DIR / "nepali_cc100.jsonl")
    save_jsonl(ne_books, CLEAN_DIR / "nepali_books.jsonl")
    save_jsonl(en_cc100, CLEAN_DIR / "english_cc100.jsonl")

    total_ne_words = sum(d["word_count"] for d in ne_cc100) + sum(d["word_count"] for d in ne_books)
    total_en_words = sum(d["word_count"] for d in en_cc100)
    print(f"\nSummary:")
    print(f"  Nepali: {len(ne_cc100) + len(ne_books)} docs, {total_ne_words:,} words")
    print(f"    CC-100: {len(ne_cc100)} docs")
    print(f"    Books:  {len(ne_books)} docs")
    print(f"  English: {len(en_cc100)} docs, {total_en_words:,} words")


if __name__ == "__main__":
    main()
