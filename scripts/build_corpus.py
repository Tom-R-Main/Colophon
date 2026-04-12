#!/usr/bin/env python3
"""Download public domain texts from Project Gutenberg and build the Colophon corpus.

Usage:
    python scripts/build_corpus.py

Requires: pip install -e ".[all]" and spaCy models for each language.
Requires: pgvector running (docker compose up -d)
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

# Handle macOS SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
DB_URL = "postgresql://colophon:colophon_dev@localhost:5433/colophon"

# Author manifest: (author, title, gutenberg_id, lang)
# Gutenberg URLs: https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt
GUTENBERG_AUTHORS = [
    # === English ===
    ("Mark Twain", "Adventures of Huckleberry Finn", 76, "en"),
    ("Charles Dickens", "A Tale of Two Cities", 98, "en"),
    ("Charles Dickens", "Great Expectations", 1400, "en"),
    ("Jane Austen", "Pride and Prejudice", 1342, "en"),
    ("Jane Austen", "Emma", 158, "en"),
    ("Edgar Allan Poe", "The Works of Edgar Allan Poe Vol. 1", 2147, "en"),
    ("Herman Melville", "Moby-Dick", 2701, "en"),
    ("Oscar Wilde", "The Picture of Dorian Gray", 174, "en"),
    ("Oscar Wilde", "The Importance of Being Earnest", 844, "en"),
    ("Arthur Conan Doyle", "The Adventures of Sherlock Holmes", 1661, "en"),
    ("Jack London", "The Call of the Wild", 215, "en"),
    ("Jack London", "White Fang", 910, "en"),
    ("Louisa May Alcott", "Little Women", 514, "en"),
    ("Henry James", "The Turn of the Screw", 209, "en"),
    ("Henry James", "The Portrait of a Lady", 2833, "en"),
    ("Walt Whitman", "Leaves of Grass", 1322, "en"),
    ("Emily Dickinson", "Poems: Series One", 12242, "en"),
    ("Frederick Douglass", "Narrative of the Life of Frederick Douglass", 23, "en"),
    ("Booker T. Washington", "Up from Slavery", 2376, "en"),
    ("W.E.B. Du Bois", "The Souls of Black Folk", 408, "en"),
    ("Lewis Carroll", "Alice's Adventures in Wonderland", 11, "en"),
    ("H.P. Lovecraft", "The Call of Cthulhu", 68283, "en"),
    ("Robert Louis Stevenson", "Treasure Island", 120, "en"),
    ("H.G. Wells", "The War of the Worlds", 36, "en"),
    ("H.G. Wells", "The Time Machine", 35, "en"),
    ("Henry David Thoreau", "Walden", 205, "en"),
    # === French ===
    ("Victor Hugo", "Les Misérables Tome I", 17489, "fr"),
    ("Alexandre Dumas", "Les Trois Mousquetaires", 13951, "fr"),
    ("Gustave Flaubert", "Madame Bovary", 14155, "fr"),
    ("Marcel Proust", "Du côté de chez Swann", 7178, "fr"),
    # === Russian (English translations on Gutenberg) ===
    ("Fyodor Dostoevsky", "Crime and Punishment", 2554, "en"),
    # === German ===
    ("Brothers Grimm", "Grimms Märchen", 2591, "de"),
    ("Johann Wolfgang von Goethe", "Faust: Der Tragödie erster Teil", 2229, "de"),
    # === Italian ===
    ("Dante Alighieri", "La Divina Commedia", 1012, "it"),
    ("Niccolò Machiavelli", "Il Principe", 1232, "it"),
    ("Giovanni Boccaccio", "Il Decamerone", 3726, "it"),
    # === Portuguese ===
    ("Machado de Assis", "Dom Casmurro", 55752, "pt"),
    ("Machado de Assis", "Memórias Póstumas de Brás Cubas", 54829, "pt"),
    # === Polish ===
    ("Henryk Sienkiewicz", "Quo Vadis", 28850, "pl"),
]

# Authors we already have analyzed locally (skip download)
LOCAL_AUTHORS = {
    "Franz Kafka": ("assets/kafka.colophon.json", "assets/kafka.colophon.analysis.json", "de"),
    "Leo Tolstoy": ("assets/tolstoy_sample.colophon.json", "assets/tolstoy_sample.colophon.analysis.json", "ru"),
    "Miguel de Cervantes": ("assets/cervantes.colophon.json", "assets/cervantes.colophon.analysis.json", "es"),
}


def download_gutenberg(gutenberg_id: int, output_path: Path) -> bool:
    """Download a text from Project Gutenberg."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Colophon/0.1 (stylometric analysis tool)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                output_path.write_bytes(data)
                return True
        except Exception:
            continue
    return False


def strip_gutenberg_header(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate."""
    lines = text.split("\n")
    start = 0
    end = len(lines)

    for i, line in enumerate(lines):
        if "*** START OF" in line.upper() or "***START OF" in line.upper():
            start = i + 1
            break

    for i in range(len(lines) - 1, -1, -1):
        if "*** END OF" in lines[i].upper() or "***END OF" in lines[i].upper():
            end = i
            break

    return "\n".join(lines[start:end]).strip()


def main():
    CORPUS_DIR.mkdir(exist_ok=True)
    download_dir = CORPUS_DIR / "downloads"
    download_dir.mkdir(exist_ok=True)

    from colophon.analysis.pipeline import analyze
    from colophon.db import get_session
    from colophon.db.operations import index_profile
    from colophon.embeddings.vectorize import style_profile_to_vector
    from colophon.models.document import Document
    from colophon.models.features import StyleProfile

    session = get_session(DB_URL)
    total = len(GUTENBERG_AUTHORS) + len(LOCAL_AUTHORS)
    done = 0

    # --- Index local authors first ---
    for author, (doc_path, analysis_path, lang) in LOCAL_AUTHORS.items():
        done += 1
        analysis_file = Path(analysis_path)
        if not analysis_file.exists():
            print(f"[{done}/{total}] SKIP {author} — analysis file not found: {analysis_path}")
            continue

        print(f"[{done}/{total}] Indexing {author} (local)...")
        profile = StyleProfile.model_validate_json(analysis_file.read_text())
        vector = style_profile_to_vector(profile)
        index_profile(session, profile, vector, author=author, lang=lang)
        print(f"  ✓ {profile.document_title} ({profile.word_count:,} words)")

    # --- Download and process Gutenberg authors ---
    for author, title, gid, lang in GUTENBERG_AUTHORS:
        done += 1
        txt_path = download_dir / f"pg{gid}.txt"
        analysis_path = CORPUS_DIR / f"{gid}.analysis.json"

        # Skip if already analyzed
        if analysis_path.exists():
            print(f"[{done}/{total}] Indexing {author} — {title} (cached)...")
            profile = StyleProfile.model_validate_json(analysis_path.read_text())
            vector = style_profile_to_vector(profile)
            index_profile(session, profile, vector, author=author, lang=lang)
            print(f"  ✓ {profile.word_count:,} words")
            continue

        # Download
        if not txt_path.exists():
            print(f"[{done}/{total}] Downloading {author} — {title} (PG#{gid})...")
            if not download_gutenberg(gid, txt_path):
                print(f"  ✗ Download failed for PG#{gid}")
                continue
            time.sleep(1)  # Be polite to Gutenberg
        else:
            print(f"[{done}/{total}] Processing {author} — {title}...")

        # Read and strip Gutenberg boilerplate
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  ✗ Read error: {e}")
            continue

        text = strip_gutenberg_header(text)
        if len(text) < 1000:
            print(f"  ✗ Text too short after stripping ({len(text)} chars)")
            continue

        word_count = len(text.split())
        print(f"  Downloaded: {word_count:,} words")

        # Ingest
        doc = Document.from_text(text=text, source_path=str(txt_path), title=title, author=author)

        # Analyze
        print(f"  Analyzing ({lang})...")
        try:
            profile = analyze(doc, lang=lang)
        except Exception as e:
            print(f"  ✗ Analysis error: {e}")
            continue

        # Save analysis
        analysis_path.write_text(profile.model_dump_json(indent=2))

        # Index into pgvector
        vector = style_profile_to_vector(profile)
        index_profile(session, profile, vector, author=author, lang=lang)
        print(f"  ✓ Indexed ({profile.word_count:,} words)")

    session.close()
    print(f"\nDone. {done} authors processed.")


if __name__ == "__main__":
    main()
