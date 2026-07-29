#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Tuple

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
RE_WS = re.compile(r"\s+")


def html_to_title_and_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    node = soup.body if soup.body else soup
    text = node.get_text(" ", strip=True)
    return RE_WS.sub(" ", title).strip(), RE_WS.sub(" ", text).strip()


def open_shard(out_dir: Path, shard_id: int):
    path = out_dir / f"shard-{shard_id:05d}.jsonl"
    return path.open("w", encoding="utf-8"), path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--docs_per_shard", type=int, default=2000)
    p.add_argument("--encoding", default="utf-8")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(
        list(input_dir.rglob("*.html")) +
        list(input_dir.rglob("*.htm"))
    )
    if not html_files:
        raise SystemExit(f"No .html/.htm files found under: {input_dir}")

    shard_id = 0
    docs_in_shard = 0
    written = 0
    skipped = 0

    out_f, shard_path = open_shard(out_dir, shard_id)

    try:
        for fp in tqdm(html_files, desc="HTML → JSONL shards"):
            html = fp.read_text(encoding=args.encoding, errors="replace")
            title, body_text = html_to_title_and_text(html)

            doc_id = fp.stem  # filename without extension

            # Build text (title + body)
            text = f"{title}\n\n{body_text}" if title else body_text
            text = (text or "").strip()

            # SKIP empty text documents
            if not text:
                skipped += 1
                continue

            out_f.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "text": text,
                        "title": title
                    },
                    ensure_ascii=False
                ) + "\n"
            )

            written += 1
            docs_in_shard += 1

            if docs_in_shard >= args.docs_per_shard:
                out_f.close()
                shard_id += 1
                docs_in_shard = 0
                out_f, shard_path = open_shard(out_dir, shard_id)

    finally:
        try:
            out_f.close()
        except Exception:
            pass

    if docs_in_shard == 0:
        try:
            shard_path.unlink()
        except Exception:
            pass

    print(f"Done. Wrote {written:,} docs into {out_dir}")
    print(f"Skipped {skipped:,} empty documents")


if __name__ == "__main__":
    main()
