import os
import csv
from pathlib import Path
from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher

# ----------------------------
# Config
# ----------------------------
index_path = "index_dialectal_query_variants"
k = 100

dialect_files = {
    "MSA": "query_variants/MSA.tsv",
    "Algerian": "query_variants/Algerian.tsv",
    "Egyptian": "query_variants/Egyptian.tsv",
    "Palestinian": "query_variants/Palestinian.tsv",
    "Qatari": "query_variants/Qatari.tsv",
    "Saudi": "query_variants/Saudi.tsv",
    "Syrian": "query_variants/Syrian.tsv",
}

out_dir = Path("runs_dialect")
out_dir.mkdir(parents=True, exist_ok=True)

EXCLUDE_FROM_COMBINED = {"MSA"}     # exclude MSA
DEDUP_QUERIES_IN_COMBINED = True   # remove exact duplicate query strings per topic

# ----------------------------
# Initialize searcher
# ----------------------------
searcher = LuceneSearcher(index_path)
searcher.set_language("ar")
searcher.set_bm25(k1=0.9, b=0.4)

def _sorted_query_cols(fieldnames):
    """Sort query columns numerically: query_1, query_2, ..."""
    def qnum(c):
        try:
            return int(c.split("_")[1])
        except Exception:
            return 10**9

    return sorted(
        [c for c in (fieldnames or []) if c and c.lower().startswith("query_")],
        key=qnum
    )

def iter_queries_from_tsv(tsv_path: str):
    """
    Per-file BM25 behavior (unchanged):
    qid = ArTest_topic_id_{variant_idx:02d} where variant_idx comes from query_#
    """
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return

        query_cols = _sorted_query_cols(reader.fieldnames)

        for row in reader:
            artest_id = (row.get("ArTest_topic_id") or "").strip()
            if not artest_id:
                continue

            for col in query_cols:
                q = (row.get(col) or "").strip()
                if not q:
                    continue

                try:
                    variant_idx = int(col.split("_")[1])
                except Exception:
                    continue

                qid = f"{artest_id}_{variant_idx:02d}"
                yield qid, q

def read_queries_grouped_by_topic(tsv_path: str):
    """
    For combined run:
    Returns dict: { ArTest_topic_id: [q1, q2, ...] }
    Preserves row order then query_1..query_n.
    """
    topic2queries = defaultdict(list)

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return topic2queries

        query_cols = _sorted_query_cols(reader.fieldnames)

        for row in reader:
            artest_id = (row.get("ArTest_topic_id") or "").strip()
            if not artest_id:
                continue

            for col in query_cols:
                q = (row.get(col) or "").strip()
                if q:
                    topic2queries[artest_id].append(q)

    return topic2queries

# ----------------------------
# (A) Retrieval per dialect file 
# ----------------------------
for dialect_name, tsv_path in dialect_files.items():
    if not os.path.exists(tsv_path):
        print(f"Skipping missing file: {tsv_path}")
        continue

    run_tag = "bm25"
    run_path = out_dir / f"bm25_{dialect_name}_top{k}.trec"

    with open(run_path, "w", encoding="utf-8") as out:
        for qid, query in iter_queries_from_tsv(tsv_path):
            hits = searcher.search(query, k=k)
            for rank, hit in enumerate(hits, start=1):
                out.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {run_tag}\n")

    print(f"Wrote: {run_path}")

# ----------------------------
# (B) Combined run: ALL NON-MSA (Dialects) queries pooled per topic
#     qid = ArTest_topic_id_{global_variant:02d}
# ----------------------------
pooled = defaultdict(list)

for dialect_name, tsv_path in dialect_files.items():
    if dialect_name in EXCLUDE_FROM_COMBINED:
        continue
    if not os.path.exists(tsv_path):
        print(f"(combined) Skipping missing file: {tsv_path}")
        continue

    per_file = read_queries_grouped_by_topic(tsv_path)
    for artest_id, queries in per_file.items():
        pooled[artest_id].extend(queries)

# Optional de-dup
if DEDUP_QUERIES_IN_COMBINED:
    for artest_id, queries in pooled.items():
        seen, deduped = set(), []
        for q in queries:
            if q not in seen:
                seen.add(q)
                deduped.append(q)
        pooled[artest_id] = deduped

def iter_combined_queries():
    # nicer sorting if numeric topic ids
    def sort_key(x):
        return int(x) if x.isdigit() else x

    for artest_id in sorted(pooled.keys(), key=sort_key):
        for idx, q in enumerate(pooled[artest_id], start=1):
            qid = f"{artest_id}_{idx:02d}"
            yield qid, q

combined_run_tag = "bm25"
combined_run_path = out_dir / f"bm25_ALL_NONMSA_top{k}.trec"

with open(combined_run_path, "w", encoding="utf-8") as out:
    for qid, query in iter_combined_queries():
        hits = searcher.search(query, k=k)
        for rank, hit in enumerate(hits, start=1):
            out.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {combined_run_tag}\n")


