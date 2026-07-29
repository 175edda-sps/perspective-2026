import os
import csv
from pathlib import Path
from collections import defaultdict

from pyserini.encode import AutoQueryEncoder
from pyserini.search.faiss import FaissSearcher

# ----------------------------
# Config
# ----------------------------
index_path = "faiss/ar-e5-large"
encoder_name = "intfloat/multilingual-e5-large"
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

RUN_TAG = "dense_e5large"


out_dir = Path("runs_dialect")
out_dir.mkdir(parents=True, exist_ok=True)

# Combined settings
EXCLUDE_FROM_COMBINED = {"MSA"}
DEDUP_QUERIES_IN_COMBINED = True

# ----------------------------
# Init encoder + searcher
# ----------------------------
encoder = AutoQueryEncoder(encoder_name)
searcher = FaissSearcher(index_path, query_encoder=encoder)

# ----------------------------
# Helpers
# ----------------------------
def _sorted_query_cols(fieldnames):
    def qnum(c):
        try:
            return int(c.split("_")[1])
        except Exception:
            return 10**9

    return sorted(
        [c for c in (fieldnames or []) if c.lower().startswith("query_")],
        key=qnum
    )

def iter_queries_from_tsv(tsv_path):
    """Per-file queries: qid = ArTest_topic_id_{query_idx:02d}"""
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return

        query_cols = _sorted_query_cols(reader.fieldnames)

        for row in reader:
            topic_id = (row.get("ArTest_topic_id") or "").strip()
            if not topic_id:
                continue

            for col in query_cols:
                q = (row.get(col) or "").strip()
                if not q:
                    continue

                try:
                    idx = int(col.split("_")[1])
                except Exception:
                    continue

                yield f"{topic_id}_{idx:02d}", q

def write_run(run_path, qid_query_iter):
    with open(run_path, "w", encoding="utf-8") as out:
        for qid, query in qid_query_iter:
            hits = searcher.search(query, k)
            for rank, hit in enumerate(hits, start=1):
                out.write(
                    f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {RUN_TAG}\n"
                )

def read_queries_grouped_by_topic(tsv_path):
    topic2queries = defaultdict(list)

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return topic2queries

        query_cols = _sorted_query_cols(reader.fieldnames)

        for row in reader:
            topic_id = (row.get("ArTest_topic_id") or "").strip()
            if not topic_id:
                continue

            for col in query_cols:
                q = (row.get(col) or "").strip()
                if q:
                    topic2queries[topic_id].append(q)

    return topic2queries

# ----------------------------
# (A) Per-dialect dense runs
# ----------------------------
for dialect, tsv_path in dialect_files.items():
    if not os.path.exists(tsv_path):
        print(f"Skipping missing file: {tsv_path}")
        continue

    run_path = out_dir / f"dense_{dialect}_top{k}.trec"
    write_run(run_path, iter_queries_from_tsv(tsv_path))
    print(f"Wrote: {run_path}")

# ----------------------------
# (B) Combined NON-MSA dense run
# ----------------------------
pooled = defaultdict(list)

for dialect, tsv_path in dialect_files.items():
    if dialect in EXCLUDE_FROM_COMBINED:
        continue
    if not os.path.exists(tsv_path):
        continue

    per_file = read_queries_grouped_by_topic(tsv_path)
    for topic_id, queries in per_file.items():
        pooled[topic_id].extend(queries)

if DEDUP_QUERIES_IN_COMBINED:
    for topic_id, queries in pooled.items():
        seen, deduped = set(), []
        for q in queries:
            if q not in seen:
                seen.add(q)
                deduped.append(q)
        pooled[topic_id] = deduped

def iter_combined_queries():
    for topic_id in sorted(pooled, key=lambda x: int(x) if x.isdigit() else x):
        for i, q in enumerate(pooled[topic_id], start=1):
            yield f"{topic_id}_{i:02d}", q

combined_run_path = out_dir / f"dense_ALL_NONMSA_top{k}.trec"
write_run(combined_run_path, iter_combined_queries())

