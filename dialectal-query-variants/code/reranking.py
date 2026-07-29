
#!/usr/bin/env python3
import os
import glob
import json
import csv
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel
from pyserini.search.lucene import LuceneSearcher


# ============================
# CONFIG (edit if needed)
# ============================
index_path = "index_dialectal_query_variants"                 
run_dir = "runs_dialect"                       # where bm25_*.trec are
out_dir = "runs_dialect_rerank"                # write reranked runs here

# Provide the TSVs used to build qid->query for each BM25 run you already have
dialect_files = {
    "MSA": "query_variants/MSA.tsv",
    "Algerian": "query_variants/Algerian.tsv",
    "Egyptian": "query_variants/Egyptian.tsv",
    "Palestinian": "query_variants/Palestinian.tsv",
    "Qatari": "query_variants/Qatari.tsv",
    "Saudi": "query_variants/Saudi.tsv",
    "Syrian": "query_variants/Syrian.tsv",
}


combined_run_filename = "bm25_ALL_NONMSA_top100.trec"
EXCLUDE_FROM_COMBINED = {"MSA"}
DEDUP_QUERIES_IN_COMBINED = True

# Reranking params
topk_rerank = 100                               # rerank top N docs from BM25
encoder_name = "intfloat/multilingual-e5-large"
run_tag = "bm25_rerank_e5"
batch_size_docs = 128
# ============================


# ----------------------------
# Query builders (must match your qid format)
# ----------------------------
def _sorted_query_cols(fieldnames):
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
    """qid = ArTest_topic_id_{variant_idx:02d} where variant_idx comes from query_#."""
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
    """Returns { ArTest_topic_id: [q1, q2, ...] } preserving order."""
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


def build_qmap_for_dialect(dialect_name: str) -> dict:
    """Build qid->query for a single dialect TSV."""
    tsv_path = dialect_files.get(dialect_name)
    if not tsv_path or not os.path.exists(tsv_path):
        return {}
    qmap = {}
    for qid, q in iter_queries_from_tsv(tsv_path):
        qmap[qid] = q
    return qmap


def build_qmap_for_combined_nonmsa() -> dict:
    """Build qid->query for pooled non-MSA combined run."""
    pooled = defaultdict(list)

    for dialect_name, tsv_path in dialect_files.items():
        if dialect_name in EXCLUDE_FROM_COMBINED:
            continue
        if not os.path.exists(tsv_path):
            continue

        per_file = read_queries_grouped_by_topic(tsv_path)
        for artest_id, queries in per_file.items():
            pooled[artest_id].extend(queries)

    if DEDUP_QUERIES_IN_COMBINED:
        for artest_id, queries in pooled.items():
            seen, deduped = set(), []
            for q in queries:
                if q not in seen:
                    seen.add(q)
                    deduped.append(q)
            pooled[artest_id] = deduped

    def sort_key(x):
        return int(x) if x.isdigit() else x

    qmap = {}
    for artest_id in sorted(pooled.keys(), key=sort_key):
        for idx, q in enumerate(pooled[artest_id], start=1):
            qid = f"{artest_id}_{idx:02d}"
            qmap[qid] = q

    return qmap


# ----------------------------
# Run + doc fetch
# ----------------------------
def load_trec_run(path: str, topk: int = 100):
    """Load a TREC run file, keeping topk docids per query in input order."""
    run = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            qid, _, docid, rank, score, tag = parts
            if len(run[qid]) < topk:
                run[qid].append(docid)
    return run


def get_doc_text(searcher: LuceneSearcher, docid: str) -> str:
    """Retrieve document text from Lucene."""
    doc = searcher.doc(docid)
    if doc is None:
        return ""

    raw = ""
    if hasattr(doc, "raw") and callable(doc.raw):
        try:
            raw = doc.raw() or ""
        except Exception:
            raw = ""

    if (not raw.strip()) and hasattr(doc, "contents") and callable(doc.contents):
        try:
            raw = doc.contents() or ""
        except Exception:
            raw = ""

    raw = (raw or "").strip()
    if not raw:
        return ""

    if raw.startswith("{") and raw.endswith("}"):
        try:
            obj = json.loads(raw)
            txt = obj.get("contents") or obj.get("text") or ""
            return (txt or "").strip()
        except Exception:
            return raw

    return raw


# ----------------------------
# Dense scoring (bi-encoder)
# ----------------------------
@torch.no_grad()
def encode_texts(tokenizer, model, texts, device, batch_size=64):
    """Mean-pool + L2 normalize embeddings."""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = model(**tok)

        attn = tok["attention_mask"].unsqueeze(-1)
        x = (out.last_hidden_state * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        embs.append(x.cpu())
    return torch.cat(embs, dim=0)


def infer_dialect_from_run_filename(fname: str):
    """
    Expects:
      bm25_{Dialect}_top100.trec
    and optionally:
      bm25_ALL_NONMSA_top100.trec
    """
    base = os.path.basename(fname)
    if combined_run_filename and base == os.path.basename(combined_run_filename):
        return "__COMBINED__"

    if base.startswith("bm25_") and base.endswith(".trec") and "_top" in base:
        mid = base[len("bm25_"):]
        return mid.split("_top", 1)[0]
    return None


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Model
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(device)
    model.eval()

    # Lucene fetcher for doc text
    searcher = LuceneSearcher(index_path)
    searcher.set_language("ar")

    # Collect BM25 runs (already ready)
    runfiles = sorted(glob.glob(os.path.join(run_dir, "bm25_*_top*.trec")))
    if not runfiles:
        raise FileNotFoundError(f"No bm25 runfiles found in {run_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Cache qmaps
    qmap_cache = {}

    for bm25_run in runfiles:
        dialect = infer_dialect_from_run_filename(bm25_run)
        if dialect is None:
            print(f"Skipping (unrecognized): {bm25_run}")
            continue

        # Build qid->query map that matches this run's qids
        if dialect == "__COMBINED__":
            if dialect not in qmap_cache:
                qmap_cache[dialect] = build_qmap_for_combined_nonmsa()
                print("Loaded combined non-MSA queries:", len(qmap_cache[dialect]))
            qmap = qmap_cache[dialect]
        else:
            if dialect not in qmap_cache:
                qmap_cache[dialect] = build_qmap_for_dialect(dialect)
                print(f"Loaded queries for {dialect}:", len(qmap_cache[dialect]))
            qmap = qmap_cache[dialect]

        run = load_trec_run(bm25_run, topk=topk_rerank)

        out_run = os.path.join(
            out_dir,
            os.path.basename(bm25_run).replace("bm25_", "bm25_rerank_e5_", 1)
        )

        skipped_no_query, reranked_q = 0, 0

        with open(out_run, "w", encoding="utf-8") as out:
            for qid, docids in run.items():
                query = qmap.get(qid)
                if not query:
                    skipped_no_query += 1
                    continue

                q_text = f"query: {query}"

                passages, kept_docids = [], []
                for docid in docids:
                    txt = get_doc_text(searcher, docid)
                    if not txt:
                        continue
                    passages.append(f"passage: {txt}")
                    kept_docids.append(docid)

                if not kept_docids:
                    continue

                q_emb = encode_texts(tokenizer, model, [q_text], device=device, batch_size=1)
                d_emb = encode_texts(tokenizer, model, passages, device=device, batch_size=batch_size_docs)

                scores = (d_emb @ q_emb.T).squeeze(1).tolist()
                ranked = sorted(zip(kept_docids, scores), key=lambda x: x[1], reverse=True)

                for rank, (docid, score) in enumerate(ranked, start=1):
                    out.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_tag}\n")

                reranked_q += 1

        print(f"Reranked: {bm25_run}")
        print(f"  -> {out_run}")
        print(f"  queries reranked: {reranked_q}, skipped (no query text): {skipped_no_query}")

    print("Done.")


if __name__ == "__main__":
    main()

