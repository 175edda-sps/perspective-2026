#!/usr/bin/env python3
import os
import math
import csv
import random
import glob
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Tuple, Set
import pytrec_eval
# ----------------------------
# CONFIG
# ----------------------------
RUN_DIR = "runs_dialect"
OUT_RESULTS = "results_allruns.csv"
QRELS_PATH = "ArTest_judgments.txt"





# ----------------------------
# Settings
# ----------------------------
# One evaluation cutoff
K_EVAL = 10

# Variant subsampling
K_VARIANTS = 3
REPEATS = 50
SEED = 42

# Which files to evaluate
INCLUDE_GLOBS = [
    "*.trec",
    # or be stricter:
    # "bm25_*.trec",
]

# If True: require every run to have exactly the same topics
ASSERT_SAME_TOPICS = True


# Helpers
# ----------------------------
def base_topic(tid: str) -> str:
    """tid like '48_01' -> '48' (also handles 'prefix::48_01')"""
    return tid.split("::")[-1].split("_")[0]


def read_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Bad qrels line {line_no}: {line}")
            topic, _iter, docid, rel_s = parts[:4]
            qrels[topic][docid] = int(rel_s)
    return qrels


def read_run_grouped_by_variant(path: str) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, List[str]]]:
    """
    Returns:
      variant_docs: variant_tid -> [(docid, score), ...] after max-score merge, sorted desc by score
      base_to_variants: base_topic -> sorted list of variant tids present in this run
    """
    variant_doc2score = defaultdict(dict)
    base_to_variants = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Bad run line {line_no}: {line}")

            tid, _q0, docid, _rank, score_s, _tag = parts[:6]
            score = float(score_s)

            b = base_topic(tid)
            base_to_variants[b].add(tid)

            prev = variant_doc2score[tid].get(docid)
            if prev is None or score > prev:
                variant_doc2score[tid][docid] = score

    variant_docs = {}
    for tid, doc2score in variant_doc2score.items():
        ranked = sorted(doc2score.items(), key=lambda x: (-x[1], x[0]))
        variant_docs[tid] = ranked

    base_to_variants_out = {b: sorted(list(vs)) for b, vs in base_to_variants.items()}
    return variant_docs, base_to_variants_out


def choose_k_variants(variants: List[str], k: int, rng: random.Random) -> List[str]:
    if len(variants) <= k:
        return list(variants)
    return rng.sample(variants, k)


def merge_selected_variants(
    variant_docs: Dict[str, List[Tuple[str, float]]],
    selected_variants_by_base: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Merge selected variants into one ranked list per base topic using max score per doc."""
    run_base = {}
    for b, tids in selected_variants_by_base.items():
        doc2score = {}
        for tid in tids:
            for docid, score in variant_docs.get(tid, []):
                prev = doc2score.get(docid)
                if prev is None or score > prev:
                    doc2score[docid] = score
        ranked = sorted(doc2score.items(), key=lambda x: (-x[1], x[0]))
        run_base[b] = [d for d, _ in ranked]
    return run_base


# ----------------------------
# Metrics (pytrec_eval)
# ----------------------------
def eval_run_base_pytrec(
    run_base: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    topics: List[str],
    k_eval: int,
) -> Dict[str, float]:
    
    measures = {
        "map",
        "recip_rank",
        f"P_{k_eval}",
        f"recall_{k_eval}",
        f"ndcg_cut_{k_eval}",
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)

    # ranked list -> {docid: score} with decreasing scores to preserve ordering
    run = {}
    for t in topics:
        ranked = run_base.get(t, [])
        run[t] = {docid: float(len(ranked) - i) for i, docid in enumerate(ranked)}

    res = evaluator.evaluate(run)

    def avg(key: str) -> float:
        return sum(res.get(t, {}).get(key, 0.0) for t in topics) / len(topics)

    return {
        "MAP": avg("map"),
        f"P@{k_eval}": avg(f"P_{k_eval}"),
        f"NDCG@{k_eval}": avg(f"ndcg_cut_{k_eval}"),
        "MRR": avg("recip_rank"),
        f"Recall@{k_eval}": avg(f"recall_{k_eval}"),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    rng = random.Random(SEED)

    qrels = read_qrels(QRELS_PATH)
    qrels_topics = set(qrels.keys())

    # Collect run paths
    run_paths = []
    for pat in INCLUDE_GLOBS:
        run_paths.extend(glob.glob(os.path.join(RUN_DIR, pat)))
    run_paths = sorted(list(dict.fromkeys(run_paths)))
    if not run_paths:
        raise ValueError(f"No run files found in {RUN_DIR} matching: {INCLUDE_GLOBS}")

    # Read all runs
    per_run_variant_docs = {}
    per_run_base_variants = {}
    per_run_topics = {}

    for path in run_paths:
        vdocs, b2v = read_run_grouped_by_variant(path)
        per_run_variant_docs[path] = vdocs
        per_run_base_variants[path] = b2v
        per_run_topics[path] = set(b2v.keys()) & qrels_topics

    # Compute common topics across all runs (and qrels)
    common_topics = set.intersection(*per_run_topics.values()) if per_run_topics else set()
    common_topics = sorted(list(common_topics & qrels_topics))

    if not common_topics:
        raise ValueError("No common topics across all runs (after intersecting with qrels).")

    # Optional strict check: every run has exactly these topics
    if ASSERT_SAME_TOPICS:
        for path in run_paths:
            missing = set(common_topics) - per_run_topics[path]
            extra = per_run_topics[path] - set(common_topics)
            if missing or extra:
                raise ValueError(
                    f"Topic mismatch in {os.path.basename(path)}:\n"
                    f"  missing: {sorted(missing)}\n"
                    f"  extra:   {sorted(extra)}"
                )

    # Variant sufficiency check
    for path in run_paths:
        for t in common_topics:
            vcount = len(per_run_base_variants[path][t])
            if vcount < K_VARIANTS:
                raise ValueError(
                    f"{os.path.basename(path)} topic {t} has {vcount} variants (<{K_VARIANTS})."
                )

    # Determine repeats
    repeats_effective = 1
    if any(any(len(per_run_base_variants[p][t]) > K_VARIANTS for t in common_topics) for p in run_paths):
        repeats_effective = REPEATS

    metric_cols = ["MAP", f"P@{K_EVAL}", f"NDCG@{K_EVAL}", "MRR", f"Recall@{K_EVAL}"]
    metrics_by_run = {p: defaultdict(list) for p in run_paths}

    for _rep in range(repeats_effective):
        for path in run_paths:
            selected_by_base = {
                t: choose_k_variants(per_run_base_variants[path][t], K_VARIANTS, rng)
                for t in common_topics
            }
            run_base = merge_selected_variants(per_run_variant_docs[path], selected_by_base)

            scores = eval_run_base_pytrec(run_base, qrels, common_topics, K_EVAL)
            for m, v in scores.items():
                metrics_by_run[path][m].append(v)

    # Write results
    with open(OUT_RESULTS, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["run", "topics_used", "K_variants", "repeats_used"]
        for m in metric_cols:
            fieldnames += [f"{m}_mean", f"{m}_std"]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for path in run_paths:
            row = {
                "run": os.path.basename(path),
                "topics_used": len(common_topics),
                "K_variants": K_VARIANTS,
                "repeats_used": repeats_effective,
            }
            for m in metric_cols:
                vals = metrics_by_run[path][m]
                row[f"{m}_mean"] = mean(vals)
                row[f"{m}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
            w.writerow(row)

    print(f"[OK] Evaluated {len(run_paths)} runs on {len(common_topics)} common topics.")
    print(f"[OK] Wrote {OUT_RESULTS} | K_VARIANTS={K_VARIANTS} | K_EVAL={K_EVAL} | repeats={repeats_effective}")


if __name__ == "__main__":
    main()




