import os
import glob
from collections import defaultdict
import pandas as pd
import rbo  # pip install rbo

# ----------------------------
# Config
# ----------------------------
RUN_DIR = "runs_dialect"   # contains *_MSA_top100.trec and *_ALL_NONMSA_top100.trec
OUT_DIR = "eval_rbo"
os.makedirs(OUT_DIR, exist_ok=True)

TOP_K = 10
P = 0.9

OUT_PER_TOPIC = os.path.join(OUT_DIR, f"rbo_per_topic_msa_nonmsa_all_p{P}_k{TOP_K}.tsv")
OUT_SUMMARY  = os.path.join(OUT_DIR, f"rbo_summary_msa_nonmsa_all_p{P}_k{TOP_K}.tsv")

SYSTEMS = ["bm25_rerank_e5", "dense", "bm25"]  # matching priority


# ----------------------------
# Helpers
# ----------------------------
def system_from_filename(fname: str) -> str:
    base = os.path.basename(fname).lower()
    for s in SYSTEMS:
        if base.startswith(s):
            return s
    if base.startswith("bm25"):
        return "bm25"
    if base.startswith("dense"):
        return "dense"
    return "unknown"

def is_msa_file(fname: str) -> bool:
    return os.path.basename(fname).lower().endswith("_msa_top100.trec")

def is_nonmsa_file(fname: str) -> bool:
    return os.path.basename(fname).lower().endswith("_all_nonmsa_top100.trec")

def base_topic(qid: str) -> str:
    # qid like "48_01" -> "48"
    return qid.split("_")[0]

def read_run_as_lists(path: str, top_k: int):
    """
    Read TREC run and return:
      lists: dict[qid] -> [docid1, docid2, ...] (top_k by rank order)
    """
    tmp = defaultdict(list)  # qid -> list of (rank, docid)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, *_ = parts
            try:
                rank = int(rank)
            except Exception:
                continue
            tmp[qid].append((rank, docid))

    lists = {}
    for qid, rank_docs in tmp.items():
        rank_docs.sort(key=lambda x: x[0])
        lists[qid] = [d for _, d in rank_docs][:top_k]
    return lists

def rbo_score(a, b, p=0.9):
    return rbo.RankingSimilarity(a, b).rbo_ext(p=p)

def avg_rbo_within(list_of_lists, p=0.9):
    """
    Average pairwise RBO among a set of ranked lists.
    Returns None if <2 lists or no valid pairs.
    """
    n = len(list_of_lists)
    if n < 2:
        return None
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            A, B = list_of_lists[i], list_of_lists[j]
            if not A or not B:
                continue
            scores.append(rbo_score(A, B, p=p))
    return (sum(scores) / len(scores)) if scores else None

def group_lists_by_topic(qid_to_list: dict):
    """
    Convert qid->docs into topic->list_of_ranked_lists
    """
    out = defaultdict(list)
    for qid, docs in qid_to_list.items():
        if docs:
            out[base_topic(qid)].append(docs)
    return out


# ----------------------------
# Locate runs
# ----------------------------
run_files = sorted(glob.glob(os.path.join(RUN_DIR, "*.trec")))
if not run_files:
    raise RuntimeError(f"No .trec files found in {RUN_DIR}")

sys_runs = defaultdict(dict)
for rf in run_files:
    sys = system_from_filename(rf)
    if sys == "unknown":
        continue
    if is_msa_file(rf):
        sys_runs[sys]["msa"] = rf
    elif is_nonmsa_file(rf):
        sys_runs[sys]["nonmsa"] = rf

print("Detected systems and files:")
for s in sorted(sys_runs.keys()):
    d = sys_runs[s]
    print(" -", s, "msa:", os.path.basename(d.get("msa","")), "nonmsa:", os.path.basename(d.get("nonmsa","")))


# ----------------------------
# Compute PER-TOPIC RBO per system
# ----------------------------
rows = []

for sys in sorted(sys_runs.keys()):
    msa_path = sys_runs[sys].get("msa")
    nonmsa_path = sys_runs[sys].get("nonmsa")
    if not msa_path or not nonmsa_path:
        continue

    msa_qid_lists = read_run_as_lists(msa_path, top_k=TOP_K)
    nonmsa_qid_lists = read_run_as_lists(nonmsa_path, top_k=TOP_K)

    msa_by_topic = group_lists_by_topic(msa_qid_lists)
    nonmsa_by_topic = group_lists_by_topic(nonmsa_qid_lists)

    topics = sorted(set(msa_by_topic.keys()) & set(nonmsa_by_topic.keys()))
    if not topics:
        print(f"[WARN] {sys}: no overlapping topics between MSA and NONMSA runs.")
        continue

    for t in topics:
        msa_lists = msa_by_topic.get(t, [])
        nonmsa_lists = nonmsa_by_topic.get(t, [])
        all_lists = msa_lists + nonmsa_lists

        rbo_msa_t = avg_rbo_within(msa_lists, p=P)
        rbo_nonmsa_t = avg_rbo_within(nonmsa_lists, p=P)
        rbo_all_t = avg_rbo_within(all_lists, p=P)

        rows.append({
            "system": sys,
            "topic": t,
            "p": P,
            "k": TOP_K,
            "n_msa_lists_topic": len(msa_lists),
            "n_nonmsa_lists_topic": len(nonmsa_lists),
            "n_all_lists_topic": len(all_lists),
            "rbo_msa_within_topic": rbo_msa_t,
            "rbo_nonmsa_within_topic": rbo_nonmsa_t,
            "rbo_all_within_topic": rbo_all_t,
        })

per_topic_df = pd.DataFrame(rows).sort_values(["system", "topic"])
per_topic_df.to_csv(OUT_PER_TOPIC, sep="\t", index=False)
print("\n Wrote per-topic:", OUT_PER_TOPIC)

# ----------------------------
# Summary per system (mean/median over topics)
# ----------------------------
def safe_mean(x):
    x = [v for v in x if pd.notna(v)]
    return float(sum(x)/len(x)) if x else float("nan")

def safe_median(x):
    x = sorted([v for v in x if pd.notna(v)])
    if not x:
        return float("nan")
    mid = len(x)//2
    return float(x[mid]) if len(x)%2==1 else float((x[mid-1]+x[mid])/2)

summary_rows = []
for sys, sdf in per_topic_df.groupby("system"):
    summary_rows.append({
        "system": sys,
        "p": P,
        "k": TOP_K,
        "n_topics": int(sdf["topic"].nunique()),
        "msa_mean": safe_mean(sdf["rbo_msa_within_topic"]),
        "nonmsa_mean": safe_mean(sdf["rbo_nonmsa_within_topic"]),
        "all_mean": safe_mean(sdf["rbo_all_within_topic"]),
        "msa_median": safe_median(sdf["rbo_msa_within_topic"]),
        "nonmsa_median": safe_median(sdf["rbo_nonmsa_within_topic"]),
        "all_median": safe_median(sdf["rbo_all_within_topic"]),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("system")
summary_df.to_csv(OUT_SUMMARY, sep="\t", index=False)
print(" Wrote summary:", OUT_SUMMARY)
