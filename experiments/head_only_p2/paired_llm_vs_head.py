#!/usr/bin/env python3
"""Paired participant-level comparison: LLM readout vs. the two head-only controls (Protocol 2).

Read-only: uses the locked R1a fold outputs (through aggregate_head_only_p2.load_llm_folds) and
results/E16_head_only_protocol2.json. Writes results/E16_paired_llm_vs_head.json.
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate_head_only_p2 as A  # noqa: E402

RESAMPLES, SEED = 10000, 20260725
out = HERE / "results" / "E16_paired_llm_vs_head.json"
if out.exists():
    sys.exit(f"refusing to overwrite {out}")
llm = A.aggregate(A.load_llm_folds())
L = {r["subject_id"]: r for r in llm["subjects"]}
heads = json.loads((HERE / "results" / "E16_head_only_protocol2.json").read_text())["heads"]
U = sorted(L); Y = np.array([L[u]["label"] for u in U]); S = np.array([L[u]["mean_score"] for u in U])
bacc = lambda y, s: balanced_accuracy_score(y, (s >= 0.5).astype(int))
result = {"unit": "participant", "n": len(U), "resamples": RESAMPLES, "seed": SEED, "threshold": 0.5,
          "llm_readout": {"balanced_accuracy": bacc(Y, S), "auc": roc_auc_score(Y, S)}, "llm_minus_head": {}}
rng = np.random.default_rng(SEED); n = len(U)
idx = [j for j in (rng.integers(0, n, n) for _ in range(RESAMPLES)) if 0 < Y[j].sum() < n]
for name, h in heads.items():
    T = np.array([{r["subject_id"]: r["mean_score"] for r in h["subjects"]}[u] for u in U])
    db = [bacc(Y[j], S[j]) - bacc(Y[j], T[j]) for j in idx]; da = [roc_auc_score(Y[j], S[j]) - roc_auc_score(Y[j], T[j]) for j in idx]
    ls, hs = (S >= 0.5).astype(int), (T >= 0.5).astype(int)
    result["llm_minus_head"][name] = {
        "delta_bacc": bacc(Y, S) - bacc(Y, T), "delta_bacc_95ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
        "delta_auc": roc_auc_score(Y, S) - roc_auc_score(Y, T), "delta_auc_95ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
        "decisions_differ": int((ls != hs).sum()), "llm_right_head_wrong": int(((ls == Y) & (hs != Y)).sum()),
        "head_right_llm_wrong": int(((ls != Y) & (hs == Y)).sum()), "score_correlation": float(np.corrcoef(S, T)[0, 1])}
out.write_text(json.dumps(result, indent=2) + "\n")
print(f"LLM readout: BAcc {100*result['llm_readout']['balanced_accuracy']:.1f}%  AUC {result['llm_readout']['auc']:.3f}")
for name, r in result["llm_minus_head"].items():
    print(f"LLM minus {name:<13}: dBAcc {100*r['delta_bacc']:+.1f} pp [{100*r['delta_bacc_95ci'][0]:+.1f}, {100*r['delta_bacc_95ci'][1]:+.1f}] | dAUC {r['delta_auc']:+.3f} [{r['delta_auc_95ci'][0]:+.3f}, {r['delta_auc_95ci'][1]:+.3f}] | "
          f"decisions differ {r['decisions_differ']}/95 (LLM right & head wrong {r['llm_right_head_wrong']}, head right & LLM wrong {r['head_right_llm_wrong']}) | r = {r['score_correlation']:.3f}")
