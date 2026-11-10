"""Question-level diagnostic tables for the frozen retrieval experiment."""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from pathlib import Path
from collect import read_questions

DATA = Path(__file__).resolve().parents[1] / "data"

NAMES = ("matplotlib", "blender", "qgis")


def load_questions():
    rows = []
    for name in NAMES:
        questions = read_questions(name)
        qwen = {r["id"]: r for r in json.loads((DATA / f"{name}-qwen-rankings.json").read_text())}
        original = {r["id"]: r for r in json.loads((DATA / f"{name}-rankings.json").read_text())}
        ids = {r["id"] for r in questions}
        if len(ids) != len(questions) or ids != qwen.keys() or ids != original.keys():
            raise ValueError(f"Question IDs do not match in {name}")
        for q in questions:
            qid = q["id"]
            gold = set(q["targets"])
            hits = {}
            for mode, prediction in qwen[qid]["methods"].items():
                hit = int(bool(gold.intersection(prediction["top10"][:5])))
                if hit != prediction["hit5"]:
                    raise ValueError(f"Hit@5 mismatch in {name}, question {qid}, mode {mode}")
                hits[mode] = hit
            rows.append({"domain":name, "id":qid, "group":q["group"], "raw_image":bool(q["images"]),
                         "usable_image":bool(original[qid]["query_image"]), **hits})
    return pd.DataFrame(rows)


def misses(frame, mode, image_field):
    images = frame.loc[frame[image_field], mode]
    no_images = frame.loc[~frame[image_field], mode]
    return (int((1-images).sum()), len(images), int((1-no_images).sum()), len(no_images))


def standardized_gap(frame, mode="A", image_field="raw_image", controls=("domain", "group"), draws=20000):
    """Fixed group weights; independent resampling within exposure cells, randomization under the null."""
    rng = np.random.default_rng(42)
    gaps = np.zeros(draws)
    null = np.zeros(draws)
    observed = 0.0
    groups = frame.groupby(list(controls), sort=True) if controls else [(None, frame)]
    for _, group in groups:
        a, n_img, c, n_other = misses(group, mode, image_field)
        if min(n_img, n_other) == 0:
            raise ValueError("Both exposure groups are required in each stratum")
        share = len(group) / len(frame)
        observed += share * (a/n_img - c/n_other)
        gaps += share * (rng.binomial(n_img, a/n_img, draws)/n_img - rng.binomial(n_other, c/n_other, draws)/n_other)
        shuffled = rng.hypergeometric(a+c, len(group)-a-c, n_img, draws)
        null += share * (shuffled/n_img - (a+c-shuffled)/n_other)
    lo, hi = np.quantile(gaps, [.025,.975])
    p = (1+np.count_nonzero(np.abs(null) >= abs(observed)-1e-12))/(draws+1)
    return {"Δ, п.п.":100*observed, "95% low":100*lo, "95% high":100*hi, "p (перестановки)":p}


def paired_comparison(frame, candidate, reference, draws=20000):
    delta = (frame[candidate]-frame[reference]).to_numpy(dtype=int)
    rng = np.random.default_rng(42)
    win, loss = int((delta>0).sum()), int((delta<0).sum())
    simulated = rng.multinomial(len(delta), [win/len(delta),loss/len(delta),1-(win+loss)/len(delta)],size=draws)
    lo, hi = np.quantile(100*(simulated[:,0]-simulated[:,1])/len(delta), [.025,.975])
    return {"n":len(delta), "Вошла":win, "Выпала":loss, "Δ, п.п.":100*delta.mean(),
            "95% low":lo, "95% high":hi,
            "p (McNemar exact)":binomtest(win,win+loss,.5).pvalue if win+loss else 1.0}
