# Learned fusion of the free (no-API) witnesses: every moment where any of SAM breaks (A), in-place changes (P),
# evidence-track events (E) or count steps (K) fires gets a feature vector, and a logistic regression scores it.
# Evaluated leave-one-object-out on the development lists, so no object is scored by a model that saw its labels.
# Usage: python research\fusion.py clips\physion-labels.json clips\physion-tune.json clips\physion-tune2.json
import json, sys
import numpy as np
from paths import trace_path
from eval_evidence import load, score, TOL
from count_change import steps

R = 5  # frames within which witnesses describe the same moment


def moments(c, frames):
    """[(frame, kinds, features)] for one object."""
    K = steps(frames, c["n"], 0.4)
    E = sorted((e for e in c["E"] if not e["edge"] and e["length"] >= 10), key=lambda e: -e["strength"])
    src = {"A": [(f, {k}, 1.0) for k, f, _ in c["A"]],
           "P": [(f, set(), z) for f, z, _ in c["deep"][:5]],
           "E": [(e["frame"], {e["kind"]}, e["strength"]) for e in E[:5]],
           "K": [(f, {k}, d) for k, f, d in K[:5]]}
    rank = {s: {f: r for r, (f, _, _) in enumerate(v)} for s, v in src.items()}

    out = []
    for f in sorted({f for v in src.values() for f, _, _ in v}):
        if any(abs(f - g) < R and g != f for g, _, _ in out):
            continue
        x, kinds = [], set()
        for s, v in src.items():
            near = [(g, ks, val) for g, ks, val in v if abs(g - f) <= R]
            kinds |= set().union(*[ks for _, ks, _ in near]) if near else set()
            x += [1.0 if near else 0.0, max((val for _, _, val in near), default=0.0),
                  min((rank[s][g] for g, _, _ in near), default=5) / 5]
        cov = np.mean([bool(fr.get("out")) for fr in frames.values()])
        x += [cov, len(c["A_all"]) / 20]
        out.append((f, kinds or {"appear", "vanish"}, np.array(x)))
    return out


def fit(X, y, l2=1.0, steps_=2000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(steps_):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y))
    return lambda X2: 1 / (1 + np.exp(-np.c_[np.ones(len(X2)), (X2 - mu) / sd] @ w))


if __name__ == "__main__":
    clips = load([a for a in sys.argv[1:] if not a.startswith("--")])
    M = [moments(c, json.loads(trace_path(c["file"], c["name"]).read_text())["frames"]) for c in clips]
    lab = [[float(any(abs(fr - f) <= TOL for _, fr in c["truth"])) for f, _, _ in m] for c, m in zip(clips, M)]
    prob = []
    for i in range(len(clips)):  # leave one object out
        X = np.array([x for j, m in enumerate(M) if j != i for _, _, x in m])
        y = np.array([l for j, ls in enumerate(lab) if j != i for l in ls])
        prob.append(fit(X, y)(np.array([x for _, _, x in M[i]])) if M[i] else np.array([]))

    idx = {id(c): i for i, c in enumerate(clips)}
    print(f"{sum(map(len, M))} moments, {int(sum(map(sum, lab)))} near a labelled event")
    for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        ca, tot, fa = score(clips, lambda c: [(k, f) for (f, ks, _), p in zip(M[idx[id(c)]], prob[idx[id(c)]])
                                              if p >= t for k in ks])
        print(f"p >= {t:.2f}: caught {ca}/{tot} ({ca / tot:.0%}), false alarms {fa}")
