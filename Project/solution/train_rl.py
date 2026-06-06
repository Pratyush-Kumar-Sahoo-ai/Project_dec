"""Train the linear placement policy (solution/rl_strategy.py) by the
Cross-Entropy Method (CEM).

CEM is a derivative-free policy-search algorithm: it samples weight vectors from
a Gaussian, evaluates each by *running the simulator* on the train split, keeps
the best ("elite") fraction, and refits the Gaussian to the elites — repeating
until the mean weight vector converges. Because the simulator is deterministic
given a weight vector, every evaluation is noise-free, so CEM optimises the
*true* objective (reshuffles/retrieval) with no reward shaping.

Usage:
    python -m solution.train_rl                  # train + save rl_weights.json
    python -m solution.train_rl --quick          # smaller/faster search

Output: solution/rl_weights.json  (consumed automatically by RLStrategy).
"""

import argparse
import json
import math
import os
import random
import statistics
import time

from src.models import Event
from src.yard_state import YardState
from src.simulator import Simulator
from solution.rl_strategy import RLStrategy, FEATURES, _GREEDY_WEIGHTS
from multiprocessing import Pool, cpu_count

_HERE = os.path.dirname(__file__)
_WEIGHTS_PATH = os.path.join(_HERE, "rl_weights.json")


def _load(split):
    layout = json.load(open(os.path.join("data", "yard_layout.json")))
    init = json.load(open(os.path.join("data", split, "initial_state.json")))
    rows = [json.loads(l) for l in open(os.path.join("data", split, "events.jsonl"))]
    events = [Event(**{k: v for k, v in e.items()
                       if k in Event.__dataclass_fields__}) for e in rows]
    return layout, init, events


def evaluate(weights, layout, init, events):
    """Run the policy with the given weights; return reshuffles/retrieval."""
    yard = YardState(layout)
    yard.load_initial_state(init)
    strat = RLStrategy(weights=weights)
    strat.initialize(layout, init)
    stats = Simulator(yard, strat).run(events)
    # Penalise any constraint violation heavily (should never happen).
    return stats.reshuffles_per_retrieval + 10.0 * stats.hard_constraint_violations

def _evaluate_worker(args):
    weights, layout, init, events = args
    return evaluate(weights, layout, init, events)

def cem(layout, init, events, pop=16, elite=4, iters=8, init_sigma=1.5, seed=0, n_workers = None):
    if n_workers is None:
        n_workers = min(pop, max(1, cpu_count() - 1))
    rng = random.Random(seed)
    dim = len(FEATURES)
    mu = list(_GREEDY_WEIGHTS)                 # warm-start at the greedy policy
    sigma = [init_sigma] * dim
    best_w, best_score = list(mu), evaluate(mu, layout, init, events)
    print(f"  warm-start (greedy) score = {best_score:.4f}")
    print(f"  using {n_workers} parallel workers")
    for it in range(iters):
        t0 = time.time()
        candidates = [list(best_w)] # keep best candidate so far
        for _ in range(pop-1):
            w = [mu[i] + sigma[i] * rng.gauss(0, 1) for i in range(dim)]
            candidates.append(w)
        # Evaluate candidates in parallel
        work_items = [(w, layout, init, events) for w in candidates]
        with Pool(n_workers) as pool:
            scores = pool.map(_evaluate_worker, work_items)
        samples = list(zip(scores, candidates))
        samples.sort(key=lambda x: x[0])
        elites = [w for _, w in samples[:elite]]
        mu = [statistics.mean(e[i] for e in elites) for i in range(dim)]
        sigma_floor = max(0.01, 0.3*(1 - it/iters))
        sigma = [statistics.pstdev([e[i] for e in elites]) + sigma_floor for i in range(dim)]
        if samples[0][0] < best_score:
            best_score, best_w = samples[0]
        print(f"  iter {it+1}/{iters}: best={samples[0][0]:.4f} "
              f"mean_elite={statistics.mean(s for s, _ in samples[:elite]):.4f} "
              f"overall_best={best_score:.4f} ({time.time()-t0:.0f}s)")
    return best_w, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-events", type=int, default=6000,
                    help="number of leading train events used per CEM evaluation")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="number of parallel workers to use for evaluation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.train_events, args.pop, args.iters = 4000, 12, 5

    layout, init, train_events = _load("train")
    sub = train_events[:args.train_events]
    print(f"CEM policy search on first {len(sub)} train events "
          f"(pop={args.pop}, elite={args.elite}, iters={args.iters})")

    best_w, best_score = cem(layout, init, sub, pop=args.pop, elite=args.elite,
                             iters=args.iters, seed=args.seed, n_workers=args.workers)

    # Full-split validation of the learned policy.
    full_train = evaluate(best_w, layout, init, train_events)
    layout_te, init_te, test_events = _load("test")
    full_test = evaluate(best_w, layout_te, init_te, test_events)
    print(f"\nLearned weights (subsample score {best_score:.4f}):")
    for f, w in zip(FEATURES, best_w):
        print(f"    {f:16s} {w:+.3f}")
    print(f"Full train reshuffles/retrieval = {full_train:.4f}")
    print(f"Full test  reshuffles/retrieval = {full_test:.4f}")

    with open(_WEIGHTS_PATH, "w") as fh:
        json.dump({
            "weights": dict(zip(FEATURES, best_w)),
            "train_subsample_score": best_score,
            "full_train_ratio": full_train,
            "full_test_ratio": full_test,
            "config": vars(args),
        }, fh, indent=2)
    print(f"Saved -> {_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
