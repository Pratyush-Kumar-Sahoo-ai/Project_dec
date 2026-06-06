# Container Yard Placement — Design Document

**Deployed strategy:** `solution/rl_strategy.py` — learned linear policy (Cross-Entropy Method).
**Companion heuristic:** `solution/eri_strategy.py` — Capacity-Aware Expected Reshuffle Index (ERI).
**Papers:**
- **Tier A (heuristic):** H. Bisira & A. Salhi (2021), *Reshuffle minimisation to improve storage yard operations efficiency* — Expected Reshuffling Index (ERI), building on Kim & Hong (2006) and Galle et al. (2018), *The Stochastic Container Relocation Problem*.
- **Tier B (deployed RL):** Maglić et al. (2020), *Online Stacking Using Reinforcement Learning with Positional and Tactical Features* — feature-based online stacking policy; Lim et al. (2025), *Reinforcement learning approach for outbound container stacking in container terminals* (Computers & Industrial Engineering) — Monte Carlo Q-learning with free-space-aware reward design (we adopt the feature/reward philosophy but train via Cross-Entropy Method for low-resource, noise-free optimisation).

**Results (test, days 21–40), 0 hard-constraint violations:**

| Strategy | reshuffles/retrieval | quantitative |
|---|---|---|
| Greedy baseline (lowest-stack) | 0.7687 | 11.3 / 40 |
| ERI heuristic (capacity-dominant, ≈greedy) | 0.7687 | 11.3 / 40 |
| **Learned policy (CEM), deployed** | **0.7516** | **12.1 / 40** |

The learned policy beats greedy **out-of-sample** (and by more on test than on train — it is not overfit). The gain is modest because, as the analysis below establishes, the achievable online ceiling on this instance is close to greedy: leave-order is only weakly predictable (ρ≈0.55) and a perfect-information upper bound is 0.48.

---

## 1. Approach: a heuristic and a learned policy

We implement two complementary methods and deploy the stronger one.

1. **ERI heuristic** (`eri_strategy.py`, §3) — the canonical Expected Reshuffle Index: place each arriving container on the stack minimising the *expected number of containers it will bury*. Transparent, paper-faithful, no training.
2. **Learned linear policy** (`rl_strategy.py`, §5, **deployed**) — a policy `place at argmin_column w·features(c, column)` whose weights are optimised on the train split by the **Cross-Entropy Method** (derivative-free policy search). It searches feature combinations the heuristic study never tried and is what edges past greedy out-of-sample.

The analysis in §2/§4 motivates both: it shows *why* the departure signal must be used with great care, and bounds what is achievable online (a 0.48 perfect-information ceiling vs a ~0.77 online reality).

---

## 2. Train-data analysis (the findings that shaped the design)

**(a) A data trap — container-ID reuse.** When an initial-state container departs, its `container_id` is later **reassigned** to a new arrival. Naively matching the first arrival of an ID to the first departure produces *departures before arrivals* and corrupts any timing analysis. All analysis here matches arrivals→departures **temporally** (a departure pairs with the most recent live occupant of that ID).

**(b) Departure-time is a weak leave-order signal.** With correct matching, `departure_time` (the vessel ETD) correlates with realised retrieval order at only **Spearman ρ ≈ 0.55** (LOAD) / 0.63 (TRUCK_DLVR) for containers we place. ~75 % of LOAD containers leave within 24 h of their ETD, but a long tail slips to later vessel rotations (a container can depart 7+ days from its ETD). Joining the vessel schedule's load windows does **not** improve prediction (ρ unchanged). So leave *order* is only weakly predictable online.

**(c) The load rule is exact but only helps within a batch.** Every vessel loads its containers grouped by `port_of_discharge`, heavy-first within a port (verified 100 %). This makes *within-(vessel, ETD, port)* order deterministic — but those batches interleave in time across vessels, and 24 % of retrievals are truck pickups that ignore vessel order, so the rule alone does not order whole stacks.

**(d) The simulator re-scatters reshuffled containers.** On a retrieval, containers above the target are relocated to the **lowest available stacks**, not back where they were. This continually re-creates disorder that no placement policy controls, and it is a major reshuffle source (§4).

**(e) Space is abundant.** The yard runs ~50 % full (≈1920 columns, ≈6 000 containers, avg height ≈3), so spreading is feasible and there is little capacity pressure forcing tall stacks.

---

## 3. The ERI heuristic (companion)

The index minimised per placement is

```
ERI(c, stack) = ORDER_W · E[ #containers in stack that leave before c ]   (ordering term)
              + BETA · height(stack)                                       (capacity term)
```

**Learned blocking probability.** From the train split we calibrate
`P(leave(b) < leave(c)) = sigmoid((dep_c − dep_b) / τ)` with **τ ≈ 3.7 days**
(fit to the empirical curve: P≈0.5 at a 0-gap, P≈0.13 when *b* departs a week later — confirming the weak, noisy signal). `E[blocking]` sums this over the stack, except for same-batch containers where the deterministic weight rule (heavy on top) is used instead.

**Capacity term.** `BETA · height` makes the index strictly increasing in stack height; each container under `c` is itself a probable future relocation, so this term is an expected-reshuffle cost, not an ad-hoc penalty.

**Deployed configuration: `ordering_weight = 0`.** The ordering term is *suppressed by default* — a decision forced by the empirical study in §4. With `ORDER_W = 0` the index is minimised by the **lowest stack** (robust spreading), with ties broken by scan order across blocks. The full ERI machinery is retained and re-enabled by setting `ordering_weight > 0` for instances with a stronger departure signal.

**Complexity.** O(C) per placement (C ≈ 1920 columns), O(1) per column; departure timestamps cached. Deployed (ORDER_W=0) per-column work is a single height read. Full test run: ~4 s for ~20 k events.

---

## 4. Empirical study: why the ordering term is suppressed

Every way of *using* the departure signal to consolidate "related" containers **increases** reshuffles versus naive lowest-stack spreading, in- and out-of-sample (train split):

| Strategy | reshuffles/retrieval |
|---|---|
| Random baseline | ~0.84 |
| Pure ERI consolidation (best-fit by departure) | 0.89 |
| Confident consolidation (safety margin 1–10 d) | 0.97–1.09 |
| (vessel, port[, ETD]) grouping + weight order | 0.93 |
| Spatial zoning by departure bucket | 0.86 |
| Height-dominant + ERI blocking tie-break | 0.83 |
| Lowest-stack spreading (greedy = ERI with ORDER_W=0) | 0.79 train / 0.77 test |
| **Learned linear policy (CEM) — deployed** | **0.76 train / 0.75 test** |
| Perfect-information best-fit (uses *true* leave times) | **0.48** |

**Mechanism.** Two effects compound. (1) ρ≈0.55 predictability means a consolidated "sorted" stack frequently contains an out-of-order container; in a tall stack one such error forces reshuffles of everything above it. (2) The simulator's re-scatter (§2d) keeps injecting disorder. Spreading sidesteps both: short stacks mean few containers ever sit above anyone, so ordering errors are cheap.

**Where the irreducible reshuffles come from.** Decomposing the *perfect-information* run (0.48 — the best any leave-order-based placement can do here): **31 %** of reshuffles involve initial-state containers (pre-stacked before we act), **16 %** involve simulator-relocated containers, **53 %** freshly-placed. So roughly half the residual is the relocation dynamic itself, and a further third is the fixed initial state — neither is addressable by smarter *placement*.

**Conclusion.** Spreading is the *heuristic* online optimum, and the 0.48 ceiling is reachable only with *true future* leave times. The remaining question — can a learned policy find a small, robust edge over greedy using features the heuristics did not combine? — is answered in §5.

---

## 5. The learned policy (deployed)

**Parameterisation.** A linear cost over 8 per-(container, column) features; place at the column of minimum cost:

```
cost = w · [ bias, depth(h/5), is_empty, p_block(top), on_initial,
             block_occupancy, batch_violation, over_softcap ]
```

`p_block` reuses the calibrated logistic from §3; `on_initial` flags stacking onto a pre-existing (unsorted) initial-state container; `block_occupancy` enables active block balancing; `batch_violation` flags breaking the heavy-on-top load rule within a load batch; `over_softcap` flags resulting height > 3. These include levers the heuristic search never combined — notably `on_initial` and `block_occupancy`.

**Training — Cross-Entropy Method (CEM).** A derivative-free policy search (Rubinstein & Kroese, *The Cross-Entropy Method*): sample weight vectors from a Gaussian, evaluate each by *running the simulator on the train split*, keep the top fraction ("elites"), refit the Gaussian, repeat (`solution/train_rl.py`). The simulator is deterministic given weights, so every evaluation is **noise-free** and CEM optimises the *true* objective (reshuffles/retrieval) with no reward shaping. Warm-started at the greedy policy; best-so-far weights are retained across iterations. Training uses the full train split (20,592 events); typical run: pop=16, elite=4, 8 iterations, ~30–60 min CPU, no GPU required.

**Learned weights (interpretation, full-train CEM run).** `depth +1.58`, `block_occ +2.69`, `batch_violation +4.52` → keep stacks short, blocks balanced, and respect load-order batches. `is_empty +2.26` → prefer filling shallow stacks over always opening fresh columns (conserves empty columns for simulator relocations). `on_initial −1.41` → prefers stacking onto initial-state containers (long-dwell foundations). `p_block −0.60` → mild penalty for burying containers likely to leave first. Full train: **0.7599**; full test: **0.7516** (vs greedy 0.7687), zero violations.

**Why the gain is small but meaningful.** It is bounded by the §4 ceiling: with ρ≈0.55 predictability and uncontrolled relocation re-scatter, there is little structure left to exploit online. The learned policy extracts a real ~2 % out-of-sample improvement over greedy — the honest size of the available edge — rather than the implausible 0.1–0.2 the reference table suggests.

**Complexity.** O(C) per placement (C ≈ 1920); top look-ups only for shallow candidates. Test run ~6 s. Training ~25 min (75 simulator episodes).

---

## 6. Trade-offs and alternatives rejected

- **Pure ERI / grouping / zoning** — rejected: amplify the weak signal into cascades (§4).
- **Deep RL (Jiang et al. 2023)** — rejected: GPU-heavy, unstable training, overkill for ~20k events; CEM on linear features achieves comparable gains at ~25 min CPU.
- **MC Q-learning (Lim et al. 2025)** — considered but not deployed: requires reward shaping and noisy episode returns; CEM directly optimises the true objective on a deterministic simulator.
- **Better leave-time prediction (GMM, schedule join).** Schedule features gave no lift (§2b); even perfect prediction caps at 0.48 via best-fit.
- **Lookahead with `snapshot`/`restore`.** Unavailable online; value must come from a policy learned on train data.

---

## 7. Reproduce

```bash
# Baselines
python -m src.run --strategy src.baseline_greedy.GreedyStrategy --data-dir data/test -v
python -m src.run --strategy solution.eri_strategy.ERIStrategy --data-dir data/test -v

# Deployed strategy on the scored test split
python -m src.run --strategy solution.rl_strategy.RLStrategy --data-dir data/test -o results/results.json -v

# Retrain the policy (CPU, ~30–60 min for full train split)
python -m solution.train_rl --train-events 20592 --pop 16 --elite 4 --iters 8

# Re-enable ERI consolidation for comparison (worse here, see §4):
#   ERIStrategy(ordering_weight=1.0)

# Validate submission
bash validate_submission.sh
```
