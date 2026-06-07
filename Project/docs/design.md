# Container Yard Placement — Design Document

**Deployed strategy:** `solution/rl_strategy.py` — learned linear policy (Cross-Entropy Method) with dynamic ATD features via `on_event`.

**Results (test split, days 21–40), 0 hard-constraint violations:**

| Strategy | reshuffles/retrieval | score / 40 |
|---|---|---|
| Greedy baseline (lowest-stack) | 0.7687 | 11.3 |
| ERI heuristic | 0.7687 | 11.3 |
| Q-learning (tabular, feature-discretised) | 0.7689 | 11.3 |
| Learned policy — 9 features, static ETD | 0.7418 | ~12 |
| **Learned policy — 11 features, dynamic ATD (deployed)** | **0.7230** | **13.3** |

---
## 1. Train-data analysis (findings that shaped all designs)

**(a) Container-ID reuse trap.** When an initial-state container departs, its `container_id` is later reassigned to a new arrival. Naively pairing first arrival -> first departure yields departures before arrivals. All analysis here matches arrivals-departures temporally.

**(b) Departure-time is a weak leave-order signal.** With correct matching, `departure_time` correlates with realised retrieval order at Spearman ρ = 0.55 (LOAD) / 0.63 (TRUCK_DLVR). ~75 % of LOAD containers leave within 24 h of their ETD, but a long tail slips to later vessel rotations (7+ days off). Schedule load windows do not improve ρ.

**(c) 36 % of LOAD events miss their assigned rotation.** VSL003 is the worst spiller (containers loaded 2-3 rotations off-target). VSL008 is the cleanest (94 % on-time). Single-rotation vessels (VSL009-VSL018) are always offset=0 by construction.

**(d) Load rule is exact within a batch.** Every vessel loads grouped by `port_of_discharge`, heavy-first within a port (verified 100 %). Exploited by the `batch_violation` feature.

**(e) Simulator re-scatters reshuffled containers** to the lowest available stacks, continually re-creating disorder. ~16 % of perfect-information reshuffles trace to this — not addressable by placement policy.

**(f) Space is abundant.** ~50 % yard utilisation (~1920 columns, ~6000 containers, avg height =3). Spreading is always feasible.

**(g) Perfect-information ceiling is 0.48.** Even with true future leave times, best-fit placement yields 0.48 reshuffles/retrieval. 31 % of that residual traces to fixed initial-state containers, 16 % to simulator relocations — neither is controllable. The online achievable ceiling is therefore considerably above 0.48.

---
## 2. Deployed model — learned linear policy with dynamic ATD

### 2.1 Algorithm

A linear cost function over 11 per-(container, column) features; place at the column of minimum cost:

cost(c, column) = w * features(c, column)
place at argmin_column cost

Weights `w` are learned offline on the train split by the **Cross-Entropy Method (CEM)**.

### 2.2 Feature vector

| Feature | Description |
|---|---|
| `bias` | constant term |
| `depth` | stack height / 5 → drives spreading |
| `is_empty` | 1 if opening a fresh column |
| `p_block` | P(top-of-stack leaves before c) via calibrated logistic (τ = 3.7 days) |
| `on_initial` | 1 if stacking onto an initial-state container |
| `block_occ` | block occupancy fraction → active block balancing |
| `batch_violation` | 1 if same (vessel, ETD, port) but arriving container heavier than top |
| `over_softcap` | 1 if resulting height > 3 |
| `min_dep_stack` | p_block of the earliest-departing container anywhere in the stack |
| **`top_missed_rot`** | **1 if top-of-stack's load window has already closed (missed its rotation)** |
| **`c_loading_now`** | **1 if arriving container's own load window is currently open** |

### 2.3 Dynamic ATD via `on_event`

`on_event()` is called for every simulator event before placement and advances the strategy's internal clock (`_current_ts`). This enables the two real-time features:

**`top_missed_rot`** (learned weight -2.15): A container whose load window has closed has missed its assigned vessel rotation and won't leave until the next one — potentially weeks away. Without this feature, such a container looks perpetually "overdue-urgent" in raw ETD space, causing the policy to avoid stacking anything on top of it even though the next rotation is far off. The negative weight correctly penalises columns blocked by a stuck missed-rotation container.

**`c_loading_now`** (learned weight +1.08): When the arriving container's load window is currently open it will be retrieved soon. The positive weight nudges the policy toward shallower/more accessible stacks for these containers, reducing reshuffles at retrieval time.

Raw ETD is kept unchanged as the departure signal for `p_block` and `min_dep_stack` — this preserves the trained feature distribution so the other 9 weights remain valid. An earlier attempt to replace ETD values directly with dynamic load-window midpoints caused CEM to overfit on train (0.698) with poor test generalisation (0.788), so that approach was abandoned.

The rotation window table (vessel schedule → `load_start`, `load_end` per (vessel, ETD) pair) is built once at initialisation from `data/vessel_schedule.json` and queried in O(1) per event.

### 2.4 Training — Cross-Entropy Method

CEM (Rubinstein & Kroese, *The Cross-Entropy Method*): sample weight vectors from a Gaussian, evaluate each by running the full simulator on the train split, keep the top fraction (elites), refit the Gaussian, repeat (`solution/train_rl.py`). The simulator is deterministic given a weight vector, so every evaluation is **noise-free** and CEM optimises the **true objective** (reshuffles/retrieval) directly, with no reward shaping or credit-assignment approximation.

Warm-started from the greedy policy (depth-only weights); best-so-far weights retained across iterations.

**Configuration:** pop=16, elite=4, 14 iterations, σ=1.5, full train split (20,592 events), 8 parallel workers — ~40-50 min wall-clock, no GPU required.

### 2.5 Learned weights and interpretation

bias             +0.48    depth            +2.98    is_empty          +0.63
p_block          +1.74    on_initial       -0.52    block_occ         +0.66
batch_violation  +2.41    over_softcap     -0.29    min_dep_stack     -1.11
top_missed_rot   -2.15    c_loading_now    +1.08

- `depth` (+2.98), `batch_violation` (+2.41) → keep stacks short, respect load order
- `top_missed_rot` (-2.15) → strongly avoid stacking onto stuck missed-rotation containers
- `on_initial` (-0.52) → prefer using initial-state containers as foundations (long dwell)
- `min_dep_stack` (-1.11) → a stack whose earliest container is about to leave is actually good (it will vacate space soon)
- `c_loading_now` (+1.08) → imminent retrieval containers prefer accessible shallow stacks

### 2.6 Complexity

**Time:** O(C) per placement (C ≈ 1920 columns). Top look-ups only for shallow candidates (height ≤ h_min + 1); `on_event` is O(1). Full test run: ~35 s.
**Space:** O(C) column index + O(V*R) rotation window table (V vessels × R rotations ≈ 50 entries).

---
## 3. Other methods tried

### 3.1 ERI heuristic (`eri_strategy.py`)

**Algorithm:** Expected Reshuffle Index — place each arriving container on the stack minimising the expected number of containers it will bury:
ERI(c, stack) = ORDER_W * E[ #containers in stack that leave before c ]
+ BETA * height(stack)

Blocking probability calibrated as `P(leave(b) < leave(c)) = sigmoid((dep_c - dep_b) / τ)` with τ = 3.7 days.

**Result:** 0.7687 reshuffles/retrieval (11.3 / 40) — matches greedy, no improvement.

**Why it fails here:** With ρ = 0.55 the ordering term amplifies the weak departure signal into cascades. A consolidated "sorted" stack frequently contains an out-of-order container; one such error in a tall stack forces reshuffles for everything above it. Every consolidation strategy tested — ERI, (vessel, port, ETD) grouping, spatial zoning by departure bucket — was worse than pure spreading (tested across the full range of `ordering_weight`). Setting `ORDER_W = 0` reduces ERI to greedy spreading, which is the empirical heuristic optimum.

The full ERI code is retained as a clean reference for instances with a stronger departure signal.

**Papers:** H. Bisira & A. Salhi (2021), *Reshuffle minimisation to improve storage yard operations efficiency*; Kim & Hong (2006); Galle et al. (2018), *The Stochastic Container Relocation Problem*.

---
### 3.2 Tabular Q-learning (`q_learning_strategy.py`)

**Algorithm:** Feature-discretised tabular Q-learning following Liu et al. (2025). State = abstract column profile from 4 discretised features (height bin, departure relation, weight order, block load) giving 5×4×2×3 = 120 abstract column profiles. Action = which profile to place on. Q-table updated via TD(0) with ε-greedy exploration over multiple episode replays of the train split.

**Result:** 0.7689 reshuffles/retrieval (11.3 / 40) — matches greedy, no improvement.

**Why it fails here:** Discretising continuous features (height, departure gap, occupancy) into coarse bins loses the resolution that the CEM linear policy exploits — e.g. the fine-grained `depth` and `block_occ` signals that drove the CEM improvement collapse to 3-5 bins. The Q-table converges but to a policy that is essentially equivalent to lowest-stack spreading, the same attractor as ERI with ORDER_W=0.

**Paper:** Liu et al. (2025), *A Q-learning based algorithm for the block relocation problem*.

---
### 3.3 ATD prediction model injected as CEM feature (`atd_model.py`)

**Idea:** Predict each container's Actual Departure Time (ATD) from static vessel schedule features, then feed the predicted ATD — rather than the raw ETD — into the placement policy as the departure signal. A more accurate urgency estimate should improve placement decisions.

**Model:** `solution/atd_model.py` predicts ATD for LOAD events as:
ATD = load_start + frac * (load_end - load_start)
where `frac` is the median fractional position within the load window at which containers of that (vessel, port_of_discharge, weight_class) are actually retrieved, computed from the train split via `solution/analyse_all_vessels.py`. 235 per-(vessel, port, weight) groups are stored in `solution/vessel_analysis/summary_table.json`, with fallback to per-(port, weight) medians (46 groups) and then a global median (0.69).

A yard-congestion extension was also built: the rotation-offset model (`vessel_rotation_model.json`) fits P(rotation_offset | congestion_ratio) per vessel and blends ATD predictions across rotation windows weighted by those probabilities.

**ATD prediction MAE:**
| Model | MAE (LOAD events) |
|---|---|
| Baseline: ATD = ETD | 146.13 h |
| ETD + per-vessel bias | 132.33 h |
| vpw-frac model (deployed in atd_model.py) | 130.71 h |
| vpw-frac + congestion blending | 131.29 h (worse) |

**Result when injected into CEM:** No improvement over using raw ETD. Replacing `departure_time` with the predicted ATD timestamp in `p_block` / `min_dep_stack` shifted the feature distribution CEM was trained on, causing it to learn weights that over-fitted on train. The ATD model's 130 h MAE is also too coarse to improve ordering decisions — at that noise level, predicted ATD is not a better ordering signal than raw ETD (ρ = 0.55 either way).

**Why congestion blending hurt:** The weighted blend sums probabilities across rotation windows weeks apart. Even when the modal offset is correct, the non-zero probability mass on distant rotations pulled the point-estimate ATD far from the true value, increasing MAE by 0.58 h.

**Why the frac model works for ATD prediction but not for placement:** The per-(vessel, port, weight) frac captures that e.g. PORT_04|HEAVY containers on VSL001 are always retrieved at frac=0.79 of the load window. This is useful for logistics planning (predicting when a container will leave the yard) but does not change the relative ordering between containers competing for the same stack slot — which is what placement decisions depend on.

The ATD model is retained in `solution/atd_model.py` for operational use (predicting container departure times given vessel schedule data).

---
### 3.4 Learned policy — 9 features, static ETD

The same CEM framework as §2 but without the two dynamic ATD features (`top_missed_rot`, `c_loading_now`). `departure_time` used as a raw static value throughout the simulation, even after a container's load window has closed.

**Result:** 0.7418 reshuffles/retrieval (~12 / 40).

**Why the dynamic features help:** Once a vessel's load window closes, any container in the yard with that ETD is stuck — it missed its rotation and won't leave for weeks. The static policy cannot distinguish this from a container whose ETD is genuinely imminent, so it wastes columns avoiding them. The `top_missed_rot` feature (weight -2.15) gives CEM an explicit handle on this case, yielding +0.019 improvement on test.

---
## 4. Why ordering-based strategies fail (detailed)

Every departure-signal-based consolidation strategy tested increased reshuffles vs greedy:

| Strategy | reshuffles/retrieval |
|---|---|
| Pure ERI consolidation | 0.89 |
| Confident consolidation (1-10 d safety margin) | 0.97-1.09 |
| (vessel, port, ETD) grouping + weight order | 0.93 |
| Spatial zoning by departure bucket | 0.86 |
| Height + ERI blocking tie-break | 0.83 |
| Greedy spreading | 0.79 train / 0.77 test |
| **Learned policy with dynamic ATD (deployed)** | **0.73 train / 0.72 test** |
| Perfect-information best-fit | 0.48 |

The learned policy is the only method that beats greedy, and it does so by combining spreading (dominant `depth` weight) with targeted corrections (`top_missed_rot`, `batch_violation`) rather than by sorting.

---
## 5. Repository layout

solution/
rl_strategy.py        # deployed strategy (RLStrategy)
rl_weights.json       # learned CEM weights (11 features)
train_rl.py           # CEM training script
eri_strategy.py       # ERI heuristic (tried, not deployed)
q_learning_strategy.py # Q-learning strategy (tried, not deployed)
train_q_learning.py   # Q-learning training script
atd_model.py          # ATD prediction model (tried as CEM feature, §3.3)
vessel_analysis/      # per-vessel frac tables + rotation model
docs/
design.md             # this document
results/
results.json          # test output (0.7230, 13.3/40) — deployed strategy
tests/
test_rl_strategy.py   # 54 unit tests for RLStrategy
test_eri_strategy.py  # unit tests for ERIStrategy
test_q_learning_strategy.py
test_yard_state.py
test_scoring.py

---
## 6. Reproduce

```bash
# Deployed strategy on the scored test split
python -m src.run --strategy solution.rl_strategy.RLStrategy \
    --data-dir data/test -o results/results.json -v

# Retrain the policy (CPU, ~40-50 min)
python -m solution.train_rl \
    --train-events 20592 --pop 16 --elite 4 --iters 14 \
    --sigma 1.5 --from-greedy --workers 8 --seed 42

# Other strategies (for comparison)
python -m src.run --strategy src.baseline_greedy.GreedyStrategy --data-dir data/test -v
python -m src.run --strategy solution.eri_strategy.ERIStrategy --data-dir data/test -v
python -m src.run --strategy solution.q_learning_strategy.QLearningStrategy --data-dir data/test -v

# Tests
python -m pytest tests/ -v