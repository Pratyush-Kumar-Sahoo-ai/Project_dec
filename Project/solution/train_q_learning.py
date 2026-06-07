"""Train the Q-learning placement policy (solution/q_learning_strategy.py).

Implements offline Q-learning for the container yard placement problem,
following the approach of Liu et al. (2025), "A Q-learning based algorithm
for the block relocation problem."

Key insight: The BRP has **delayed, sparse rewards** - reshuffles happen at
retrieval, far after the placement decision. We solve credit assignment with
a two-phase approach:

Phase 1 (Monte Carlo estimation): Run episodes tracking which *profile* each
container was placed under. When a retrieval has reshuffles, attribute cost to
the profiles of containers that were blocking. Accumulate per-profile average
cost across many placements.

Phase 2 (Q-value refinement): Use the MC estimates as initial Q-values, then
run targeted exploration episodes that try alternative profiles for the
high-cost ones, updating Q-values with temporal-difference learning.

The critical fix over naive Q-learning: we only attribute reshuffles to
containers **we placed** (not those relocated by the simulator or present
in the initial state), and we track whether a container is still in its
*original* placement position (vs. having been reshuffled elsewhere by the
simulator).

Usage:
    python -m solution.train_q_learning          # train + save q_table.json
    python -m solution.train_q_learning --episodes 40 --quick

Output: solution/q_table.json (consumed by QLearningStrategy).
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from src.models import Container, Event, Position
from src.yard_state import YardState
from src.placement_interface import PlacementStrategy
from src.scoring import SimulationStats
from solution.q_learning_strategy import (
    NUM_PROFILES, profile_index, dep_relation_bin, block_load_bin,
    _WEIGHT_RANK, _TAU_SECONDS, NUM_HEIGHT_BINS
)

_HERE = os.path.dirname(__file__)
_Q_TABLE_PATH = os.path.join(_HERE, "q_table.json")


def _load(split):
    layout = json.load(open(os.path.join("data", "yard_layout.json")))
    init = json.load(open(os.path.join("data", split, "initial_state.json")))
    rows = [json.loads(l) for l in open(os.path.join("data", split, "events.jsonl"))]
    events = [Event(**{k: v for k, v in e.items()
                       if k in Event.__dataclass_fields__}) for e in rows]
    return layout, init, events


def _dep_ts(iso: str, cache: dict) -> float:
    if not iso:
        return float("inf")
    v = cache.get(iso)
    if v is None:
        try:
            v = datetime.fromisoformat(iso).timestamp()
        except ValueError:
            v = float("inf")
        cache[iso] = v
    return v


class QLearningTrainer:
    """Monte Carlo Q-estimation with targeted refinement."""

    def __init__(self, layout: dict, init: dict, events: List[Event],
                 alpha: float = 0.02, epsilon_start: float = 0.15,
                 epsilon_end: float = 0.01, seed: int = 0):
        self.layout = layout
        self.init = init
        self.events = events
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.rng = random.Random(seed)
        # Initialize Q-table at zero - this reproduces greedy (lowest-stack)
        # behavior. Q-learning then learns *corrections* that improve on greedy.
        self.q_table = [0.0] * NUM_PROFILES
        self._dep_cache: Dict[str, float] = {}

        self._columns: List[Tuple[str, int, int]] = []
        self._max_tiers: Dict[str, int] = {}
        self._cap_block: Dict[str, int] = {}
        self._init_ids: Set[str] = set()
        for block, info in layout["blocks"].items():
            self._max_tiers[block] = info["tiers"]
            n_cols = info["bays"] * info["rows"]
            self._cap_block[block] = info["tiers"] * n_cols
            for bay in range(1, info["bays"] + 1):
                for row in range(1, info["rows"] + 1):
                    self._columns.append((block, bay, row))
        self._init_ids = {c["container_id"]
                          for c in init.get("containers", [])}

        # Per-profile accumulators (lifetime totals for analysis)
        self._profile_placements: List[int] = [0] * NUM_PROFILES
        self._profile_reshuffles: List[int] = [0] * NUM_PROFILES

    def _compute_profile(self, yard: YardState, block: str, bay: int,
                         row: int, height: int, dep_c: float,
                         weight_c: int, vessel_c: str, port_c: str,
                         dep_time_c: str, occ_block: int) -> int:
        height_bin = min(height, NUM_HEIGHT_BINS - 1)

        if height == 0:
            dep_bin = 0
        else:
            top_id = yard.get_container_at(block, bay, row, height)
            top = yard.get_container_info(top_id) if top_id else None
            if top is None:
                dep_bin = 0
            else:
                dep_top = _dep_ts(top.departure_time, self._dep_cache)
                dep_bin = dep_relation_bin(dep_top, dep_c)

        weight_bin = 0
        if height > 0:
            top_id = yard.get_container_at(block, bay, row, height)
            top = yard.get_container_info(top_id) if top_id else None
            if top is not None:
                if (top.vessel_id == vessel_c
                        and top.departure_time == dep_time_c
                        and top.port_of_discharge == port_c
                        and weight_c > _WEIGHT_RANK.get(top.weight_class, 1)):
                    weight_bin = 1

        cap = self._cap_block.get(block, 1)
        load_bin = block_load_bin(occ_block / cap if cap > 0 else 0.0)

        return profile_index(height_bin, dep_bin, weight_bin, load_bin)

    def _select_action(self, yard: YardState, event: Event, epsilon: float
                       ) -> Tuple[Position, int]:
        c = event.to_container()
        dep_c = _dep_ts(c.departure_time, self._dep_cache)
        weight_c = _WEIGHT_RANK.get(c.weight_class, 1)
        get_h = yard.get_stack_height

        occ: Dict[str, int] = {b: 0 for b in self._max_tiers}
        heights: Dict[Tuple[str, int, int], int] = {}
        for col in self._columns:
            block, bay, row = col
            h = get_h(block, bay, row)
            heights[col] = h
            occ[block] += h

        candidates: List[Tuple[float, Position, int]] = []
        for col in self._columns:
            block, bay, row = col
            h = heights[col]
            if h >= self._max_tiers[block]:
                continue
            pidx = self._compute_profile(
                yard, block, bay, row, h, dep_c,
                weight_c, c.vessel_id, c.port_of_discharge,
                c.departure_time, occ[block])
            q_val = self.q_table[pidx] + h * 0.001
            candidates.append((q_val, Position(block, bay, row, h + 1), pidx))

        if not candidates:
            block, bay, row = self._columns[0]
            return Position(block, bay, row, 1), 0

        if self.rng.random() < epsilon:
            _, pos, pidx = self.rng.choice(candidates)
            return pos, pidx
        else:
            candidates.sort(key=lambda x: x[0])
            _, pos, pidx = candidates[0]
            return pos, pidx

    def run_episode(self, epsilon: float) -> float:
        """Run one episode with advantage-based Q-updates."""

        # Key idea: track reshuffle rate per profile and update Q-values
        # based on whether a profile performs above or below the running
        # average. This prevents all profiles from being pushed in the same
        # direction (which happened with absolute penalties).

        yard = YardState(self.layout)
        yard.load_initial_state(self.init)

        # Track containers we placed: container_id -> (profile_idx, block, bay, row)
        placement_info: Dict[str, Tuple[int, str, int, int]] = {}

        # Per-profile episode stats
        ep_placements: Dict[int, int] = defaultdict(int)
        ep_caused_reshuffles: Dict[int, int] = defaultdict(int)

        total_reshuffles = 0
        total_retrievals = 0

        for event in self.events:
            if event.type in ("DISCHARGE", "TRUCK_RECV"):
                pos, pidx = self._select_action(yard, event, epsilon)
                container = event.to_container()
                if yard.is_position_valid(pos):
                    yard.place_container(container, pos)
                    placement_info[event.container_id] = (
                        pidx, pos.block, pos.bay, pos.row)
                    ep_placements[pidx] += 1
                    self._profile_placements[pidx] += 1
                else:
                    fb = self._fallback(yard)
                    if fb:
                        yard.place_container(container, fb)

            elif event.type in ("LOAD", "TRUCK_DLVR"):
                total_retrievals += 1
                cpos = yard.get_container_position(event.container_id)
                if cpos is None:
                    continue

                above = yard.get_containers_above(event.container_id)
                reshuffles = len(above)
                total_reshuffles += reshuffles

                if reshuffles > 0:
                    for blocking_id in above:
                        if blocking_id not in placement_info:
                            continue
                        info = placement_info[blocking_id]
                        pidx = info[0]
                        cur_pos = yard.get_container_position(blocking_id)
                        if cur_pos is None:
                            continue
                        orig_block, orig_bay, orig_row = info[1], info[2], info[3]
                        if (cur_pos.block == orig_block
                                and cur_pos.bay == orig_bay
                                and cur_pos.row == orig_row):
                            ep_caused_reshuffles[pidx] += 1
                            self._profile_reshuffles[pidx] += 1

                self._do_retrieval(yard, event.container_id, above)

        # Compute per-profile reshuffle RATE for this episode
        # and compare to the global average rate (advantage estimation)
        total_placed = sum(ep_placements.values())
        total_caused = sum(ep_caused_reshuffles.values())
        avg_rate = total_caused / total_placed if total_placed > 0 else 0.0

        # Update Q-values: profiles with above-average reshuffle rate get
        # increased Q (making them less preferred); below-average get decreased.
        for pidx, count in ep_placements.items():
            if count < 3:
                continue # skip profiles with too few samples
            rate = ep_caused_reshuffles.get(pidx, 0) / count
            advantage = rate - avg_rate  # positive = worse than average
            old_q = self.q_table[pidx]
            self.q_table[pidx] = old_q + self.alpha * advantage

        if total_retrievals == 0:
            return 0.0
        return total_reshuffles / total_retrievals

    def _do_retrieval(self, yard: YardState, target_id: str,
                      above: List[str]) -> None:
        if not above:
            yard.remove_container(target_id)
            return

        pos = yard.get_container_position(target_id)
        if pos is None:
            return
        block = pos.block

        temp_containers = []
        for cid in reversed(above):
            info = yard.get_container_info(cid)
            yard.remove_container(cid)
            if info:
                temp_containers.append((cid, info))

        yard.remove_container(target_id)

        for cid, cont in reversed(temp_containers):
            rpos = self._find_reshuffle_position(yard, block)
            if rpos:
                yard.place_container(cont, rpos)

    def _find_reshuffle_position(self, yard: YardState,
                                 block: str) -> Optional[Position]:
        bi_tiers = self._max_tiers.get(block, 5)
        best_pos = None
        best_h = bi_tiers + 1

        for b, bay, row in self._columns:
            if b != block:
                continue
            h = yard.get_stack_height(b, bay, row)
            if h < bi_tiers and h < best_h:
                best_h = h
                best_pos = Position(b, bay, row, h + 1)

        if best_pos is None:
            for b, bay, row in self._columns:
                if b == block:
                    continue
                h = yard.get_stack_height(b, bay, row)
                tiers = self._max_tiers.get(b, 5)
                if h < tiers and h < best_h:
                    best_h = h
                    best_pos = Position(b, bay, row, h + 1)

        return best_pos

    def _fallback(self, yard: YardState) -> Optional[Position]:
        best_pos = None
        best_h = 99
        for block, bay, row in self._columns:
            h = yard.get_stack_height(block, bay, row)
            if h < self._max_tiers[block] and h < best_h:
                best_h = h
                best_pos = Position(block, bay, row, h + 1)
        return best_pos

    def train(self, episodes: int = 40) -> List[float]:
        """Run training. Phase 1: MC estimation, Phase 2: refinement."""
        scores = []

        # Phase 1: Collect MC estimates with moderate exploration
        phase1_eps = max(episodes // 2, 5)
        print(f" Phase 1: MC estimation ({phase1_eps} episodes)")
        for ep in range(phase1_eps):
            t0 = time.time()
            frac = ep / max(phase1_eps - 1, 1)
            epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac * 0.5
            score = self.run_episode(epsilon)
            elapsed = time.time() - t0
            scores.append(score)
            print(f"    episode {ep+1}/{phase1_eps}: "
                  f"reshuffles/retrieval={score:.4f} "
                  f"eps={epsilon:.3f} ({elapsed:.0f}s)")

        # Phase 2: Refine with low exploration (greedy exploitation)
        phase2_eps = episodes - phase1_eps
        print(f" Phase 2: Refinement ({phase2_eps} episodes)")
        for ep in range(phase2_eps):
            t0 = time.time()
            frac = ep / max(phase2_eps - 1, 1)
            epsilon = self.epsilon_end + (0.05 - self.epsilon_end) * (1.0 - frac)
            score = self.run_episode(epsilon)
            elapsed = time.time() - t0
            scores.append(score)
            print(f"    episode {ep+1}/{phase2_eps}: "
                  f"reshuffles/retrieval={score:.4f} "
                  f"eps={epsilon:.3f} ({elapsed:.0f}s)")

        return scores


def evaluate(q_values, layout, init, events):
    """Run greedy Q-policy, return reshuffles/retrieval and violations."""
    from solution.q_learning_strategy import QLearningStrategy
    from src.simulator import Simulator

    yard = YardState(layout)
    yard.load_initial_state(init)
    strat = QLearningStrategy(q_values=q_values)
    strat.initialize(layout, init)
    stats = Simulator(yard, strat).run(events)
    return stats.reshuffles_per_retrieval, stats.hard_constraint_violations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--epsilon-start", type=float, default=0.15)
    ap.add_argument("--epsilon-end", type=float, default=0.01)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.quick:
        args.episodes = 15

    print("Loading train data...")
    layout, init, train_events = _load("train")
    print(f"  {len(train_events)} events loaded")

    print(f"\nQ-learning training ({args.episodes} episodes, alpha={args.alpha})")
    trainer = QLearningTrainer(
        layout, init, train_events,
        alpha=args.alpha,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        seed=args.seed)

    scores = trainer.train(args.episodes)

    # Evaluate greedy policy on train and test
    print("\nEvaluating learned Q-table (greedy)...")
    train_ratio, train_viol = evaluate(trainer.q_table, layout, init, train_events)
    print(f" Train: reshuffles/retrieval = {train_ratio:.4f}, violations = {train_viol}")

    print("Loading test data...")
    layout_te, init_te, test_events = _load("test")
    test_ratio, test_viol = evaluate(trainer.q_table, layout_te, init_te, test_events)
    print(f" Test:  reshuffles/retrieval = {test_ratio:.4f}, violations = {test_viol}")

    # Profile analysis
    active = sum(1 for v in trainer._profile_placements if v > 0)
    print(f"\n Active profiles: {active}/{NUM_PROFILES}")
    print(" Top-5 highest-cost profiles:")
    ranked = sorted(range(NUM_PROFILES), key=lambda i: trainer.q_table[i],
                    reverse=True)
    for i, pidx in enumerate(ranked[:5]):
        placements = trainer._profile_placements[pidx]
        reshuffles = trainer._profile_reshuffles[pidx]
        print(f"    #{i+1} profile={pidx:3d} Q={trainer.q_table[pidx]:.3f} "
              f"placements={placements} reshuffles_caused={reshuffles}")

    # Save Q-table
    output = {
        "q_values": trainer.q_table,
        "profile_placements": trainer._profile_placements,
        "profile_reshuffles": trainer._profile_reshuffles,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "train_violations": train_viol,
        "test_violations": test_viol,
        "active_profiles": active,
        "config": vars(args),
        "episode_scores": scores,
    }
    with open(_Q_TABLE_PATH, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nSaved Q-table -> {_Q_TABLE_PATH}")


if __name__ == "__main__":
    main()