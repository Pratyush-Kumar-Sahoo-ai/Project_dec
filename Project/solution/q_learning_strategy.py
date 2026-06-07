"""Q-Learning placement policy for the Block Relocation Problem.

Implements a tabular Q-learning approach based on:

    Liu et al. (2025), "A Q-learning based algorithm for the block
    relocation problem"

The key idea: discretize the per-column state into a compact profile
(height, departure relation, weight order, block load) and learn Q-values
for each profile offline. At decision time, compute the profile for every
candidate column and place on the one with the lowest Q-value.

State representation (per candidate column):
  - height_bin:   stack height {0,1,2,3,4}          (5 bins)
  - dep_relation: departure gap to top container    (4 bins)
                  {EMPTY, EARLIER, SIMILAR, LATER}
  - weight_order: heavy-on-top rule compliance      (2 bins)
  - block_load:   block occupancy fraction          (3 bins)

Total profiles: 5 * 4 * 2 * 3 = 120.
Q-table: profile_index -> expected future reshuffles (lower = better).

Training: offline via episode replay on the train split with delayed reward
attribution -- when a retrieval causes reshuffles, the containers that were
blocking get penalized. See ``solution/train_q_learning.py``.

Complexity: O(C) per placement (C ~ 1920 columns), O(1) per column.
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

_WEIGHT_RANK = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}
_TAU_SECONDS = 3.7 * 86400.0
_SOFT_CAP = 3

NUM_HEIGHT_BINS = 5
NUM_DEP_BINS = 4      # EMPTY=0, EARLIER=1, SIMILAR=2, LATER=3
NUM_WEIGHT_BINS = 2   # OK=0, VIOLATION=1
NUM_LOAD_BINS = 3     # LOW=0, MED=1, HIGH=2

NUM_PROFILES = NUM_HEIGHT_BINS * NUM_DEP_BINS * NUM_WEIGHT_BINS * NUM_LOAD_BINS

_DEFAULT_Q_TABLE_PATH = os.path.join(os.path.dirname(__file__), "q_table.json")


def profile_index(height_bin: int, dep_bin: int, weight_bin: int,
                  load_bin: int) -> int:
    return (height_bin * NUM_DEP_BINS * NUM_WEIGHT_BINS * NUM_LOAD_BINS
            + dep_bin * NUM_WEIGHT_BINS * NUM_LOAD_BINS
            + weight_bin * NUM_LOAD_BINS
            + load_bin)


def dep_relation_bin(dep_top: float, dep_c: float) -> int:
    if dep_top == float("inf") and dep_c == float("inf"):
        return 2  # SIMILAR (both unknown)
    gap_days = (dep_c - dep_top) / 86400.0
    if gap_days > 3.0:
        return 3  # LATER: top leaves much earlier than c -> c buries it
    elif gap_days < -3.0:
        return 1  # EARLIER: c leaves much earlier -> c won't bury top
    else:
        return 2  # SIMILAR: within 3 days


def block_load_bin(occupancy_frac: float) -> int:
    if occupancy_frac < 0.4:
        return 0  # LOW
    elif occupancy_frac < 0.7:
        return 1  # MED
    else:
        return 2  # HIGH


class QLearningStrategy(PlacementStrategy):
    """Tabular Q-learning placement policy (Liu et al., 2025)."""

    def __init__(self, q_table_path: str = _DEFAULT_Q_TABLE_PATH,
                 q_values: Optional[List[float]] = None) -> None:
        self._q_table = q_values if q_values else self._load_q_table(q_table_path)
        self._dep_cache: Dict[str, float] = {}
        self._columns: List[Tuple[str, int, int]] = []
        self._max_tiers: Dict[str, int] = {}
        self._cap_block: Dict[str, int] = {}
        self._init_ids: set = set()

    @staticmethod
    def _load_q_table(path: str) -> List[float]:
        try:
            with open(path) as fh:
                data = json.load(fh)
            return [float(v) for v in data["q_values"]]
        except (OSError, KeyError, ValueError, TypeError):
            return [0.0] * NUM_PROFILES

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._columns = []
        self._max_tiers = {}
        n_cols: Dict[str, int] = {}
        for block, info in yard_layout["blocks"].items():
            self._max_tiers[block] = info["tiers"]
            n_cols[block] = info["bays"] * info["rows"]
            for bay in range(1, info["bays"] + 1):
                for row in range(1, info["rows"] + 1):
                    self._columns.append((block, bay, row))
        self._cap_block = {b: self._max_tiers[b] * n_cols[b]
                           for b in self._max_tiers}
        self._init_ids = {c["container_id"]
                          for c in initial_state.get("containers", [])}

    def _dep(self, iso: str) -> float:
        if not iso:
            return float("inf")
        v = self._dep_cache.get(iso)
        if v is None:
            try:
                v = datetime.fromisoformat(iso).timestamp()
            except ValueError:
                v = float("inf")
            self._dep_cache[iso] = v
        return v

    def _compute_profile(self, yard_state: YardState, block: str, bay: int,
                         row: int, height: int, dep_c: float,
                         weight_c: int, vessel_c: str, port_c: str,
                         dep_time_c: str, occ_block: int) -> int:
        height_bin = min(height, NUM_HEIGHT_BINS - 1)

        if height == 0:
            dep_bin = 0  # EMPTY
        else:
            top_id = yard_state.get_container_at(block, bay, row, height)
            top = yard_state.get_container_info(top_id) if top_id else None
            if top is None:
                dep_bin = 0
            else:
                dep_top = self._dep(top.departure_time)
                dep_bin = dep_relation_bin(dep_top, dep_c)

        weight_bin = 0  # OK
        if height > 0:
            top_id = yard_state.get_container_at(block, bay, row, height)
            top = yard_state.get_container_info(top_id) if top_id else None
            if top is not None:
                if (top.vessel_id == vessel_c
                        and top.departure_time == dep_time_c
                        and top.port_of_discharge == port_c
                        and weight_c > _WEIGHT_RANK.get(top.weight_class, 1)):
                    weight_bin = 1  # VIOLATION

        cap = self._cap_block.get(block, 1)
        load_bin = block_load_bin(occ_block / cap if cap > 0 else 0.0)

        return profile_index(height_bin, dep_bin, weight_bin, load_bin)

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        c = event.to_container()
        dep_c = self._dep(c.departure_time)
        weight_c = _WEIGHT_RANK.get(c.weight_class, 1)

        get_h = yard_state.get_stack_height

        # Pre-compute per-block occupancy
        occ: Dict[str, int] = {b: 0 for b in self._max_tiers}
        heights: Dict[Tuple[str, int, int], int] = {}
        for col in self._columns:
            block, bay, row = col
            h = get_h(block, bay, row)
            heights[col] = h
            occ[block] += h

        best_pos: Optional[Position] = None
        best_q = float("inf")
        fallback: Optional[Position] = None

        for col in self._columns:
            block, bay, row = col
            h = heights[col]
            if h >= self._max_tiers[block]:
                continue

            if fallback is None:
                fallback = Position(block, bay, row, h + 1)

            pidx = self._compute_profile(
                yard_state, block, bay, row, h, dep_c,
                weight_c, c.vessel_id, c.port_of_discharge,
                c.departure_time, occ[block])

            q_val = self._q_table[pidx]

            # Tie-break: prefer lower stacks (spreading bias)
            cost = q_val + h * 0.001

            if cost < best_q:
                best_q = cost
                best_pos = Position(block, bay, row, h + 1)

        return best_pos if best_pos is not None else fallback