"""Learned placement policy (linear policy trained by Cross-Entropy Method).

This is the learning-based counterpart to the ERI heuristic in
``eri_strategy.py``. It follows the feature-based reinforcement-learning line of
work on online container stacking - e.g. Hottung & Tierney's neural models and,
most directly,

    Maglić et al. / "Online Stacking Using Reinforcement Learning with
    Positional and Tactical Features" (2020),

and the policy-search framing common to the DRL container-stacking literature
(Jiang et al., *Container stacking optimization based on Deep Reinforcement
Learning*, Eng. Appl. AI 2023). Rather than a deep network (overkill for a
~20k-event horizon and risky to train reliably here), we use a **linear policy
over hand-designed features optimised by the Cross-Entropy Method (CEM)** - a
derivative-free policy-search algorithm that directly minimises the *true*
episode objective (total reshuffles), with no reward-shaping or credit-assignment
approximations.

Decision rule (per arriving container ``c``):

    cost(c, column) = w * features(c, column)
    place at argmin_column cost

The weight vector ``w`` is learned offline on the train split by CEM and stored
in ``solution/rl_weights.json``; see ``solution/train_rl.py``. If no trained
weights are present the policy falls back to weights that reproduce lowest-stack
spreading (the strong baseline established in docs/design.md).

Why this can, in principle, beat the hand-tuned heuristics: CEM searches over
feature combinations the manual study never tried - in particular
``on_initial`` (avoid stacking on pre-existing, unsorted initial-state
containers, which account for ~31% of the perfect-information oracle's
reshuffles) and ``block_occupancy`` (active block balancing so the simulator's
reshuffle relocations land on shallow stacks). Whether these yield a real gain
over greedy is an empirical question the training answers honestly.

Complexity: O(C) per placement (C ≈ 1920 columns); expensive top-container
look-ups are computed only for the shallow candidate columns.
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

# Ordered feature names (must match train_rl.py).
FEATURES = (
    "bias",             # constant
    "depth",            # current height / 5 -> drives spreading
    "is_empty",         # 1 if opening a fresh column
    "p_block",          # learned P(top leaves before c)  (departure signal)
    "on_initial",       # 1 if stacking onto an initial-state container
    "block_occ",        # occupancy fraction of the candidate's block
    "batch_violation",  # 1 if same load-batch but heavier-than-top (breaks order)
    "over_softcap",     # 1 if resulting height exceeds the soft cap
    "min_dep_stack",    # p_block of the earliest-departing container in the stack
    # Dynamic ATD features (populated by on_event clock tracking):
    "top_missed_rot",   # 1 if top-of-stack container's load window has CLOSED
                        # (it missed its rotation - stacking here blocks for weeks)
    "c_loading_now",    # 1 if arriving container's load window is currently OPEN
                        # (needs to be retrieved soon - prefer shallow/accessible)
)

# Fallback weights: reproduce lowest-stack spreading (cost == depth only).
_GREEDY_WEIGHTS = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "rl_weights.json")
_SCHEDULE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vessel_schedule.json")


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _build_rotation_windows() -> Dict[Tuple[str, str], dict]:
    """Build lookup: (vessel_id, etd_iso) -> load window + next-rotation midpoint.

    Used by _dep_for() to return the current effective departure timestamp
    given what the simulator clock says "now" is:
    - before load_start : load window midpoint (steady state expectation)
    - during load window: midpoint of the remaining window (rising urgency)
    - after load_end   : next rotation's load midpoint, or +30 days if none
                         (container missed this rotation - not immediately urgent)
    """
    try:
        sched = json.load(open(_SCHEDULE_PATH))
    except OSError:
        return {}
    windows: Dict[Tuple[str, str], dict] = {}
    for vessel in sched["vessels"]:
        vid = vessel["vessel_id"]
        rots = sorted(vessel["rotations"], key=lambda r: r["etd"])
        for i, rot in enumerate(rots):
            ls = _ts(rot["load_start"])
            le = _ts(rot["load_end"])
            mid = (ls + le) / 2.0
            if i + 1 < len(rots):
                nxt      = rots[i + 1]
                nxt_ls   = _ts(nxt["load_start"])
                nxt_le   = _ts(nxt["load_end"])
                next_mid = (nxt_ls + nxt_le) / 2.0
            else:
                next_mid = None
            windows[(vid, rot["etd"])] = {
                "load_start": ls,
                "load_end":   le,
                "load_mid":   mid,
                "next_mid":   next_mid,
            }
    return windows


class RLStrategy(PlacementStrategy):
    """Linear placement policy trained by Cross-Entropy Method.

    Dynamic ATD via on_event
    ------------------------
    on_event() advances the simulator clock so two new binary features can
    correctly reflect the live vessel schedule state at placement time:

      top_missed_rot : 1 if the top-of-stack container's load window has
                       already closed. It missed its rotation and won't leave
                       until the next one (weeks away). Stacking anything on
                       top of it is risky - penalise such columns.

      c_loading_now  : 1 if the arriving container's own load window is
                       currently open. It needs to be retrieved soon -
                       prefer shallow/accessible stacks.

    Raw ETD is kept unchanged as the departure signal for p_block /
    min_dep_stack so the existing trained feature scale is not disturbed.
    The two new features are separate boolean inputs that CEM can weight
    independently.
    """

    def __init__(self, weights: Optional[List[float]] = None,
                 weights_path: str = _DEFAULT_WEIGHTS_PATH) -> None:
        if weights is None:
            weights = self._load_weights(weights_path)
        self.weights = list(weights)
        self._dep_cache: Dict[str, float] = {}
        self._columns: list = []
        self._max_tiers: dict = {}
        self._init_ids: set = set()   # container_ids present in the initial state

        # Dynamic ATD state
        self._rot_windows = _build_rotation_windows()  # (vid, etd) -> window info
        self._current_ts: float = 0.0                  # updated by on_event

    @staticmethod
    def _load_weights(path: str) -> List[float]:
        try:
            with open(path) as fh:
                data = json.load(fh)
            return [float(data["weights"][f]) for f in FEATURES]
        except (OSError, KeyError, ValueError, TypeError):
            return list(_GREEDY_WEIGHTS)

    # --------------------------------------------------------------------------- setup
    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._columns = []
        self._max_tiers = {}
        n_cols = {}
        for block, info in yard_layout["blocks"].items():
            self._max_tiers[block] = info["tiers"]
            n_cols[block] = info["bays"] * info["rows"]
            for bay in range(1, info["bays"] + 1):
                for row in range(1, info["rows"] + 1):
                    self._columns.append((block, bay, row))
        # Static per-block slot capacity (for the occupancy feature).
        self._cap_block = {b: self._max_tiers[b] * n_cols[b]
                           for b in self._max_tiers}
        self._init_ids = {c["container_id"]
                          for c in initial_state.get("containers", [])}

    # ------------------------------------------------------------------------- helpers
    def _dep(self, iso: str) -> float:
        """Parse ISO timestamp to epoch seconds (cached)."""
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

    def _loading_now(self, vessel_id: str, etd_iso: str) -> bool:
        """True if the load window for (vessel, etd) is currently open."""
        if not vessel_id or not etd_iso:
            return False
        win = self._rot_windows.get((vessel_id, etd_iso))
        if win is None:
            return False
        return win["load_start"] <= self._current_ts <= win["load_end"]

    def _missed_rotation(self, vessel_id: str, etd_iso: str) -> bool:
        """True if the load window for (vessel, etd) has already closed."""
        if not vessel_id or not etd_iso:
            return False
        win = self._rot_windows.get((vessel_id, etd_iso))
        if win is None:
            return False
        return self._current_ts > win["load_end"]

    @staticmethod
    def _p_block(dep_below: float, dep_c: float) -> float:
        if dep_below == float("inf"):
            return 0.0
        if dep_c == float("inf"):
            return 1.0
        x = (dep_c - dep_below) / _TAU_SECONDS
        if x > 30:
            return 1.0
        if x < -30:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    # ------------------------------------------------------------------------ on_event
    def on_event(self, event: Event) -> None:
        """Advance the simulator clock so _dep_for() reflects current time."""
        try:
            self._current_ts = datetime.fromisoformat(event.timestamp).timestamp()
        except (ValueError, AttributeError):
            pass

    # ------------------------------------------------------------------------ decision
    def place_container(self, yard_state: YardState, event: Event) -> Position:
        c = event.to_container()
        dep_c = self._dep(c.departure_time)  # raw ETD - keeps trained feature scale
        c_loading_now = 1.0 if self._loading_now(c.vessel_id, c.departure_time) else 0.0
        w_c = _WEIGHT_RANK.get(c.weight_class, 1)
        w = self.weights

        get_h = yard_state.get_stack_height
        get_at = yard_state.get_container_at
        get_info = yard_state.get_container_info

        # Pass 1: heights, per-block occupancy, and the minimum height.
        heights = {}
        occ = {b: 0 for b in self._max_tiers}
        h_min = 99
        for col in self._columns:
            block, bay, row = col
            h = get_h(block, bay, row)
            heights[col] = h
            occ[block] += h
            if h < self._max_tiers[block] and h < h_min:
                h_min = h
        cap_block = self._cap_block

        best_pos: Optional[Position] = None
        best_cost = float("inf")
        fallback: Optional[Position] = None

        # Pass 2: score candidate columns. Top look-ups only for shallow
        # candidates (height <= h_min + 1) - deeper stacks are never preferred
        # under any non-degenerate weighting, so skipping them is safe and fast.
        for col in self._columns:
            block, bay, row = col
            h = heights[col]
            cap = self._max_tiers[block]
            if h >= cap:
                continue
            if fallback is None or h < heights.get(
                    (fallback.block, fallback.bay, fallback.row), 99):
                fallback = Position(block, bay, row, h + 1)
            if h > h_min + 1:
                continue

            is_empty = 1.0 if h == 0 else 0.0
            p_block = 0.0
            on_initial = 0.0
            batch_violation = 0.0
            min_dep_stack = 0.0
            top_missed_rot = 0.0
            if h > 0:
                top_id = get_at(block, bay, row, h)
                top = get_info(top_id) if top_id else None
                if top is not None:
                    p_block = self._p_block(self._dep(top.departure_time), dep_c)
                    if top_id in self._init_ids:
                        on_initial = 1.0
                    if (top.vessel_id == c.vessel_id
                            and top.departure_time == c.departure_time
                            and top.port_of_discharge == c.port_of_discharge
                            and w_c > _WEIGHT_RANK.get(top.weight_class, 1)):
                        batch_violation = 1.0
                    if self._missed_rotation(top.vessel_id, top.departure_time):
                        top_missed_rot = 1.0
            # Scan the whole stack for the earliest-departing container.
            min_dep = float("inf")
            for tier in range(1, h + 1):
                cid = get_at(block, bay, row, tier)
                if cid:
                    info = get_info(cid)
                    if info:
                        d = self._dep(info.departure_time)
                        if d < min_dep:
                            min_dep = d
            min_dep_stack = self._p_block(min_dep, dep_c)

            feats = (
                1.0,
                h / 5.0,
                is_empty,
                p_block,
                on_initial,
                occ[block] / cap_block[block] if cap_block[block] else 0.0,
                batch_violation,
                1.0 if (h + 1) > _SOFT_CAP else 0.0,
                min_dep_stack,
                top_missed_rot,
                c_loading_now,
            )

            cost = (w[0]*feats[0] + w[1]*feats[1] + w[2]*feats[2]
                    + w[3]*feats[3] + w[4]*feats[4] + w[5]*feats[5]
                    + w[6]*feats[6] + w[7]*feats[7] + w[8]*feats[8]
                    + w[9]*feats[9] + w[10]*feats[10])
            if cost < best_cost:
                best_cost = cost
                best_pos = Position(block, bay, row, h + 1)

        return best_pos if best_pos is not None else fallback