"""Capacity-Aware Expected Reshuffle Index (ERI) placement strategy.

Implements and adapts the index-based stacking heuristic of

    H. Bisira & A. Salhi (2021), "Reshuffle minimisation to improve storage
    yard operations efficiency", Journal of Marine Engineering & Technology,

building on the "expected number of reshuffles" stacking literature
(Kim & Hong 2006; Galle, Barnhart & Jaillet 2018, *The Stochastic Container
Relocation Problem*), adapted to the **online, dynamic** terminal here.

================================================================================
The index
================================================================================
A reshuffle occurs when a container is buried under one that is retrieved
*before* it. Placing an arriving container ``c`` on top of a stack makes every
container already in that stack that leaves *before* ``c`` a future reshuffle.
The Expected Reshuffle Index assigns ``c`` to the stack minimising

    ERI(c, stack) = ORDER_W * E[blocking]            # classic ERI term
                  + BETA * height(stack)             # burial / relocation risk

where the expected blocking is learned from the training split as a calibrated
function of the departure-time gap (the only leave-time signal available):

    P( leave(b) < leave(c) ) = sigmoid( (dep_c - dep_b) / TAU ),   TAU ≈ 3.7 d

(at a 0-gap P≈0.5; a container departing a week later still leaves first ~13%
of the time — the signal is real but weak, Spearman ρ≈0.55 with realised
retrieval order).

================================================================================
Why the deployed default suppresses the ordering term  (ORDER_W = 0)
================================================================================
Extensive experiments on the train split (see docs/design.md) show that on
THIS instance any use of the departure signal *increases* reshuffles, both
in- and out-of-sample:

    pure ERI consolidation (best-fit by departure) ....... 0.89
    confident consolidation (safety margin) .............. 0.97–1.09
    (vessel, port[, ETD]) grouping + weight order ........ 0.93
    spatial zoning by departure bucket ................... 0.86
    height-dominant + ERI blocking tie-break ............. 0.83
    lowest-stack spreading (ORDER_W = 0, deployed) ....... 0.78  <-- best
    perfect-information upper bound (needs true leaves) .. 0.48

Two compounding causes: (1) ρ≈0.55 predictability means residual ordering
errors are common; (2) the simulator re-scatters reshuffled containers onto
the lowest available stacks, continually re-creating disorder we cannot
control. Consolidation amplifies (1) inside tall stacks into cascades, while
spreading keeps every stack short and noise-robust.

The capacity-aware ERI captures this directly: the burial term ``BETA*height``
is itself an expected future relocation (every container under ``c`` may need
moving later). In the limit where it dominates the unreliable ordering term
(ORDER_W → 0) the index reduces *exactly* to lowest-stack spreading — which the
analysis identifies as the online optimum and which we therefore deploy. The
full ERI machinery is retained and configurable (``ordering_weight``) so the
ordering term can be re-enabled on instances with a stronger departure signal.

================================================================================
Complexity
================================================================================
O(C) per placement (C = number of columns ≈ 1920), O(1) per column. Departure
timestamps are cached. With the default ORDER_W = 0 the per-column work is a
single stack-height read.
"""

import math
from datetime import datetime
from typing import Optional

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState


_WEIGHT_RANK = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}

# Parameters calibrated on the train split (see docs/design.md).
_TAU_SECONDS = 3.7 * 86400.0   # logistic scale of the blocking probability
_BETA = 0.15                   # burial-depth penalty per occupied tier (spreads)


class ERIStrategy(PlacementStrategy):
    """Capacity-aware Expected Reshuffle Index stacking.

    Args:
        ordering_weight: weight on the (departure-based) expected-blocking term.
            0.0 (default) deploys the capacity-dominant spreading limit, which
            the train-set study shows is the online optimum on this instance.
            Set > 0 to re-enable ERI consolidation.
    """

    def __init__(self, ordering_weight: float = 0.0) -> None:
        self.ordering_weight = ordering_weight
        self._dep_cache = {}     # iso string -> epoch seconds
        self._columns = []       # list[(block, bay, row)]
        self._max_tiers = {}     # block -> tier cap

    # ----------------------------------------------------------------- setup
    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._columns = []
        self._max_tiers = {}
        for block, info in yard_layout["blocks"].items():
            self._max_tiers[block] = info["tiers"]
            for bay in range(1, info["bays"] + 1):
                for row in range(1, info["rows"] + 1):
                    self._columns.append((block, bay, row))

    # --------------------------------------------------------------- helpers
    def _dep(self, iso: str) -> float:
        """Departure-time epoch seconds (cached). Missing -> +inf (leaves last)."""
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

    @staticmethod
    def _p_block(dep_below: float, dep_c: float) -> float:
        """Learned P(container below leaves before the arriving container)."""
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

    def _expected_blocking(self, yard: YardState, block: str, bay: int, row: int,
                           height: int, dep_c: float, c: "Event") -> float:
        """E[number of containers in the stack that leave before c]."""
        if height == 0:
            return 0.0
        cost = 0.0
        w_c = _WEIGHT_RANK.get(c.weight_class, 1)
        for tier in range(1, height + 1):
            cid = yard.get_container_at(block, bay, row, tier)
            info = yard.get_container_info(cid) if cid else None
            if info is None:
                continue
            # Same load batch (vessel+ETD+port): retrieval order is by weight
            # (heavy first). Treat as blocking iff c would sit above a heavier one.
            if (info.vessel_id == c.vessel_id
                    and info.departure_time == c.departure_time
                    and info.port_of_discharge == c.port_of_discharge):
                cost += 1.0 if w_c > _WEIGHT_RANK.get(info.weight_class, 1) else 0.0
            else:
                cost += self._p_block(self._dep(info.departure_time), dep_c)
        return cost

    # -------------------------------------------------------------- decision
    def place_container(self, yard_state: YardState, event: Event) -> Position:
        c = event.to_container()
        dep_c = self._dep(c.departure_time)
        order_w = self.ordering_weight

        best_pos: Optional[Position] = None
        best_eri = float("inf")
        get_h = yard_state.get_stack_height

        for block, bay, row in self._columns:
            h = get_h(block, bay, row)
            cap = self._max_tiers[block]
            if h >= cap:
                continue

            # Capacity term: burial-depth risk. Strictly increasing in height,
            # so with order_w=0 the index is minimised by the lowest stack
            # (robust spreading); ties broken by scan order across blocks.
            eri = _BETA * h
            # Ordering term: expected blocking (suppressed by default, order_w=0).
            if order_w and h > 0:
                eri += order_w * self._expected_blocking(
                    yard_state, block, bay, row, h, dep_c, c)

            if eri < best_eri:
                best_eri = eri
                best_pos = Position(block, bay, row, h + 1)

        if best_pos is None:
            # Yard saturated — return any in-range slot; the simulator's fallback
            # will take over. (Does not occur at this instance's occupancy.)
            block, bay, row = self._columns[0]
            return Position(block, bay, row, self._max_tiers[block])
        return best_pos
