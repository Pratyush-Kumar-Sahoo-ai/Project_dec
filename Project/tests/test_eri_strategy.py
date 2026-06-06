"""Tests for the Capacity-Aware ERI placement strategy."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Container, Event, Position
from src.yard_state import YardState
from solution.eri_strategy import ERIStrategy


@pytest.fixture
def layout():
    return {
        "blocks": {
            "A": {"bays": 2, "rows": 2, "tiers": 5},
            "B": {"bays": 2, "rows": 2, "tiers": 5},
        }
    }


@pytest.fixture
def yard(layout):
    return YardState(layout)


def _event(cid="E1", weight="MEDIUM", dep="2025-01-10T12:00:00",
           vessel="V1", port="P1", etype="DISCHARGE"):
    return Event(event_id=1, timestamp="2025-01-01T00:00:00", type=etype,
                 container_id=cid, size=20, weight_class=weight,
                 vessel_id=vessel, port_of_discharge=port, departure_time=dep)


def _strategy(layout, **kw):
    s = ERIStrategy(**kw)
    s.initialize(layout, {"containers": []})
    return s


# --------------------------------------------------------------- validity
class TestValidity:
    def test_returns_valid_position_on_empty_yard(self, yard, layout):
        s = _strategy(layout)
        pos = s.place_container(yard, _event())
        assert yard.is_position_valid(pos)
        assert pos.tier == 1  # empty column

    def test_every_placement_is_valid_across_a_run(self, yard, layout):
        """No hard-constraint violation over many sequential placements."""
        s = _strategy(layout)
        for i in range(40):  # 40 < total capacity (2 blocks * 4 cols * 5 tiers)
            ev = _event(cid=f"E{i}", dep=f"2025-01-{(i % 27) + 1:02d}T00:00:00")
            pos = s.place_container(yard, ev)
            assert yard.is_position_valid(pos), f"invalid at step {i}: {pos}"
            yard.place_container(ev.to_container(), pos)

    def test_picks_lowest_available_tier(self, yard, layout):
        """Default (ordering_weight=0) is lowest-stack spreading."""
        s = _strategy(layout)
        # Fill every column to height 1 except leave one taller to confirm
        # the strategy never prefers a taller stack.
        for col in [("A", 1, 1), ("A", 1, 2), ("A", 2, 1), ("A", 2, 2),
                    ("B", 1, 1), ("B", 1, 2), ("B", 2, 1)]:
            yard.place_container(Container(f"x{col}"), Position(*col, 1))
        # Only ("B",2,2) is empty -> lowest stack is height 0 there.
        pos = s.place_container(yard, _event("new"))
        assert pos.tier == 1
        assert (pos.block, pos.bay, pos.row) == ("B", 2, 2)


# --------------------------------------------------------------- spreading
class TestSpreading:
    def test_default_prefers_empty_over_occupied(self, yard, layout):
        s = _strategy(layout)
        yard.place_container(Container("a"), Position("A", 1, 1, 1))
        pos = s.place_container(yard, _event("b"))
        assert pos.tier == 1  # opens a fresh column rather than stacking

    def test_saturated_block_is_skipped(self, yard, layout):
        s = _strategy(layout)
        # Fill block A column (A,1,1) to the tier cap.
        for t in range(1, 6):
            yard.place_container(Container(f"a{t}"), Position("A", 1, 1, t))
        pos = s.place_container(yard, _event("c"))
        # Must not exceed the cap of the full column.
        assert not (pos.block == "A" and pos.bay == 1 and pos.row == 1)
        assert yard.is_position_valid(pos)


# --------------------------------------------------------------- ordering term
class TestOrderingTerm:
    def test_blocking_probability_monotonic(self):
        s = ERIStrategy()
        # container below departs much earlier -> high block prob
        early = s._p_block(dep_below=0.0, dep_c=10 * 86400.0)
        # below departs much later -> low block prob
        late = s._p_block(dep_below=10 * 86400.0, dep_c=0.0)
        same = s._p_block(dep_below=0.0, dep_c=0.0)
        assert early > 0.9
        assert late < 0.1
        assert abs(same - 0.5) < 1e-6

    def test_unknown_departure_treated_as_leaves_last(self):
        s = ERIStrategy()
        inf = float("inf")
        # below never leaves -> cannot block
        assert s._p_block(dep_below=inf, dep_c=0.0) == 0.0
        # arriving never leaves -> everything below blocks it
        assert s._p_block(dep_below=0.0, dep_c=inf) == 1.0

    def test_ordering_weight_prefers_safe_stack(self, yard, layout):
        """With ordering on and no empty column left, the new container is
        placed on the SAFE stack (top departs later -> blocks nobody) rather
        than an unsafe one (top departs earlier -> would be buried)."""
        s = _strategy(layout, ordering_weight=5.0)
        cols = [("A", 1, 1), ("A", 1, 2), ("A", 2, 1), ("A", 2, 2),
                ("B", 1, 1), ("B", 1, 2), ("B", 2, 1), ("B", 2, 2)]
        safe_col = ("B", 2, 2)
        # Fill every column to height 1. All tops depart EARLIER than the new
        # container except the designated safe column, whose top departs later.
        for col in cols:
            dep = ("2025-02-01T00:00:00" if col == safe_col
                   else "2025-01-02T00:00:00")
            yard.place_container(Container(f"t{col}", departure_time=dep),
                                 Position(*col, 1))
        new = _event("mid", dep="2025-01-15T00:00:00")
        pos = s.place_container(yard, new)
        assert pos.tier == 2
        assert (pos.block, pos.bay, pos.row) == safe_col
