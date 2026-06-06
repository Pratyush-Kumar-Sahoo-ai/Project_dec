"""Tests for the learned (CEM) placement policy."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Container, Event, Position
from src.yard_state import YardState
from solution.rl_strategy import RLStrategy, FEATURES, _GREEDY_WEIGHTS


@pytest.fixture
def layout():
    return {"blocks": {"A": {"bays": 2, "rows": 2, "tiers": 5},
                       "B": {"bays": 2, "rows": 2, "tiers": 5}}}


def _event(cid="E1", weight="MEDIUM", dep="2025-01-10T12:00:00",
           vessel="V1", port="P1"):
    return Event(event_id=1, timestamp="2025-01-01T00:00:00", type="DISCHARGE",
                 container_id=cid, size=20, weight_class=weight,
                 vessel_id=vessel, port_of_discharge=port, departure_time=dep)


def _strategy(layout, init=None, **kw):
    s = RLStrategy(**kw)
    s.initialize(layout, init or {"containers": []})
    return s


class TestWeights:
    def test_fallback_weights_are_greedy(self, layout):
        # Nonexistent path -> falls back to greedy weights.
        s = RLStrategy(weights_path="/no/such/file.json")
        assert s.weights == list(_GREEDY_WEIGHTS)

    def test_feature_order_matches_weight_length(self, layout):
        s = _strategy(layout)
        assert len(s.weights) == len(FEATURES)


class TestValidity:
    def test_returns_valid_positions_across_a_run(self, layout):
        yard = YardState(layout)
        s = _strategy(layout)
        for i in range(35):  # 35 < total capacity (2 blocks * 4 cols * 5 tiers = 40)
            ev = _event(cid=f"E{i}", dep=f"2025-01-{(i % 27) + 1:02d}T00:00:00")
            pos = s.place_container(yard, ev)
            assert yard.is_position_valid(pos), f"invalid at {i}: {pos}"
            yard.place_container(ev.to_container(), pos)

    def test_skips_full_columns(self, layout):
        yard = YardState(layout)
        s = _strategy(layout)
        for t in range(1, 6):  # fill (A,1,1) to cap
            yard.place_container(Container(f"a{t}"), Position("A", 1, 1, t))
        pos = s.place_container(yard, _event("c"))
        assert not (pos.block == "A" and pos.bay == 1 and pos.row == 1)
        assert yard.is_position_valid(pos)


class TestPolicyBehaviour:
    def test_greedy_weights_spread_to_empty(self, layout):
        """With greedy weights the policy opens fresh columns (lowest stack)."""
        yard = YardState(layout)
        s = _strategy(layout, weights=list(_GREEDY_WEIGHTS))
        yard.place_container(Container("a"), Position("A", 1, 1, 1))
        pos = s.place_container(yard, _event("b"))
        assert pos.tier == 1

    def test_on_initial_feature_avoids_initial_containers(self, layout):
        """A large positive 'on_initial' weight should steer placement away from
        stacking on an initial-state container when an equal-height alternative
        exists."""
        init = {"containers": [
            {"container_id": "INIT1",
             "position": {"block": "A", "bay": 1, "row": 1, "tier": 1},
             "departure_time": "2025-03-01T00:00:00"}]}
        yard = YardState(layout)
        yard.load_initial_state(init)
        # Put a non-initial container at (A,1,2) tier1 as the alternative top.
        yard.place_container(
            Container("own", departure_time="2025-03-01T00:00:00"),
            Position("A", 1, 2, 1))
        # Fill remaining columns so the only height-1 candidates are the two above.
        for col in [("A", 2, 1), ("A", 2, 2), ("B", 1, 1), ("B", 1, 2),
                    ("B", 2, 1), ("B", 2, 2)]:
            yard.place_container(Container(f"f{col}"), Position(*col, 1))
        # Weights: penalise depth lightly, heavily penalise on_initial.
        w = [0.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
        s = _strategy(layout, init=init, weights=w)
        new = _event("new", dep="2025-01-05T00:00:00")
        pos = s.place_container(yard, new)
        assert (pos.block, pos.bay, pos.row) == ("A", 1, 2)  # avoided INIT1
