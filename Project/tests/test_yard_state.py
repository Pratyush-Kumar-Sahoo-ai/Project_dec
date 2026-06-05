"""Tests for YardState."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Container, Position, BlockInfo
from src.yard_state import YardState


@pytest.fixture
def simple_layout():
    return {
        "blocks": {
            "A": {"bays": 3, "rows": 2, "tiers": 5},
            "B": {"bays": 2, "rows": 2, "tiers": 5},
        }
    }


@pytest.fixture
def yard(simple_layout):
    return YardState(simple_layout)


def _make_container(cid="C001"):
    return Container(container_id=cid, size=20, weight_class="MEDIUM",
                     vessel_id="V1", port_of_discharge="SGSIN",
                     departure_time="2025-01-10T12:00:00")


class TestPlacement:
    def test_place_on_empty_stack(self, yard):
        c = _make_container("C001")
        pos = Position("A", 1, 1, 1)
        assert yard.place_container(c, pos) is True
        assert yard.get_stack_height("A", 1, 1) == 1
        assert yard.get_container_at("A", 1, 1, 1) == "C001"

    def test_place_stacking(self, yard):
        c1 = _make_container("C001")
        c2 = _make_container("C002")
        yard.place_container(c1, Position("A", 1, 1, 1))
        yard.place_container(c2, Position("A", 1, 1, 2))
        assert yard.get_stack_height("A", 1, 1) == 2
        assert yard.get_container_at("A", 1, 1, 2) == "C002"

    def test_place_wrong_tier_rejected(self, yard):
        c = _make_container("C001")
        # Tier 2 on empty stack should fail
        assert yard.place_container(c, Position("A", 1, 1, 2)) is False

    def test_place_over_max_tiers(self, yard):
        for i in range(1, 6):
            c = _make_container(f"C{i:03d}")
            yard.place_container(c, Position("A", 1, 1, i))
        # Tier 6 should fail (max is 5)
        c6 = _make_container("C006")
        assert yard.place_container(c6, Position("A", 1, 1, 6)) is False

    def test_place_invalid_block(self, yard):
        c = _make_container("C001")
        assert yard.place_container(c, Position("Z", 1, 1, 1)) is False

    def test_place_invalid_bay(self, yard):
        c = _make_container("C001")
        assert yard.place_container(c, Position("A", 99, 1, 1)) is False


class TestRemoval:
    def test_remove_top(self, yard):
        c = _make_container("C001")
        yard.place_container(c, Position("A", 1, 1, 1))
        pos = yard.remove_container("C001")
        assert pos is not None
        assert pos.block == "A"
        assert yard.get_stack_height("A", 1, 1) == 0

    def test_remove_not_on_top_fails(self, yard):
        c1 = _make_container("C001")
        c2 = _make_container("C002")
        yard.place_container(c1, Position("A", 1, 1, 1))
        yard.place_container(c2, Position("A", 1, 1, 2))
        # C001 is not on top
        assert yard.remove_container("C001") is None

    def test_remove_nonexistent(self, yard):
        assert yard.remove_container("FAKE") is None


class TestContainersAbove:
    def test_above_single(self, yard):
        for i in range(1, 4):
            c = _make_container(f"C{i:03d}")
            yard.place_container(c, Position("A", 1, 1, i))
        above = yard.get_containers_above("C001")
        assert above == ["C002", "C003"]

    def test_above_top_container(self, yard):
        for i in range(1, 3):
            c = _make_container(f"C{i:03d}")
            yard.place_container(c, Position("A", 1, 1, i))
        assert yard.get_containers_above("C002") == []

    def test_above_nonexistent(self, yard):
        assert yard.get_containers_above("FAKE") == []


class TestOccupancy:
    def test_block_occupancy(self, yard):
        occupied, capacity = yard.get_block_occupancy("A")
        assert occupied == 0
        assert capacity == 3 * 2 * 5  # bays * rows * tiers

        c = _make_container("C001")
        yard.place_container(c, Position("A", 1, 1, 1))
        occupied, _ = yard.get_block_occupancy("A")
        assert occupied == 1

    def test_total_containers(self, yard):
        assert yard.total_containers() == 0
        c = _make_container("C001")
        yard.place_container(c, Position("A", 1, 1, 1))
        assert yard.total_containers() == 1


class TestSnapshotRestore:
    def test_snapshot_restore(self, yard):
        c = _make_container("C001")
        yard.place_container(c, Position("A", 1, 1, 1))

        snap = yard.snapshot()
        assert yard.total_containers() == 1

        # Add another container
        c2 = _make_container("C002")
        yard.place_container(c2, Position("A", 1, 1, 2))
        assert yard.total_containers() == 2

        # Restore
        yard.restore(snap)
        assert yard.total_containers() == 1
        assert yard.get_container_at("A", 1, 1, 1) == "C001"
        assert yard.get_container_at("A", 1, 1, 2) is None


class TestValidation:
    def test_valid_position(self, yard):
        pos = Position("A", 1, 1, 1)
        assert yard.is_position_valid(pos) is True

    def test_invalid_block(self, yard):
        assert yard.is_position_valid(Position("Z", 1, 1, 1)) is False

    def test_invalid_bay(self, yard):
        assert yard.is_position_valid(Position("A", 0, 1, 1)) is False
        assert yard.is_position_valid(Position("A", 99, 1, 1)) is False

    def test_invalid_tier_gap(self, yard):
        # Tier 2 on empty stack
        assert yard.is_position_valid(Position("A", 1, 1, 2)) is False
