"""Core data models for the container yard placement test."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    """A position in the yard: block, bay, row, tier."""
    block: str
    bay: int
    row: int
    tier: int

    def __repr__(self):
        return f"({self.block}, bay={self.bay}, row={self.row}, tier={self.tier})"


@dataclass
class Container:
    """A container with its attributes."""
    container_id: str
    size: int = 20  # Always 20ft in this test
    weight_class: str = "MEDIUM"  # LIGHT, MEDIUM, HEAVY
    vessel_id: str = ""
    port_of_discharge: str = ""
    departure_time: str = ""  # ISO-8601 timestamp


@dataclass
class Event:
    """A single event in the event stream."""
    event_id: int
    timestamp: str  # ISO-8601
    type: str  # DISCHARGE, LOAD, TRUCK_RECV, TRUCK_DLVR
    container_id: str
    # Only populated for DISCHARGE/TRUCK_RECV (placement events):
    size: int = 20
    weight_class: str = "MEDIUM"
    vessel_id: str = ""
    port_of_discharge: str = ""
    departure_time: str = ""

    def to_container(self) -> Container:
        """Convert event to a Container object."""
        return Container(
            container_id=self.container_id,
            size=self.size,
            weight_class=self.weight_class,
            vessel_id=self.vessel_id,
            port_of_discharge=self.port_of_discharge,
            departure_time=self.departure_time,
        )


@dataclass
class BlockInfo:
    """Metadata for a yard block."""
    name: str
    bays: int
    rows: int
    tiers: int

    @property
    def capacity(self) -> int:
        """Total number of container slots."""
        return self.bays * self.rows * self.tiers
