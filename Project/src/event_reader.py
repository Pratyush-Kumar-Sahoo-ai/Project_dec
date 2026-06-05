"""Event reader: loads and iterates over the event stream."""

import json
from typing import Iterator, List

from .models import Event


def read_events(filepath: str) -> List[Event]:
    """Read all events from a JSONL file.

    Args:
        filepath: Path to events.jsonl

    Returns:
        List of Event objects in chronological order.
    """
    events = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            events.append(Event(
                event_id=d["event_id"],
                timestamp=d["timestamp"],
                type=d["type"],
                container_id=d["container_id"],
                size=d.get("size", 20),
                weight_class=d.get("weight_class", "MEDIUM"),
                vessel_id=d.get("vessel_id", ""),
                port_of_discharge=d.get("port_of_discharge", ""),
                departure_time=d.get("departure_time", ""),
            ))
    return events


def iter_events(filepath: str) -> Iterator[Event]:
    """Iterate over events from a JSONL file (memory-efficient).

    Args:
        filepath: Path to events.jsonl

    Yields:
        Event objects in chronological order.
    """
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield Event(
                event_id=d["event_id"],
                timestamp=d["timestamp"],
                type=d["type"],
                container_id=d["container_id"],
                size=d.get("size", 20),
                weight_class=d.get("weight_class", "MEDIUM"),
                vessel_id=d.get("vessel_id", ""),
                port_of_discharge=d.get("port_of_discharge", ""),
                departure_time=d.get("departure_time", ""),
            )
