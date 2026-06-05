"""Adapter for external (non-Python) solvers using stdin/stdout JSON protocol.

Protocol:
  1. On INIT: send {"type": "INIT", "yard_layout": {...}, "initial_state": {...}}
     Expect: {"status": "ok"}
  2. On PLACE: send {"type": "PLACE", "event": {...}, "yard_summary": {...}}
     Expect: {"block": "B01", "bay": 1, "row": 1, "tier": 1}
  3. On RETRIEVE: send {"type": "RETRIEVE", "container_id": "...", "reshuffles": N}
     Expect: {"status": "ok"}
  4. On END: send {"type": "END"}
     Process terminates.

The external process is started once and communicates via stdin/stdout.
Each message is a single JSON line (newline-delimited).
"""

import json
import subprocess
import sys
from typing import Optional

from .models import Event, Position
from .yard_state import YardState
from .placement_interface import PlacementStrategy


class ExternalStrategy(PlacementStrategy):
    """Wraps an external solver process."""

    def __init__(self, executable: str):
        self.executable = executable
        self.process: Optional[subprocess.Popen] = None

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        """Start the external process and send initialization data."""
        self.process = subprocess.Popen(
            [self.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )

        self._send({
            "type": "INIT",
            "yard_layout": yard_layout,
            "initial_state": initial_state,
        })
        response = self._recv()
        if response.get("status") != "ok":
            raise RuntimeError(
                f"External solver failed to initialize: {response}")

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        """Send placement request and receive position."""
        # Build a compact yard summary for the solver
        yard_summary = self._build_yard_summary(yard_state)

        self._send({
            "type": "PLACE",
            "event": {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "type": event.type,
                "container_id": event.container_id,
                "size": event.size,
                "weight_class": event.weight_class,
                "vessel_id": event.vessel_id,
                "port_of_discharge": event.port_of_discharge,
                "departure_time": event.departure_time,
            },
            "yard_summary": yard_summary,
        })

        response = self._recv()
        return Position(
            block=response["block"],
            bay=response["bay"],
            row=response["row"],
            tier=response["tier"],
        )

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        """Notify solver of retrieval."""
        self._send({
            "type": "RETRIEVE",
            "container_id": container_id,
            "position": {
                "block": position.block,
                "bay": position.bay,
                "row": position.row,
                "tier": position.tier,
            },
            "reshuffles": reshuffles,
        })
        self._recv()  # Acknowledge

    def on_event(self, event: Event) -> None:
        """We don't send all events to external — too much overhead."""
        pass

    def _build_yard_summary(self, yard_state: YardState) -> dict:
        """Build a compact summary of yard state for the solver."""
        summary = {"blocks": {}}
        for block_name, bi in yard_state.blocks.items():
            occupied, capacity = yard_state.get_block_occupancy(block_name)
            stacks = {}
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h > 0:
                        # Include stack contents for solver
                        stack_contents = []
                        for tier in range(1, h + 1):
                            cid = yard_state.get_container_at(
                                block_name, bay, row, tier)
                            if cid:
                                info = yard_state.get_container_info(cid)
                                stack_contents.append({
                                    "container_id": cid,
                                    "departure_time": info.departure_time if info else "",
                                    "weight_class": info.weight_class if info else "",
                                })
                        stacks[f"{bay},{row}"] = stack_contents

            summary["blocks"][block_name] = {
                "bays": bi.bays,
                "rows": bi.rows,
                "tiers": bi.tiers,
                "occupied": occupied,
                "capacity": capacity,
                "stacks": stacks,
            }
        return summary

    def _send(self, msg: dict):
        """Send a JSON message to the external process."""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("External solver process is not running")
        line = json.dumps(msg) + "\n"
        self.process.stdin.write(line)
        self.process.stdin.flush()

    def _recv(self) -> dict:
        """Receive a JSON response from the external process."""
        if self.process is None:
            raise RuntimeError("External solver process is not running")
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read()
            raise RuntimeError(
                f"External solver closed unexpectedly. stderr: {stderr}")
        return json.loads(line.strip())

    def __del__(self):
        """Clean up the external process."""
        if self.process and self.process.poll() is None:
            try:
                self._send({"type": "END"})
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
