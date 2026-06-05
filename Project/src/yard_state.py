"""Yard state manager: 3D grid tracking container positions.

This is provided to candidates as part of the scaffold. It handles all
yard state bookkeeping — candidates only need to query it and return
placement decisions.
"""

import copy
import json
from typing import Dict, List, Optional, Tuple

from .models import BlockInfo, Container, Position


class YardState:
    """Manages the 3D grid of container positions across all blocks."""

    def __init__(self, yard_layout: dict):
        """Initialize from yard layout JSON.

        Args:
            yard_layout: Dict with 'blocks' key mapping block names to
                         {bays, rows, tiers} dicts.
        """
        self.blocks: Dict[str, BlockInfo] = {}
        # grid[block][bay][row] = list of container_ids from tier 1 up
        self._grid: Dict[str, List[List[List[str]]]] = {}
        # container_id -> (Position, Container)
        self._containers: Dict[str, Tuple[Position, Container]] = {}

        for name, info in yard_layout["blocks"].items():
            bi = BlockInfo(
                name=name,
                bays=info["bays"],
                rows=info["rows"],
                tiers=info["tiers"],
            )
            self.blocks[name] = bi
            # bays are 1-indexed, rows are 1-indexed
            self._grid[name] = [
                [[] for _ in range(bi.rows + 1)]  # row 0 unused
                for _ in range(bi.bays + 1)  # bay 0 unused
            ]

    def load_initial_state(self, initial_state: dict):
        """Load containers from initial state JSON.

        Args:
            initial_state: Dict with 'containers' list, each having
                           container_id, position {block, bay, row, tier},
                           and container attributes.
        """
        for entry in initial_state.get("containers", []):
            pos = Position(
                block=entry["position"]["block"],
                bay=entry["position"]["bay"],
                row=entry["position"]["row"],
                tier=entry["position"]["tier"],
            )
            container = Container(
                container_id=entry["container_id"],
                size=entry.get("size", 20),
                weight_class=entry.get("weight_class", "MEDIUM"),
                vessel_id=entry.get("vessel_id", ""),
                port_of_discharge=entry.get("port_of_discharge", ""),
                departure_time=entry.get("departure_time", ""),
            )
            # Place directly without validation (initial state is trusted)
            stack = self._grid[pos.block][pos.bay][pos.row]
            while len(stack) < pos.tier:
                stack.append("")  # pad if tiers are sparse
            if pos.tier <= len(stack):
                stack[pos.tier - 1] = container.container_id
            else:
                stack.append(container.container_id)
            self._containers[container.container_id] = (pos, container)

    def place_container(self, container: Container, position: Position) -> bool:
        """Place a container at the given position.

        Args:
            container: The container to place.
            position: Target position.

        Returns:
            True if placed successfully, False if invalid.
        """
        if not self.is_position_valid(position):
            return False

        stack = self._grid[position.block][position.bay][position.row]
        expected_tier = len(stack) + 1
        if position.tier != expected_tier:
            return False

        stack.append(container.container_id)
        self._containers[container.container_id] = (position, container)
        return True

    def remove_container(self, container_id: str) -> Optional[Position]:
        """Remove a container from the yard.

        The container must be at the top of its stack.

        Returns:
            The position it was at, or None if not found/not on top.
        """
        if container_id not in self._containers:
            return None

        pos, _ = self._containers[container_id]
        stack = self._grid[pos.block][pos.bay][pos.row]

        if not stack or stack[-1] != container_id:
            return None

        stack.pop()
        del self._containers[container_id]
        return pos

    def force_remove_container(self, container_id: str) -> Optional[Position]:
        """Remove a container regardless of position in stack.

        Used internally by the simulator for reshuffle handling.

        Returns:
            The position it was at, or None if not found.
        """
        if container_id not in self._containers:
            return None

        pos, _ = self._containers[container_id]
        stack = self._grid[pos.block][pos.bay][pos.row]

        # Find and remove from stack
        try:
            idx = stack.index(container_id)
            stack.pop(idx)
            # Renumber tiers for containers above
            for i in range(idx, len(stack)):
                cid = stack[i]
                if cid in self._containers:
                    old_pos, cont = self._containers[cid]
                    new_pos = Position(old_pos.block, old_pos.bay, old_pos.row,
                                       i + 1)
                    self._containers[cid] = (new_pos, cont)
        except ValueError:
            return None

        del self._containers[container_id]
        return pos

    def get_containers_above(self, container_id: str) -> List[str]:
        """Get list of container IDs stacked above the given container.

        This is how reshuffles are counted: each container above the target
        must be moved before the target can be retrieved.

        Returns:
            List of container_ids above the target (bottom to top order),
            or empty list if container not found.
        """
        if container_id not in self._containers:
            return []

        pos, _ = self._containers[container_id]
        stack = self._grid[pos.block][pos.bay][pos.row]

        try:
            idx = stack.index(container_id)
            return list(stack[idx + 1:])
        except ValueError:
            return []

    def get_stack_height(self, block: str, bay: int, row: int) -> int:
        """Get the current stack height at a position."""
        if block not in self._grid:
            return 0
        bi = self.blocks[block]
        if bay < 1 or bay > bi.bays or row < 1 or row > bi.rows:
            return 0
        return len(self._grid[block][bay][row])

    def get_container_at(self, block: str, bay: int, row: int,
                          tier: int) -> Optional[str]:
        """Get container ID at a specific position, or None."""
        if block not in self._grid:
            return None
        bi = self.blocks[block]
        if bay < 1 or bay > bi.bays or row < 1 or row > bi.rows:
            return None
        stack = self._grid[block][bay][row]
        if tier < 1 or tier > len(stack):
            return None
        return stack[tier - 1]

    def get_container_position(self, container_id: str) -> Optional[Position]:
        """Get the current position of a container."""
        if container_id not in self._containers:
            return None
        return self._containers[container_id][0]

    def get_container_info(self, container_id: str) -> Optional[Container]:
        """Get the Container object for a given ID."""
        if container_id not in self._containers:
            return None
        return self._containers[container_id][1]

    def is_position_valid(self, position: Position) -> bool:
        """Check if a position satisfies all hard constraints.

        Hard constraints:
        1. Block exists
        2. Bay/row within block dimensions
        3. Tier does not exceed max_tiers
        4. Tier is exactly stack_height + 1 (no gaps)
        5. Slot is not occupied
        """
        if position.block not in self.blocks:
            return False
        bi = self.blocks[position.block]
        if position.bay < 1 or position.bay > bi.bays:
            return False
        if position.row < 1 or position.row > bi.rows:
            return False
        if position.tier < 1 or position.tier > bi.tiers:
            return False

        stack_height = self.get_stack_height(
            position.block, position.bay, position.row)
        if position.tier != stack_height + 1:
            return False

        return True

    def get_block_occupancy(self, block: str) -> Tuple[int, int]:
        """Get (occupied_slots, total_capacity) for a block."""
        if block not in self.blocks:
            return (0, 0)
        bi = self.blocks[block]
        occupied = 0
        for bay in range(1, bi.bays + 1):
            for row in range(1, bi.rows + 1):
                occupied += len(self._grid[block][bay][row])
        return (occupied, bi.capacity)

    def get_all_containers(self) -> Dict[str, Tuple[Position, Container]]:
        """Get all containers currently in the yard."""
        return dict(self._containers)

    def get_containers_by_vessel(self, vessel_id: str) -> List[str]:
        """Get all container IDs associated with a vessel."""
        return [cid for cid, (_, cont) in self._containers.items()
                if cont.vessel_id == vessel_id]

    def total_containers(self) -> int:
        """Total number of containers in the yard."""
        return len(self._containers)

    def snapshot(self) -> dict:
        """Create a snapshot of the current yard state.

        Use this for lookahead, what-if analysis, or solver-based
        approaches. Call restore() to revert to a snapshot.

        Returns:
            An opaque snapshot dict.
        """
        return {
            "grid": {
                block: [
                    [list(row) for row in bay_rows]
                    for bay_rows in bays
                ]
                for block, bays in self._grid.items()
            },
            "containers": {
                cid: (
                    Position(pos.block, pos.bay, pos.row, pos.tier),
                    Container(
                        container_id=cont.container_id,
                        size=cont.size,
                        weight_class=cont.weight_class,
                        vessel_id=cont.vessel_id,
                        port_of_discharge=cont.port_of_discharge,
                        departure_time=cont.departure_time,
                    ),
                )
                for cid, (pos, cont) in self._containers.items()
            },
        }

    def restore(self, snapshot: dict):
        """Restore yard state from a snapshot."""
        self._grid = {
            block: [
                [list(row) for row in bay_rows]
                for bay_rows in bays
            ]
            for block, bays in snapshot["grid"].items()
        }
        self._containers = {
            cid: (
                Position(pos.block, pos.bay, pos.row, pos.tier),
                Container(
                    container_id=cont.container_id,
                    size=cont.size,
                    weight_class=cont.weight_class,
                    vessel_id=cont.vessel_id,
                    port_of_discharge=cont.port_of_discharge,
                    departure_time=cont.departure_time,
                ),
            )
            for cid, (pos, cont) in snapshot["containers"].items()
        }

    def export_state(self) -> dict:
        """Export current state as JSON-serializable dict.

        Used to generate initial_state.json for the test split.
        """
        containers = []
        for cid, (pos, cont) in self._containers.items():
            containers.append({
                "container_id": cid,
                "position": {
                    "block": pos.block,
                    "bay": pos.bay,
                    "row": pos.row,
                    "tier": pos.tier,
                },
                "size": cont.size,
                "weight_class": cont.weight_class,
                "vessel_id": cont.vessel_id,
                "port_of_discharge": cont.port_of_discharge,
                "departure_time": cont.departure_time,
            })
        return {"containers": containers}
