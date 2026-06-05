"""Main simulation loop: processes events, calls placement strategy, counts reshuffles."""

import json
import random
import time
from typing import List, Optional

from .models import Container, Event, Position
from .yard_state import YardState
from .placement_interface import PlacementStrategy
from .scoring import SimulationStats


class Simulator:
    """Runs the event stream through a placement strategy and tracks results."""

    def __init__(self, yard_state: YardState, strategy: PlacementStrategy,
                 verbose: bool = False):
        self.yard = yard_state
        self.strategy = strategy
        self.verbose = verbose
        self.stats = SimulationStats()

    def run(self, events: List[Event]) -> SimulationStats:
        """Process all events and return statistics.

        Args:
            events: List of events in chronological order.

        Returns:
            SimulationStats with all metrics.
        """
        start_time = time.time()

        for event in events:
            self.stats.total_events += 1

            # Notify strategy of every event
            self.strategy.on_event(event)

            if event.type in ("DISCHARGE", "TRUCK_RECV"):
                self._handle_placement(event)
            elif event.type in ("LOAD", "TRUCK_DLVR"):
                self._handle_retrieval(event)

            if self.verbose and self.stats.total_events % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {self.stats.total_events} events "
                      f"({elapsed:.1f}s) | "
                      f"reshuffles={self.stats.total_reshuffles} | "
                      f"yard={self.yard.total_containers()}")

        self.stats.elapsed_sec = time.time() - start_time
        self.stats.final_yard_count = self.yard.total_containers()
        return self.stats

    def _handle_placement(self, event: Event):
        """Handle a DISCHARGE or TRUCK_RECV event."""
        self.stats.placement_events += 1
        container = event.to_container()

        # Ask strategy for placement decision
        try:
            position = self.strategy.place_container(self.yard, event)
        except Exception as e:
            self.stats.errors += 1
            if self.verbose:
                print(f"  ERROR: Strategy raised {e} for {event.container_id}")
            # Fallback: find any valid position
            position = self._fallback_placement()
            if position is None:
                self.stats.placement_failures += 1
                return

        # Validate and place
        if not self.yard.is_position_valid(position):
            self.stats.hard_constraint_violations += 1
            if self.verbose:
                print(f"  VIOLATION: Invalid position {position} for "
                      f"{event.container_id}")
            # Try fallback
            position = self._fallback_placement()
            if position is None:
                self.stats.placement_failures += 1
                return

        success = self.yard.place_container(container, position)
        if not success:
            self.stats.hard_constraint_violations += 1
            self.stats.placement_failures += 1

    def _handle_retrieval(self, event: Event):
        """Handle a LOAD or TRUCK_DLVR event."""
        self.stats.retrieval_events += 1

        pos = self.yard.get_container_position(event.container_id)
        if pos is None:
            # Container not in yard — skip (might have been removed already)
            self.stats.retrieval_not_found += 1
            return

        # Count reshuffles: containers above the target
        above = self.yard.get_containers_above(event.container_id)
        reshuffles = len(above)
        self.stats.total_reshuffles += reshuffles
        if reshuffles > 0:
            self.stats.retrievals_with_reshuffles += 1
        self.stats.reshuffle_counts.append(reshuffles)

        # Perform the retrieval: temporarily relocate containers above,
        # remove target, put them back
        self._do_retrieval(event.container_id, above)

        # Notify strategy
        self.strategy.on_container_retrieved(
            event.container_id, pos, reshuffles)

    def _do_retrieval(self, target_id: str, above: List[str]):
        """Physically perform the retrieval with reshuffles.

        Containers above the target are moved to temporary positions
        (lowest available stack in same block), target is removed,
        then reshuffled containers are placed back.
        """
        if not above:
            # No reshuffles needed — just remove
            self.yard.remove_container(target_id)
            return

        # Get target's block
        pos = self.yard.get_container_position(target_id)
        if pos is None:
            return
        block = pos.block

        # Remove containers above (top to bottom)
        temp_containers = []
        for cid in reversed(above):
            info = self.yard.get_container_info(cid)
            removed_pos = self.yard.remove_container(cid)
            if removed_pos and info:
                temp_containers.append((cid, info))

        # Remove target
        self.yard.remove_container(target_id)

        # Put reshuffled containers back (find lowest available stacks)
        for cid, cont in reversed(temp_containers):
            reposition = self._find_reshuffle_position(block, cont)
            if reposition:
                self.yard.place_container(cont, reposition)

    def _find_reshuffle_position(self, block: str,
                                  container: Container) -> Optional[Position]:
        """Find a position for a reshuffled container.

        Prefers same block, lowest stack height.
        """
        bi = self.yard.blocks.get(block)
        if not bi:
            return None

        best_pos = None
        best_height = bi.tiers + 1

        for bay in range(1, bi.bays + 1):
            for row in range(1, bi.rows + 1):
                h = self.yard.get_stack_height(block, bay, row)
                if h < bi.tiers and h < best_height:
                    best_height = h
                    best_pos = Position(block, bay, row, h + 1)

        # If current block is full, try other blocks
        if best_pos is None:
            for bname, binfo in self.yard.blocks.items():
                if bname == block:
                    continue
                for bay in range(1, binfo.bays + 1):
                    for row in range(1, binfo.rows + 1):
                        h = self.yard.get_stack_height(bname, bay, row)
                        if h < binfo.tiers and h < best_height:
                            best_height = h
                            best_pos = Position(bname, bay, row, h + 1)
                    if best_pos is not None and best_height <= 1:
                        break

        return best_pos

    def _fallback_placement(self) -> Optional[Position]:
        """Find any valid position (used when strategy fails)."""
        for bname, bi in self.yard.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = self.yard.get_stack_height(bname, bay, row)
                    if h < bi.tiers:
                        return Position(bname, bay, row, h + 1)
        return None
