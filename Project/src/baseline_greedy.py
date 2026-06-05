"""Baseline: greedy lowest-stack placement.

Places each container on the lowest available stack. This is a reasonable
default and serves as the reference target for candidates to beat.
"""

from .models import Event, Position
from .yard_state import YardState
from .placement_interface import PlacementStrategy


class GreedyStrategy(PlacementStrategy):
    """Places containers on the lowest available stack (first-fit)."""

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        pass

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        """Find the position with the lowest stack height across all blocks."""
        best_pos = None
        best_height = float('inf')

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers and h < best_height:
                        best_height = h
                        best_pos = Position(block_name, bay, row, h + 1)
                        if best_height == 0:
                            return best_pos

        if best_pos is None:
            # Yard is full
            first_block = list(yard_state.blocks.keys())[0]
            return Position(first_block, 1, 1, 999)

        return best_pos
