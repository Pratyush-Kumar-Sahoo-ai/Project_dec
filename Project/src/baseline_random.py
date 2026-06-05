"""Baseline: random valid placement.

This is the performance floor. Any reasonable strategy should beat this.
"""

import random

from .models import Event, Position
from .yard_state import YardState
from .placement_interface import PlacementStrategy


class RandomStrategy(PlacementStrategy):
    """Places containers in a random valid position."""

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        pass

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        """Pick a random block, then random bay/row with space."""
        blocks = list(yard_state.blocks.keys())
        random.shuffle(blocks)

        for block_name in blocks:
            bi = yard_state.blocks[block_name]
            positions = []
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        positions.append(
                            Position(block_name, bay, row, h + 1))

            if positions:
                return random.choice(positions)

        # Yard is full — return an invalid position (will be caught by simulator)
        return Position(blocks[0], 1, 1, 999)
