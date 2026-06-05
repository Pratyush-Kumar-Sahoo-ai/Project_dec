"""Abstract interface that candidates must implement."""

from abc import ABC, abstractmethod

from .models import Event, Position
from .yard_state import YardState


class PlacementStrategy(ABC):
    """Base class for placement strategies.

    Candidates implement this interface. The simulator calls these methods
    during the event loop.
    """

    @abstractmethod
    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        """Called once before the simulation starts.

        Use this to set up any data structures, analyze the layout,
        or pre-process the initial yard state.

        Args:
            yard_layout: The yard layout JSON (block dimensions).
            initial_state: The initial state JSON (pre-existing containers).
        """
        pass

    @abstractmethod
    def place_container(self, yard_state: YardState, event: Event) -> Position:
        """Decide where to place an incoming container.

        Called for each DISCHARGE and TRUCK_RECV event. Must return a valid
        Position that satisfies all hard constraints:
        1. Block exists, bay/row within dimensions
        2. Tier <= max_tiers (5)
        3. Tier == current stack height + 1 (no gaps)
        4. Slot is not occupied

        Args:
            yard_state: Current yard state (query only — do not modify).
            event: The incoming container event with all attributes.

        Returns:
            Position where the container should be placed.
        """
        pass

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        """Optional callback after a LOAD or TRUCK_DLVR event.

        Informs your algorithm how many reshuffles were needed to retrieve
        this container. Use this to track performance or adapt your strategy.

        Args:
            container_id: The container that was retrieved.
            position: Where it was in the yard.
            reshuffles: Number of containers that had to be moved.
        """
        pass

    def on_event(self, event: Event) -> None:
        """Optional callback for every event (all types).

        Called before any processing. Use this to track patterns,
        maintain statistics, or update internal state.

        Args:
            event: The event being processed.
        """
        pass
