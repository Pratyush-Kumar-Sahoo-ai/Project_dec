"""CLI runner: runs a placement strategy against a dataset.

Usage:
    python -m src.run --strategy src.baseline_greedy.GreedyStrategy \\
                      --data-dir data/train --verbose

    python -m src.run --external ./my_solver \\
                      --data-dir data/test --output results.json
"""

import argparse
import importlib
import json
import os
import sys
import time

from .models import Event
from .yard_state import YardState
from .event_reader import read_events
from .simulator import Simulator
from .placement_interface import PlacementStrategy


def load_strategy(strategy_path: str) -> PlacementStrategy:
    """Load a PlacementStrategy class from a dotted path.

    Args:
        strategy_path: e.g. "src.baseline_greedy.GreedyStrategy"
                       or "solution.my_strategy.MyStrategy"

    Returns:
        An instance of the strategy.
    """
    parts = strategy_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Strategy path must be 'module.ClassName', got: {strategy_path}")

    module_path, class_name = parts
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not issubclass(cls, PlacementStrategy):
        raise TypeError(f"{class_name} does not extend PlacementStrategy")

    return cls()


def main():
    parser = argparse.ArgumentParser(
        description="Run a placement strategy against the container terminal "
                    "simulation")
    parser.add_argument(
        "--strategy", type=str,
        default="src.baseline_greedy.GreedyStrategy",
        help="Dotted path to PlacementStrategy class "
             "(default: src.baseline_greedy.GreedyStrategy)")
    parser.add_argument(
        "--external", type=str, default=None,
        help="Path to external solver executable (stdin/stdout JSON protocol)")
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing initial_state.json and events.jsonl "
             "(e.g. data/train or data/test)")
    parser.add_argument(
        "--layout", type=str, default=None,
        help="Path to yard_layout.json (default: <data-dir>/../yard_layout.json)")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file for results JSON")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress during simulation")

    args = parser.parse_args()

    # Resolve paths
    data_dir = args.data_dir
    if args.layout:
        layout_path = args.layout
    else:
        layout_path = os.path.join(os.path.dirname(data_dir), "yard_layout.json")

    initial_state_path = os.path.join(data_dir, "initial_state.json")
    events_path = os.path.join(data_dir, "events.jsonl")

    # Validate files exist
    for path, desc in [(layout_path, "yard layout"),
                       (initial_state_path, "initial state"),
                       (events_path, "events")]:
        if not os.path.exists(path):
            print(f"ERROR: {desc} file not found: {path}")
            sys.exit(1)

    # Load data
    print(f"Loading yard layout from {layout_path}...")
    with open(layout_path) as f:
        yard_layout = json.load(f)

    print(f"Loading initial state from {initial_state_path}...")
    with open(initial_state_path) as f:
        initial_state = json.load(f)

    print(f"Loading events from {events_path}...")
    events = read_events(events_path)
    print(f"  {len(events):,} events loaded")

    # Initialize yard
    yard = YardState(yard_layout)
    yard.load_initial_state(initial_state)
    print(f"  {yard.total_containers():,} containers in initial state")

    # Load strategy
    if args.external:
        from .external_adapter import ExternalStrategy
        strategy = ExternalStrategy(args.external)
        print(f"Using external solver: {args.external}")
    else:
        strategy = load_strategy(args.strategy)
        print(f"Using strategy: {args.strategy}")

    # Initialize strategy
    strategy.initialize(yard_layout, initial_state)

    # Run simulation
    print("\nRunning simulation...")
    sim = Simulator(yard, strategy, verbose=args.verbose)
    stats = sim.run(events)

    # Print results
    stats.print_report()

    # Save results
    if args.output:
        results = stats.summary()
        results["strategy"] = args.strategy if not args.external else args.external
        results["data_dir"] = data_dir
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
