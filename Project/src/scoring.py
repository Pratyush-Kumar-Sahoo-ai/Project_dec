"""Scoring engine: tracks simulation metrics and computes final score."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SimulationStats:
    """Accumulates metrics during simulation."""

    # Event counts
    total_events: int = 0
    placement_events: int = 0
    retrieval_events: int = 0

    # Reshuffle metrics
    total_reshuffles: int = 0
    retrievals_with_reshuffles: int = 0
    reshuffle_counts: List[int] = field(default_factory=list)

    # Errors and violations
    hard_constraint_violations: int = 0
    placement_failures: int = 0
    retrieval_not_found: int = 0
    errors: int = 0

    # Final state
    final_yard_count: int = 0
    elapsed_sec: float = 0.0

    @property
    def total_loads(self) -> int:
        """Number of LOAD/TRUCK_DLVR events (retrievals)."""
        return self.retrieval_events

    @property
    def reshuffles_per_retrieval(self) -> float:
        """Average reshuffles per retrieval event."""
        if self.retrieval_events == 0:
            return 0.0
        return self.total_reshuffles / self.retrieval_events

    @property
    def reshuffle_percentage(self) -> float:
        """Percentage of retrievals that required reshuffles."""
        if self.retrieval_events == 0:
            return 0.0
        return 100.0 * self.retrievals_with_reshuffles / self.retrieval_events

    @property
    def max_reshuffles(self) -> int:
        """Maximum reshuffles for a single retrieval."""
        return max(self.reshuffle_counts) if self.reshuffle_counts else 0

    def score_reshuffles(self) -> float:
        """Score reshuffles on 0-30 scale.

        30 pts if reshuffles/retrieval <= 0.10
         0 pts if reshuffles/retrieval >= 0.80
        Linear interpolation between.

        Reference baselines (test set):
          Random: ~0.80 (0 pts)
          Greedy (lowest stack): ~0.71 (~4 pts)
          Departure-time-aware: ~0.30-0.40 (17-21 pts)
          Advanced solver/ML: ~0.10-0.20 (26-30 pts)
        """
        ratio = self.reshuffles_per_retrieval
        if ratio <= 0.10:
            return 30.0
        if ratio >= 0.80:
            return 0.0
        return 30.0 * (0.80 - ratio) / (0.80 - 0.10)

    def score_violations(self) -> float:
        """Score hard constraint violations on 0-10 scale.

        10 pts if zero violations, 0 pts if any.
        """
        return 10.0 if self.hard_constraint_violations == 0 else 0.0

    def quantitative_score(self) -> float:
        """Total quantitative score (0-40)."""
        return self.score_reshuffles() + self.score_violations()

    def summary(self) -> dict:
        """Return a summary dict for JSON output."""
        return {
            "total_events": self.total_events,
            "placement_events": self.placement_events,
            "retrieval_events": self.retrieval_events,
            "total_reshuffles": self.total_reshuffles,
            "reshuffles_per_retrieval": round(self.reshuffles_per_retrieval, 4),
            "reshuffle_percentage": round(self.reshuffle_percentage, 2),
            "max_reshuffles_single": self.max_reshuffles,
            "retrievals_with_reshuffles": self.retrievals_with_reshuffles,
            "hard_constraint_violations": self.hard_constraint_violations,
            "placement_failures": self.placement_failures,
            "errors": self.errors,
            "final_yard_count": self.final_yard_count,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "score_reshuffles": round(self.score_reshuffles(), 1),
            "score_violations": round(self.score_violations(), 1),
            "quantitative_score": round(self.quantitative_score(), 1),
        }

    def print_report(self):
        """Print a formatted report to stdout."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        print(f"  Events processed:       {self.total_events:,}")
        print(f"    Placements:           {self.placement_events:,}")
        print(f"    Retrievals:           {self.retrieval_events:,}")
        print(f"  Total reshuffles:       {self.total_reshuffles:,}")
        print(f"  Reshuffles/retrieval:   {self.reshuffles_per_retrieval:.4f}")
        print(f"  Retrievals w/ reshuf:   {self.retrievals_with_reshuffles:,} "
              f"({self.reshuffle_percentage:.1f}%)")
        print(f"  Max reshuffles (single):{self.max_reshuffles}")
        print(f"  Constraint violations:  {self.hard_constraint_violations}")
        print(f"  Placement failures:     {self.placement_failures}")
        print(f"  Errors:                 {self.errors}")
        print(f"  Final yard count:       {self.final_yard_count:,}")
        print(f"  Elapsed time:           {self.elapsed_sec:.2f}s")
        print("-" * 60)
        print(f"  SCORE — Reshuffles:     {self.score_reshuffles():.1f} / 30")
        print(f"  SCORE — Violations:     {self.score_violations():.1f} / 10")
        print(f"  QUANTITATIVE TOTAL:     {self.quantitative_score():.1f} / 40")
        print("=" * 60 + "\n")
