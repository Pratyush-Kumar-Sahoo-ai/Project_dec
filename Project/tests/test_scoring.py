"""Tests for scoring engine."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import SimulationStats


class TestSimulationStats:
    def test_empty_stats(self):
        stats = SimulationStats()
        assert stats.reshuffles_per_retrieval == 0.0
        assert stats.reshuffle_percentage == 0.0
        assert stats.max_reshuffles == 0

    def test_reshuffles_per_retrieval(self):
        stats = SimulationStats()
        stats.retrieval_events = 100
        stats.total_reshuffles = 50
        assert stats.reshuffles_per_retrieval == 0.5

    def test_score_perfect(self):
        stats = SimulationStats()
        stats.retrieval_events = 1000
        stats.total_reshuffles = 50  # 0.05 per retrieval
        assert stats.score_reshuffles() == 30.0

    def test_score_worst(self):
        stats = SimulationStats()
        stats.retrieval_events = 1000
        stats.total_reshuffles = 1000  # 1.0 per retrieval
        assert stats.score_reshuffles() == 0.0

    def test_score_mid(self):
        stats = SimulationStats()
        stats.retrieval_events = 1000
        stats.total_reshuffles = 450  # 0.45 per retrieval
        score = stats.score_reshuffles()
        assert 0 < score < 30

    def test_violations_clean(self):
        stats = SimulationStats()
        stats.hard_constraint_violations = 0
        assert stats.score_violations() == 10.0

    def test_violations_any(self):
        stats = SimulationStats()
        stats.hard_constraint_violations = 1
        assert stats.score_violations() == 0.0

    def test_summary_json(self):
        stats = SimulationStats()
        stats.total_events = 100
        stats.retrieval_events = 50
        stats.total_reshuffles = 10
        stats.reshuffle_counts = [0, 1, 0, 2]
        summary = stats.summary()
        assert "total_events" in summary
        assert "quantitative_score" in summary
        assert summary["total_events"] == 100
