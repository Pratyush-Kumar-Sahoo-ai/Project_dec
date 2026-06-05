# Container Yard Placement — Technical Assessment

## Quick Start

```bash
# 1. Verify Python 3.8+
python --version

# 2. Explore the baselines
python -m src.run --strategy src.baseline_greedy.GreedyStrategy --data-dir data/train -v
python -m src.run --strategy src.baseline_random.RandomStrategy --data-dir data/train -v

# 3. Create your strategy
mkdir -p solution
# Implement PlacementStrategy in solution/my_strategy.py

# 4. Test on train data
python -m src.run --strategy solution.my_strategy.MyStrategy --data-dir data/train -v

# 5. Run on test data and save results
python -m src.run --strategy solution.my_strategy.MyStrategy --data-dir data/test -o results/results.json -v

# 6. Validate submission
bash validate_submission.sh
```

## Project Structure

```
PROBLEM.md                  # Problem statement (read this first)
EVALUATION.md               # Scoring rubric

data/
  yard_layout.json          # Yard block dimensions
  train/
    initial_state.json      # Starting yard state (day 0)
    events.jsonl            # 20 days of events
  test/
    initial_state.json      # Starting yard state (day 20)
    events.jsonl            # 20 days of events (scored)

src/
  models.py                 # Data models: Position, Container, Event
  yard_state.py             # Yard state manager (query this, don't modify)
  event_reader.py           # Event file reader
  simulator.py              # Simulation engine
  scoring.py                # Scoring and metrics
  placement_interface.py    # Abstract class to implement
  baseline_random.py        # Reference: random placement
  baseline_greedy.py        # Reference: lowest-stack placement
  run.py                    # CLI runner
  external_adapter.py       # Non-Python solver adapter

solution/                   # YOUR CODE HERE
docs/                       # YOUR DESIGN DOCUMENT HERE
results/                    # YOUR TEST RESULTS HERE
```

## Non-Python Solvers

If using a language other than Python, implement the stdin/stdout JSON protocol:

1. Your solver receives on stdin:
   - `{"type": "INIT", "yard_layout": {...}, "initial_state": {...}}`
   - Respond: `{"status": "ok"}`

2. For each placement:
   - `{"type": "PLACE", "event": {...}, "yard_summary": {...}}`
   - Respond: `{"block": "B01", "bay": 1, "row": 1, "tier": 1}`

3. After each retrieval:
   - `{"type": "RETRIEVE", "container_id": "...", "reshuffles": N}`
   - Respond: `{"status": "ok"}`

4. At end: `{"type": "END"}`

Run with: `python -m src.run --external ./my_solver --data-dir data/test`

## Key Tips

- Read `PROBLEM.md` carefully before starting
- Study the train data — patterns in vessel schedules and departure times are your biggest opportunity
- Use `yard_state.snapshot()`/`restore()` for lookahead and what-if analysis
- The `on_event()` and `on_container_retrieved()` callbacks are optional but useful for adaptive strategies
- Focus on minimizing reshuffles — that's 30 of 40 quantitative points
