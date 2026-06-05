# Container Yard Placement Optimizer

## Time Limit: 48 Hours

---

## Overview

You are building a **container placement algorithm** for a port terminal yard. When containers arrive at the terminal (from ships or trucks), your algorithm decides **where to store each container** in the yard. The goal is to minimize **reshuffles** — the number of containers that must be moved out of the way when retrieving a container later.

This is a real-world optimization problem faced by every container terminal globally. Good placement reduces operational cost, crane idle time, and vessel turnaround time.

## The Terminal

The yard consists of **10 blocks** arranged in a grid. Each block is a 3D grid of **bays** (length), **rows** (width), and **tiers** (height). Containers stack vertically from tier 1 (ground) upward.

| Block | Bays | Rows | Tiers | Capacity |
|-------|------|------|-------|----------|
| B01   | 40   | 6    | 5     | 1,200    |
| B02   | 40   | 6    | 5     | 1,200    |
| B03   | 30   | 6    | 5     | 900      |
| B04   | 30   | 6    | 5     | 900      |
| B05   | 36   | 6    | 5     | 1,080    |
| B06   | 36   | 6    | 5     | 1,080    |
| B07   | 36   | 6    | 5     | 1,080    |
| B08   | 24   | 6    | 5     | 720      |
| B09   | 24   | 6    | 5     | 720      |
| B10   | 24   | 6    | 5     | 720      |
| **Total** | | | | **9,600** |

All containers are **20ft standard** (one container per bay-row slot).

## Events

The simulation processes 4 event types:

| Event | Direction | Your Role |
|-------|-----------|-----------|
| **DISCHARGE** | Ship → Yard | **You decide placement** |
| **TRUCK_RECV** | Gate → Yard | **You decide placement** |
| **LOAD** | Yard → Ship | Handled by simulator (reshuffles counted) |
| **TRUCK_DLVR** | Yard → Gate | Handled by simulator (reshuffles counted) |

For each `DISCHARGE` and `TRUCK_RECV` event, your `place_container()` function is called. You receive the full yard state and the event details, and must return a valid `Position(block, bay, row, tier)`.

For `LOAD` and `TRUCK_DLVR` events, the simulator locates the container and retrieves it. Any containers stacked **above** the target must be temporarily moved — each one counts as a **reshuffle**.

## Container Attributes

Each event/container has these attributes:

| Attribute | Description |
|-----------|-------------|
| `container_id` | Unique identifier |
| `size` | Always 20 |
| `weight_class` | LIGHT, MEDIUM, or HEAVY |
| `vessel_id` | Associated vessel (e.g., "VSL001") |
| `port_of_discharge` | Destination port code |
| `departure_time` | Vessel ETD — all containers on the same vessel share this value |

**Key signals for placement**:
- `departure_time` is the vessel's ETD. All containers assigned to the same vessel share the same departure time. Containers leaving sooner should be easier to retrieve (stacked higher or in less obstructed positions).
- During **LOAD** operations, containers are loaded **grouped by port of discharge**, and within each port group, **heavier containers are loaded first** (HEAVY → MEDIUM → LIGHT). Understanding this loading order can inform placement decisions.

## Hard Constraints

Your placement **must** satisfy all of these or it will be rejected:

1. **Block exists** — the block name must be valid
2. **Bay/row within dimensions** — must be within the block's grid
3. **Tier ≤ max tiers (5)** — cannot exceed the maximum stack height
4. **Tier = stack_height + 1** — must stack directly on top (no gaps)
5. **Slot not occupied** — cannot place where a container already exists

Violations are penalized in scoring. If your strategy raises an exception or returns an invalid position, a fallback placement is used automatically.

## Data

You receive **40 days** of synthetic terminal data, split into two periods:

```
data/
  yard_layout.json          # Block dimensions (same for both)
  vessel_schedule.json      # Vessel rotation schedule (ETA/ETD per visit)
  train/
    initial_state.json      # Yard state at day 0 (~4,800 containers)
    events.jsonl            # Days 1-20 (~20,000 events)
  test/
    initial_state.json      # Yard state at day 20 (~5,000 containers)
    events.jsonl            # Days 21-40 (~20,000 events)
```

### Vessel Schedule

`vessel_schedule.json` provides the rotation schedule for each vessel:
- **vessel_id**: Vessel identifier (VSL001-VSL020)
- **ports**: The ports this vessel serves
- **rotations**: List of port calls, each with:
  - `rotation_number`: Sequential visit number
  - `eta`: Estimated time of arrival (discharge begins)
  - `etd`: Estimated time of departure (after loading completes)
  - `discharge_start` / `discharge_end`: Window for DISCHARGE events
  - `load_start` / `load_end`: Window for LOAD events

Some vessels visit multiple times across the 40-day period. VSL019-VSL020 are truck-only vessels with no ship calls.

- **Train** (days 1-20): Use freely for analysis, tuning, and development. Explore patterns, test strategies, iterate.
- **Test** (days 21-40): Your submission is scored on this. Same distribution as train. Run once and report results.

Both datasets are fixed and identical for all candidates.

## What You Implement

Implement the `PlacementStrategy` interface in `src/placement_interface.py`:

```python
class PlacementStrategy(ABC):
    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        """Called once before simulation. Set up data structures."""

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        """Decide where to place an incoming container."""

    def on_event(self, event: Event) -> None:
        """Optional: called for every event (track patterns)."""

    def on_container_retrieved(self, container_id, position, reshuffles) -> None:
        """Optional: called after each retrieval (track performance)."""
```

The `yard_state` object provides:
- `get_stack_height(block, bay, row)` — current stack height at a position
- `get_container_at(block, bay, row, tier)` — container ID at a specific slot
- `get_container_info(container_id)` — full container details
- `get_containers_above(container_id)` — containers stacked above a target
- `get_block_occupancy(block)` — (occupied, capacity) for a block
- `get_all_containers()` — all containers with positions
- `get_containers_by_vessel(vessel_id)` — containers for a specific vessel
- `is_position_valid(position)` — check if a position satisfies hard constraints
- `snapshot()` / `restore()` — save and restore yard state (for lookahead/search)

## Approach

Any approach is welcome:

- **Heuristic rules** — departure-time stacking, vessel grouping, weight ordering
- **Mathematical optimization** — MIP/CP/SAT formulation
- **Machine learning** — learned placement policy from train data
- **Search-based** — simulation lookahead using `snapshot()`/`restore()`
- **Hybrid** — combination of the above

## Running

```bash
# Run with greedy baseline on train data
python -m src.run --strategy src.baseline_greedy.GreedyStrategy --data-dir data/train -v

# Run your strategy on test data
python -m src.run --strategy solution.my_strategy.MyStrategy --data-dir data/test -o results/results.json -v

# Non-Python: use external solver protocol
python -m src.run --external ./my_solver --data-dir data/test -o results/results.json -v
```

## Deliverables

1. **`solution/`** — Your placement algorithm source code + `run.sh` (if non-Python)
2. **`docs/design.md`** — 1-3 page design document:
   - Algorithm description and rationale
   - Trade-offs considered and alternatives rejected
   - Analysis of train data patterns
   - Time/space complexity per placement decision
3. **`results/`** — Output from running against test dataset
4. **Tests** (optional but scored) — Unit tests for your strategy

See `EVALUATION.md` for the detailed scoring rubric.
