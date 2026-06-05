#!/bin/bash
# Submission validator — checks structure and runs a quick test
set -e

echo "============================================"
echo "  Submission Validator"
echo "============================================"
echo ""

ERRORS=0
WARNINGS=0

# ─── Check required directories ─────────────────────────────────────────
echo "Checking submission structure..."

if [ ! -d "solution" ]; then
    echo "  ERROR: solution/ directory not found"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: solution/ directory exists"
    # Check for at least one .py file
    PY_FILES=$(find solution -name "*.py" -not -name "__pycache__" | head -5)
    if [ -z "$PY_FILES" ]; then
        echo "  WARNING: No .py files found in solution/"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  OK: Found Python files in solution/"
    fi
fi

if [ ! -d "docs" ]; then
    echo "  ERROR: docs/ directory not found"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: docs/ directory exists"
    if [ ! -f "docs/design.md" ]; then
        echo "  ERROR: docs/design.md not found"
        ERRORS=$((ERRORS + 1))
    else
        LINES=$(wc -l < docs/design.md)
        echo "  OK: docs/design.md exists ($LINES lines)"
    fi
fi

if [ ! -d "results" ]; then
    echo "  WARNING: results/ directory not found (run against test data first)"
    WARNINGS=$((WARNINGS + 1))
else
    if [ ! -f "results/results.json" ]; then
        echo "  WARNING: results/results.json not found"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  OK: results/results.json exists"
        # Validate JSON
        python -c "import json; json.load(open('results/results.json'))" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  OK: results/results.json is valid JSON"
        else
            echo "  ERROR: results/results.json is not valid JSON"
            ERRORS=$((ERRORS + 1))
        fi
    fi
fi

echo ""

# ─── Check data files ───────────────────────────────────────────────────
echo "Checking data files..."

for f in data/yard_layout.json data/train/initial_state.json data/train/events.jsonl data/test/initial_state.json data/test/events.jsonl; do
    if [ ! -f "$f" ]; then
        echo "  ERROR: $f not found"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK: $f"
    fi
done

echo ""

# ─── Quick run test ──────────────────────────────────────────────────────
echo "Running quick validation test on train data (first 500 events)..."

# Create a temp file with first 500 events
TEMP_DIR=$(mktemp -d)
head -500 data/train/events.jsonl > "$TEMP_DIR/events.jsonl"
cp data/train/initial_state.json "$TEMP_DIR/initial_state.json"

# Try to find the strategy class
STRATEGY=""
if [ -d "solution" ]; then
    # Look for a class that extends PlacementStrategy
    STRATEGY_FILE=$(grep -rl "PlacementStrategy" solution/ 2>/dev/null | head -1)
    if [ -n "$STRATEGY_FILE" ]; then
        # Extract module path and class name
        MODULE=$(echo "$STRATEGY_FILE" | sed 's/\.py$//' | tr '/' '.')
        CLASS=$(grep -oP 'class\s+(\w+)\s*\(.*PlacementStrategy' "$STRATEGY_FILE" | head -1 | grep -oP 'class\s+\K\w+')
        if [ -n "$CLASS" ]; then
            STRATEGY="${MODULE}.${CLASS}"
            echo "  Found strategy: $STRATEGY"
        fi
    fi
fi

if [ -z "$STRATEGY" ]; then
    echo "  No strategy found in solution/ — using greedy baseline for test"
    STRATEGY="src.baseline_greedy.GreedyStrategy"
fi

# Run the strategy
python -m src.run \
    --strategy "$STRATEGY" \
    --data-dir "$TEMP_DIR" \
    --layout data/yard_layout.json \
    --output "$TEMP_DIR/test_results.json" 2>&1

RUN_EXIT=$?
rm -rf "$TEMP_DIR"

if [ $RUN_EXIT -eq 0 ]; then
    echo "  OK: Strategy ran successfully"
else
    echo "  ERROR: Strategy failed to run (exit code $RUN_EXIT)"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ─── Summary ─────────────────────────────────────────────────────────────
echo "============================================"
if [ $ERRORS -eq 0 ]; then
    echo "  PASSED ($WARNINGS warnings)"
else
    echo "  FAILED ($ERRORS errors, $WARNINGS warnings)"
fi
echo "============================================"

exit $ERRORS
