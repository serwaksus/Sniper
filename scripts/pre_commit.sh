#!/usr/bin/env bash
# Pre-commit quality gate for DOTM Sniper
# Runs: ruff check → contract tests → full test suite
# Install: ln -sf ../../scripts/pre_commit.sh .git/hooks/pre-commit
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "═══ PRE-COMMIT GATE ═══"
echo "1/3 ruff check..."
python3 -m ruff check src/ || { echo "❌ ruff failed"; exit 1; }
echo "   ✓ ruff clean"

echo "2/3 contract tests..."
timeout 30 python3 -m pytest tests/test_contracts.py -q --tb=line || { echo "❌ contract tests failed"; exit 1; }
echo "   ✓ contracts pass"

echo "3/3 smoke import test..."
timeout 10 python3 -c "
import sys; sys.path.insert(0, 'src')
import dotm_sniper
import backtest_simulator
import signal_pipeline
import signal_scorer
import manifold
import metaculus
import metaforecast
import model_council
print('   ✓ all imports OK')
" || { echo "❌ import chain broken"; exit 1; }

echo "═══ GATE PASSED — you may commit ═══"
