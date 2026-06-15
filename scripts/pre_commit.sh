#!/usr/bin/env bash
# Pre-commit quality gate for DOTM Sniper
# Runs: ruff check → contract tests → full test suite
# Install: ln -sf ../../scripts/pre_commit.sh .git/hooks/pre-commit
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "═══ PRE-COMMIT GATE ═══"
echo "1/4 ruff check..."
python3 -m ruff check src/ || { echo "❌ ruff failed"; exit 1; }
echo "   ✓ ruff clean"

echo "2/4 contract tests..."
timeout 30 python3 -m pytest tests/test_contracts.py -q --tb=line || { echo "❌ contract tests failed"; exit 1; }
echo "   ✓ contracts pass"

echo "3/4 smoke import test..."
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

echo "4/4 skills sync check..."
timeout 15 python3 scripts/sync_skills.py || { echo "❌ skills out of sync — run: python3 scripts/sync_skills.py --fix"; exit 1; }

echo "═══ GATE PASSED — you may commit ═══"
