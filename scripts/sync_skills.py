#!/usr/bin/env python3
"""
Sync checker: verifies that skill descriptions match actual code constants.

Reads config.py + other key modules, scans all SKILL.md files,
finds numeric references to tracked constants, and reports mismatches.

Usage:
    python3 scripts/sync_skills.py              # Check only (exit 1 on mismatch)
    python3 scripts/sync_skills.py --fix        # Auto-fix simple cases
    python3 scripts/sync_skills.py --verbose    # Show all references (not just mismatches)

Integrates into pre_commit.sh as step 4/4.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SKILLS_DIR = Path.home() / ".config" / "opencode" / "skills"

# ── Extract constants from code ──────────────────────────────────────
def load_code_constants() -> dict[str, float | int | str]:
    """Import key modules and extract tracked constants."""
    sys.path.insert(0, str(SRC_DIR))
    vals: dict[str, float | int | str] = {}

    # config.py
    import config
    _config_int_floats = [
        "MIN_P_MODEL", "MIN_CONFIDENCE", "BURN_IN_TRADES",
        "BAYESIAN_PRIOR_STRENGTH", "MIN_TRADES_FOR_WEIGHT",
        "MAX_P_MODEL_RATIO", "SIGNAL_THRESHOLD_DEFAULT",
        "MAX_CONCURRENT_TRADES", "PORTFOLIO_DRAWDOWN_STOP",
        "PER_POSITION_MAX_LOSS", "TIME_DECAY_EXIT_THRESHOLD",
        "CONVERGENCE_TAKE_PROFIT",
    ]
    for name in _config_int_floats:
        v = getattr(config, name, None)
        if v is not None:
            vals[name] = v

    # signal_scorer.py
    try:
        import signal_scorer
        for name in ["MIN_PROB_RATIO", "DOTM_PRICE_FLOOR"]:
            v = getattr(signal_scorer, name, None)
            if v is not None:
                vals[name] = v
    except Exception:
        pass

    # market_fetcher.py
    try:
        import market_fetcher
        for name in ["DOTM_PRICE_MAX", "DOTM_PRICE_FLOOR"]:
            v = getattr(market_fetcher, name, None)
            if v is not None:
                vals[name] = v
    except Exception:
        pass

    # backtest_simulator.py
    try:
        import backtest_simulator
        for name in ["BACKTEST_MAX_WORKERS", "API_RATE_LIMIT_RPS"]:
            v = getattr(backtest_simulator, name, None)
            if v is not None:
                vals[name] = v
    except Exception:
        pass

    return vals


# ── Semantic aliases: map patterns to constants ─────────────────────
# Only used when the line ALSO mentions the constant name or a unique keyword.
# Format: (regex_for_number, constant_name, required_context_keywords)
SEMANTIC_CHECKS: list[tuple[str, str, tuple[str, ...]]] = [
    # "≤ NX price" only when line mentions MAX_P_MODEL_RATIO or "ratio cap"
    (r"[≤<]\s*(\d+\.?\d*)\s*[×x]\s*price", "MAX_P_MODEL_RATIO",
     ("max_p_model_ratio", "ratio cap", "p_model ≤", "p_model cap")),
]


def format_value(v: float | int) -> str:
    """Format a value for comparison (handles floats like 3.0 vs 3)."""
    if isinstance(v, int):
        return str(v)
    if v == int(v):
        return f"{v:.1f}"
    return f"{v}"


def values_match(skill_val: float, code_val: float | int) -> bool:
    """Check if a value found in skill text matches the code value."""
    if isinstance(code_val, int):
        return abs(skill_val - code_val) < 0.01
    return abs(skill_val - code_val) < 0.01


def scan_skill_file(filepath: Path, constants: dict) -> list[dict]:
    """Scan a single SKILL.md for stale constant references.
    Returns list of issues (empty = all good)."""
    issues = []
    try:
        content = filepath.read_text()
    except Exception:
        return []

    skill_name = filepath.parent.name
    lines = content.split("\n")

    for i, line in enumerate(lines, 1):
        # Skip lines that say "raised from" (documenting a change)
        if "raised from" in line.lower() or "changed from" in line.lower():
            continue
        # Skip comments about old values
        if line.strip().startswith("<!--") or "previously" in line.lower():
            continue

        # ── Check 1: Direct constant name reference (word-boundary) ──
        for const_name, code_val in constants.items():
            # Use negative lookbehind to avoid matching substrings
            # e.g., MIN_CONFIDENCE shouldn't match ADVISOR_MIN_CONFIDENCE
            const_match = re.search(r"(?<![A-Z_])" + re.escape(const_name) + r"(?![A-Z_])", line)
            if not const_match:
                continue

            # Only look at text AFTER the constant name match, and only
            # the FIRST number that follows an = or "currently" pattern.
            # This avoids matching unrelated numbers later in the line.
            after_const = line[const_match.end():]
            num_match = re.search(
                r"(?:=\s*|currently\s+|≤\s*|=\s*|\s*:\s*)(\d+\.?\d*)x?",
                after_const,
            )
            if not num_match:
                continue

            try:
                num_val = float(num_match.group(1))
            except ValueError:
                continue

            if not values_match(num_val, code_val):
                issues.append({
                    "skill": skill_name,
                    "file": str(filepath),
                    "line": i,
                    "text": line.strip()[:100],
                    "constant": const_name,
                    "code_value": code_val,
                    "skill_value": num_val,
                    "type": "named",
                })

        # ── Check 2: Semantic patterns (require context keywords) ──
        for pattern, const_name, context_kw in SEMANTIC_CHECKS:
            if const_name not in constants:
                continue

            # Only apply if line contains at least one context keyword
            line_lower = line.lower()
            if not any(kw in line_lower for kw in context_kw):
                continue

            matches = re.finditer(pattern, line, re.IGNORECASE)
            for m in matches:
                try:
                    skill_val = float(m.group(1))
                except (ValueError, IndexError):
                    continue

                code_val = constants[const_name]
                if not values_match(skill_val, code_val):
                    # Avoid duplicate if already caught by named check
                    if not any(iss["line"] == i and iss["constant"] == const_name
                               for iss in issues):
                        issues.append({
                            "skill": skill_name,
                            "file": str(filepath),
                            "line": i,
                            "text": line.strip()[:100],
                            "constant": const_name,
                            "code_value": code_val,
                            "skill_value": skill_val,
                            "type": "semantic",
                        })

    return issues


def auto_fix(issue: dict) -> bool:
    """Attempt to auto-fix a simple mismatch. Returns True if fixed."""
    filepath = Path(issue["file"])
    try:
        lines = filepath.read_text().split("\n")
    except Exception:
        return False

    line_idx = issue["line"] - 1
    old_line = lines[line_idx]

    code_val = issue["code_value"]
    skill_val = issue["skill_value"]

    # Format replacement: try both "3.0" and "3.0x" patterns
    if isinstance(code_val, float) and code_val == int(code_val):
        code_strs = [f"{code_val:.1f}", f"{int(code_val)}"]
    else:
        code_strs = [str(code_val)]

    if isinstance(skill_val, float) and skill_val == int(skill_val):
        skill_strs = [f"{skill_val:.1f}", f"{int(skill_val)}"]
    else:
        skill_strs = [str(skill_val)]

    fixed = False
    new_line = old_line
    for sk in skill_strs:
        for cd in code_strs:
            # Replace "X.Xx" with "Y.Yx" (ratio notation)
            if f"{sk}x" in new_line:
                new_line = new_line.replace(f"{sk}x", f"{cd}x")
                fixed = True
            # Replace "X.X×" (unicode multiply)
            if f"{sk}×" in new_line:
                new_line = new_line.replace(f"{sk}×", f"{cd}×")
                fixed = True
            # Replace "X.X " with "Y.Y " (but not if part of a longer number)
            if f" {sk} " in new_line and not f"{sk}." in new_line.replace(f" {sk} ", "", 1):
                new_line = new_line.replace(f" {sk} ", f" {cd} ")
                fixed = True
            # Replace "= X.X" with "= Y.Y"
            if f"= {sk}" in new_line and not f"= {sk}." in new_line:
                new_line = new_line.replace(f"= {sk}", f"= {cd}")
                fixed = True

    if fixed and new_line != old_line:
        lines[line_idx] = new_line
        filepath.write_text("\n".join(lines))
        return True

    return False


def main() -> int:
    fix_mode = "--fix" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if not SKILLS_DIR.exists():
        print("⚠ Skills directory not found, skipping sync check")
        return 0

    print("4/4 skills sync check...")
    constants = load_code_constants()

    if verbose:
        print(f"\n   Tracked constants ({len(constants)}):")
        for k, v in sorted(constants.items()):
            print(f"     {k} = {v}")

    skill_files = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    all_issues = []

    for sf in skill_files:
        issues = scan_skill_file(sf, constants)
        all_issues.extend(issues)

    # Deduplicate
    seen = set()
    unique_issues = []
    for iss in all_issues:
        key = (iss["file"], iss["line"], iss["constant"])
        if key not in seen:
            seen.add(key)
            unique_issues.append(iss)

    if not unique_issues:
        print("   ✓ all skills in sync with code")
        return 0

    print(f"   ⚠ {len(unique_issues)} stale reference(s) found:\n")

    fixed_count = 0
    for iss in unique_issues:
        symbol = "🔧" if fix_mode else "⚠"
        print(f"   {symbol} {iss['skill']}/SKILL.md:{iss['line']} — "
              f"{iss['constant']}: skill says {iss['skill_value']}, code says {iss['code_value']}")
        print(f"      {iss['text']}")

        if fix_mode:
            if auto_fix(iss):
                print(f"      → FIXED")
                fixed_count += 1
            else:
                print(f"      → CANNOT auto-fix (manual edit needed)")

    if fix_mode and fixed_count:
        print(f"\n   ✓ Auto-fixed {fixed_count}/{len(unique_issues)} references")
        remaining = len(unique_issues) - fixed_count
        if remaining > 0:
            print(f"   ⚠ {remaining} reference(s) need manual fixing")
            return 1
        return 0

    print(f"\n   ❌ {len(unique_issues)} skill reference(s) out of sync")
    print("   Run: python3 scripts/sync_skills.py --fix")
    return 1


if __name__ == "__main__":
    sys.exit(main())
