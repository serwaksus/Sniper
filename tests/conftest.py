"""Shared pytest configuration — disable external API calls during tests."""
import os

# Prevent model_council from making real OVH API calls during tests
# This flag is checked by is_ovh_enabled() and survives load_env_file() overrides
os.environ["COUNCIL_DISABLED"] = "1"

# Disable external_oracles HTTP calls during tests.
# compute_oracle_bonus checks this flag and returns 0 bonus immediately.
# Individual oracle unit tests call the sub-functions directly (bypassing
# compute_oracle_bonus), so they are unaffected.
os.environ["ORACLES_DISABLED"] = "1"
