"""Shared pytest configuration — disable OVH API calls during tests."""
import os

# Prevent model_council from making real OVH API calls during tests
# This flag is checked by is_ovh_enabled() and survives load_env_file() overrides
os.environ["COUNCIL_DISABLED"] = "1"

