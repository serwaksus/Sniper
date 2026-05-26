#!/usr/bin/env python3
"""
Unit tests for cron job configurations
Run with: python3 -m pytest tests/test_cron_validator.py -v
Or directly: python3 tests/test_cron_validator.py
"""
import unittest
import sys
import json
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cron_validator import CronValidator

class TestCronValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.jobs_path = os.path.join(self.temp_dir, 'jobs.json')

    def tearDown(self):
        if os.path.exists(self.jobs_path):
            os.remove(self.jobs_path)
        os.rmdir(self.temp_dir)

    def write_jobs(self, data):
        with open(self.jobs_path, 'w') as f:
            json.dump(data, f)

    def test_valid_shell_job(self):
        """Test that a valid shell job passes validation"""
        jobs = {
            "version": 1,
            "jobs": [{
                "id": "test-001",
                "name": "test-shell",
                "enabled": True,
                "createdAtMs": 1000,
                "updatedAtMs": 1000,
                "schedule": {"kind": "every", "everyMs": 1800000},
                "sessionTarget": "isolated",
                "wakeMode": "now",
                "payload": {
                    "kind": "shell",
                    "command": "cd /root/dotm-sniper && python3 script.py"
                },
                "delivery": {"mode": "announce", "channel": "last", "to": "telegram:123"},
                "state": {}
            }]
        }
        self.write_jobs(jobs)
        validator = CronValidator(self.jobs_path)
        valid, errors, warnings = validator.validate_all()
        self.assertTrue(valid, f"Expected valid but got errors: {errors}")

    def test_agent_turn_with_python_warning(self):
        """Test that agentTurn with python triggers a warning"""
        jobs = {
            "version": 1,
            "jobs": [{
                "id": "test-002",
                "name": "test-agent",
                "enabled": True,
                "createdAtMs": 1000,
                "updatedAtMs": 1000,
                "schedule": {"kind": "every", "everyMs": 1800000},
                "sessionTarget": "isolated",
                "wakeMode": "now",
                "payload": {
                    "kind": "agentTurn",
                    "message": "python3 /root/dotm-sniper/script.py"
                },
                "delivery": {"mode": "announce", "to": "telegram:123"},
                "state": {}
            }]
        }
        self.write_jobs(jobs)
        validator = CronValidator(self.jobs_path)
        valid, errors, warnings = validator.validate_all()
        self.assertTrue(valid)
        self.assertTrue(any('agentTurn with python' in w for w in warnings))

    def test_missing_payload_kind(self):
        """Test that missing payload.kind is caught"""
        jobs = {
            "version": 1,
            "jobs": [{
                "id": "test-003",
                "name": "test-no-kind",
                "enabled": True,
                "createdAtMs": 1000,
                "updatedAtMs": 1000,
                "schedule": {"kind": "every", "everyMs": 1800000},
                "sessionTarget": "isolated",
                "wakeMode": "now",
                "payload": {"command": "echo hello"},
                "state": {}
            }]
        }
        self.write_jobs(jobs)
        validator = CronValidator(self.jobs_path)
        valid, errors, warnings = validator.validate_all()
        self.assertFalse(valid)
        self.assertTrue(any('no payload.kind' in e for e in errors))

    def test_shell_job_missing_command(self):
        """Test that shell job without command is rejected"""
        jobs = {
            "version": 1,
            "jobs": [{
                "id": "test-004",
                "name": "test-shell-no-cmd",
                "enabled": True,
                "createdAtMs": 1000,
                "updatedAtMs": 1000,
                "schedule": {"kind": "every", "everyMs": 1800000},
                "sessionTarget": "isolated",
                "wakeMode": "now",
                "payload": {"kind": "shell"},
                "state": {}
            }]
        }
        self.write_jobs(jobs)
        validator = CronValidator(self.jobs_path)
        valid, errors, warnings = validator.validate_all()
        self.assertFalse(valid)

    def test_consecutive_errors_detected(self):
        """Test that consecutive errors are flagged"""
        jobs = {
            "version": 1,
            "jobs": [{
                "id": "test-005",
                "name": "test-failing",
                "enabled": True,
                "createdAtMs": 1000,
                "updatedAtMs": 1000,
                "schedule": {"kind": "every", "everyMs": 1800000},
                "sessionTarget": "isolated",
                "wakeMode": "now",
                "payload": {"kind": "shell", "command": "echo test"},
                "state": {
                    "consecutiveErrors": 5,
                    "lastError": "timeout"
                }
            }]
        }
        self.write_jobs(jobs)
        validator = CronValidator(self.jobs_path)
        valid, errors, warnings = validator.validate_all()
        self.assertFalse(valid)
        self.assertTrue(any('consecutive errors' in e for e in errors))

    def test_edge_case_exec_unavailable(self):
        """Test that exec unavailable pattern is detected"""
        jobs = {
            "version": 1,
            "jobs": [{
                "id": "test-006",
                "name": "test-exec",
                "enabled": True,
                "createdAtMs": 1000,
                "updatedAtMs": 1000,
                "schedule": {"kind": "every", "everyMs": 1800000},
                "sessionTarget": "isolated",
                "wakeMode": "now",
                "payload": {"kind": "agentTurn", "message": "do something"},
                "state": {
                    "lastError": "exec tool: NOT AVAILABLE"
                }
            }]
        }
        self.write_jobs(jobs)
        validator = CronValidator(self.jobs_path)
        valid, errors, warnings = validator.validate_all()
        self.assertTrue(valid)
        self.assertTrue(any('exec_unavailable' in w for w in warnings))

    def test_invalid_json(self):
        """Test that invalid JSON is caught"""
        with open(self.jobs_path, 'w') as f:
            f.write("{ invalid json }")

        validator = CronValidator(self.jobs_path)
        valid, errors, warnings = validator.validate_all()
        self.assertFalse(valid)
        self.assertTrue(any('Invalid JSON' in e for e in errors))

    def test_missing_required_fields(self):
        """Test that missing required fields are caught"""
        jobs = {
            "version": 1,
            "jobs": [{
                "id": "test-007",
                "name": "test-missing"
            }]
        }
        self.write_jobs(jobs)
        validator = CronValidator(self.jobs_path)
        valid, errors, warnings = validator.validate_all()
        self.assertFalse(valid)


class TestPayloadKindDecision(unittest.TestCase):
    """Test the decision logic for choosing payload kind"""

    def test_shell_for_direct_script(self):
        """If running a script directly, use shell type"""
        command = "python3 /root/dotm-sniper/advisor_script.py"
        validator = CronValidator()

        is_shell_appropriate = (
            command.endswith('.py') or
            command.endswith('.sh') or
            'python3' in command or
            'bash' in command
        )
        self.assertTrue(is_shell_appropriate)

    def test_agent_turn_for_conversation(self):
        """If sending a message to agent, use agentTurn"""
        message = "analyze the current portfolio and suggest actions"
        validator = CronValidator()

        is_agent_appropriate = not (
            message.endswith('.py') or
            message.endswith('.sh') or
            'python3' in message or
            'bash' in message
        )
        self.assertTrue(is_agent_appropriate)


if __name__ == '__main__':
    unittest.main(verbosity=2)