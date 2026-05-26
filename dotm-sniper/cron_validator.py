#!/usr/bin/env python3
"""
Cron Job Validator - validates cron configuration before deployment
Run this before creating or modifying cron jobs
"""
import json
import os
import sys
from pathlib import Path

ALLOWED_WORKSPACE_PATHS = [
    '/root/.openclaw/workspace',
    '/root/dotm-sniper',
]

ALLOWED_SCRIPT_EXTENSIONS = ['.py', '.sh', '.js']

PAYLOAD_KIND_REQUIREMENTS = {
    'shell': {
        'requires_exec': True,
        'file_access': True,
        'description': 'Runs script directly in shell - no agent context'
    },
    'agentTurn': {
        'requires_exec': False,
        'file_access': True,
        'description': 'Routes to agent session - agent must have exec tool'
    },
    'systemEvent': {
        'requires_exec': False,
        'file_access': False,
        'description': 'System event trigger'
    }
}

EDGE_CASE_PATTERNS = {
    'exec_unavailable': [
        'no exec tool available',
        'exec tool: NOT AVAILABLE',
        'no shell access'
    ],
    'path_escapes_sandbox': [
        'Path escapes sandbox',
        'outside the workspace sandbox'
    ],
    'timeout': [
        'timeout',
        'timed out'
    ]
}

class CronValidator:
    def __init__(self, jobs_path=None):
        self.jobs_path = jobs_path or '/root/.openclaw/cron/jobs.json'
        self.errors = []
        self.warnings = []

    def load_jobs(self):
        try:
            with open(self.jobs_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.errors.append(f"Jobs file not found: {self.jobs_path}")
            return None
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in jobs file: {e}")
            return None

    def validate_payload(self, job):
        payload = job.get('payload', {})
        kind = payload.get('kind')

        if not kind:
            self.errors.append(f"Job {job.get('id')} has no payload.kind")
            return False

        if kind not in PAYLOAD_KIND_REQUIREMENTS:
            self.warnings.append(f"Job {job.get('id')} has unknown payload kind: {kind}")

        if kind == 'shell':
            command = payload.get('command', '')
            if not command:
                self.errors.append(f"Job {job.get('id')} is shell type but has no 'command' field")
                return False

            for path in ALLOWED_WORKSPACE_PATHS:
                if command.startswith(path) or f'cd {path}' in command:
                    return True

            self.warnings.append(
                f"Job {job.get('id')} shell command may reference non-workspace paths: {command[:50]}"
            )

        elif kind == 'agentTurn':
            message = payload.get('message', '')
            if not message:
                self.errors.append(f"Job {job.get('id')} is agentTurn but has no message")
                return False

            if message.strip().startswith('python'):
                self.warnings.append(
                    f"Job {job.get('id')} uses agentTurn with python script - consider using shell type instead"
                )

        return True

    def check_for_edge_cases(self, job):
        job_id = job.get('id')
        state = job.get('state', {})

        last_error = state.get('lastError', '')
        last_status = state.get('lastStatus', '')

        for case_name, patterns in EDGE_CASE_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in last_error.lower():
                    self.warnings.append(
                        f"Job {job_id} has historical edge case: {case_name} - {last_error[:80]}"
                    )

        consecutive_errors = state.get('consecutiveErrors', 0)
        if consecutive_errors >= 3:
            self.errors.append(
                f"Job {job_id} has {consecutive_errors} consecutive errors - needs attention"
            )

    def validate_job(self, job):
        if not job.get('id'):
            self.errors.append("Job missing required field: id")
            return False

        if not job.get('name'):
            self.errors.append(f"Job {job.get('id')} missing required field: name")

        if job.get('enabled') is None:
            self.errors.append(f"Job {job.get('id')} missing required field: enabled")

        schedule = job.get('schedule', {})
        if not schedule.get('kind'):
            self.errors.append(f"Job {job.get('id')} missing schedule.kind")

        self.validate_payload(job)
        self.check_for_edge_cases(job)

        return len([e for e in self.errors if job.get('id') in e]) == 0

    def validate_all(self):
        data = self.load_jobs()
        if not data:
            return False, self.errors, self.warnings

        jobs = data.get('jobs', [])
        if not jobs:
            self.warnings.append("No jobs found in configuration")

        all_valid = True
        for job in jobs:
            if not self.validate_job(job):
                all_valid = False

        return all_valid, self.errors, self.warnings

    def print_report(self):
        valid, errors, warnings = self.validate_all()

        print("=" * 60)
        print("CRON VALIDATION REPORT")
        print("=" * 60)

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"   - {w}")

        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for e in errors:
                print(f"   - {e}")

        if valid and not errors:
            print("\n✅ VALIDATION PASSED - all jobs configured correctly")

        return valid

def main():
    validator = CronValidator()

    if len(sys.argv) > 1:
        validator.jobs_path = sys.argv[1]

    is_valid = validator.print_report()

    print("\n" + "=" * 60)
    if is_valid:
        print("Status: PASS")
        sys.exit(0)
    else:
        print("Status: FAIL")
        sys.exit(1)

if __name__ == '__main__':
    main()