# Cron Job Creation Checklist

Before creating or modifying a cron job, verify the following:

## 1. Choose the Correct Payload Kind

| If you want to... | Use | Why |
|-------------------|-----|-----|
| Run a script directly (python, bash, etc.) | `shell` | Executes without agent context, no exec tool needed |
| Send a message to an agent for processing | `agentTurn` | Routes to agent session; agent needs exec tool to run scripts |

### Decision Tree:
```
Is it a script file (.py, .sh)?
├── YES → Use `shell` payload with `command: "python3 /path/to/script.py"`
└── NO  → Is it a message for an agent to respond to?
          ├── YES → Use `agentTurn` payload with `message: "..."`
          └── NO  → Consider if cron is the right tool
```

## 2. Path Validation

- [ ] Script path is inside allowed workspace:
  - `/root/.openclaw/workspace/`
  - `/root/dotm-sniper/`
- [ ] If referencing external paths, ensure workspace has access

## 3. Required Fields

Every cron job must have:
- [ ] `id` - unique identifier
- [ ] `name` - descriptive name
- [ ] `enabled` - boolean
- [ ] `schedule.kind` - "at", "every", or "cron"
- [ ] `schedule` - schedule-specific fields (e.g., `everyMs`)
- [ ] `payload.kind` - "shell", "agentTurn", or "systemEvent"
- [ ] `payload` - payload-specific fields
- [ ] `sessionTarget` - "main" or "isolated"
- [ ] `wakeMode` - "now" or "next-heartbeat"

## 4. Shell Payload Requirements

If `payload.kind` is `shell`:
- [ ] Must have `command` field
- [ ] Command should include `cd /path && python3 script.py` or full path
- [ ] Script must exist and be executable

## 5. AgentTurn Payload Requirements

If `payload.kind` is `agentTurn`:
- [ ] Must have `message` field
- [ ] If message contains python/shell commands → use `shell` instead
- [ ] Agent must have exec tool available (check agent config)

## 6. Delivery Configuration

- [ ] `delivery.mode` - "none", "announce", or "webhook"
- [ ] `delivery.to` - target channel (e.g., "telegram:730132245")
- [ ] Consider `bestEffort: true` for non-critical jobs

## 7. Validation Before Deploy

Run the validator:
```bash
python3 /root/dotm-sniper/cron_validator.py
```

Fix any errors before saving the configuration.

## 8. Testing

After creating a cron job:
- [ ] Run manually to verify it works
- [ ] Check first run output for errors
- [ ] Verify delivery (Telegram message received)
- [ ] Check logs in `/root/.openclaw/cron/runs/`

## 9. Edge Cases to Watch For

| Pattern | Indicates | Action |
|---------|-----------|--------|
| "no exec tool available" | Agent lacks exec, use `shell` | Change payload to shell |
| "Path escapes sandbox" | Path outside workspace | Use paths within workspace |
| consecutiveErrors >= 3 | Job failing repeatedly | Check logs, fix configuration |
| "sendMessage failed" | Telegram delivery issue | Check bot token and chat ID |

## 10. Monitoring

Set up health monitoring:
```bash
# Run health monitor periodically via system cron
0 * * * * python3 /root/dotm-sniper/cron_health_monitor.py >> /root/cron_health.log 2>&1
```

---

## Quick Reference - Common Configurations

### Python Script (shell):
```json
{
  "payload": {
    "kind": "shell",
    "command": "cd /root/dotm-sniper && python3 advisor_script.py"
  }
}
```

### Agent Analysis (agentTurn):
```json
{
  "payload": {
    "kind": "agentTurn",
    "message": "Analyze the current portfolio and suggest actions"
  }
}
```

### System Event:
```json
{
  "payload": {
    "kind": "systemEvent",
    "text": "Custom event trigger"
  }
}
```