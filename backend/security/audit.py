# Backend/security/audit.py
# Immutable audit logging — every agent action recorded
# PRODUCTION: Replace with AWS QLDB or CloudTrail

import json
import asyncio
from datetime import datetime
import os

class AuditLogger:
    def __init__(self):
        self.log_file = os.getenv("AUDIT_LOG_PATH", "audit.jsonl")

    async def log(self, event: str, metadata: dict) -> None:
        """Append-only audit record. No PII in metadata."""
        entry = {
            "event":     event,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata":  metadata
        }
        # PRODUCTION: Write to AWS QLDB for immutable ledger
        # await qldb_client.execute_statement("INSERT INTO AuditLog ?", entry)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
