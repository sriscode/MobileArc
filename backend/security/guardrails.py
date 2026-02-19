# Backend/security/guardrails.py
# LLM output validation and transfer guardrails
# Every Claude response passes through here before reaching the iOS client

import re
from typing import Any

class GuardrailViolation(Exception):
    pass

class LLMOutputGuardrails:

    # PII patterns that must never appear in LLM output
    PII_PATTERNS = [
        (r'\b\d{16}\b',               "16-digit card number"),
        (r'\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b', "formatted card number"),
        (r'\b\d{3}-\d{2}-\d{4}\b',   "SSN formatted"),
        (r'\b(?<![\d])\d{9}(?![\d])\b', "9-digit SSN/routing"),
        (r'(?i)\bcvv\b[\s:]+\d{3,4}', "CVV value"),
        (r'(?i)\bpin\b[\s:]+\d{4,6}', "PIN value"),
        (r'(?i)password[\s:=]+\S+',   "password"),
        (r'(?i)secret[\s:=]+\S+',     "secret"),
        (r'(?i)api[_\s]?key[\s:=]+\S+', "API key"),
    ]

    TRANSFER_MAX_AMOUNT = 10_000.0
    TRANSFER_MIN_AMOUNT = 0.01

    def validate_output(self, text: str) -> str:
        """Scan LLM output for PII — raises GuardrailViolation or returns cleaned text."""
        for pattern, label in self.PII_PATTERNS:
            if re.search(pattern, text):
                # In production: log security event, alert, and redact
                # For now: raise to prevent output from reaching client
                raise GuardrailViolation(f"Blocked: possible {label} in response")
        return text

    def validate_transfer(self, args: dict) -> dict:
        """Validate transfer tool arguments before execution."""
        amount = float(args.get("amount", 0))

        if amount < self.TRANSFER_MIN_AMOUNT:
            raise GuardrailViolation(f"Transfer amount must be at least ${self.TRANSFER_MIN_AMOUNT}")

        if amount > self.TRANSFER_MAX_AMOUNT:
            raise GuardrailViolation(
                f"AI-initiated transfers are limited to ${self.TRANSFER_MAX_AMOUNT:,.0f}. "
                "For larger amounts, please visit a Chase branch or call 1-800-935-9935."
            )

        from_acct = args.get("from_account", "").lower()
        to_acct   = args.get("to_account", "").lower()

        if not from_acct or not to_acct:
            raise GuardrailViolation("Both from_account and to_account are required")

        # Prevent same-account transfers
        if from_acct == to_acct:
            raise GuardrailViolation("Cannot transfer to the same account")

        return args

    def sanitize_context(self, context: dict) -> dict:
        """Remove any PII from context before passing to LLM."""
        sanitized = {}
        for key, value in context.items():
            if isinstance(value, str):
                # Redact anything matching PII patterns
                v = value
                for pattern, label in self.PII_PATTERNS:
                    v = re.sub(pattern, f"[{label.upper().replace(' ', '_')}-REDACTED]", v)
                sanitized[key] = v
            else:
                sanitized[key] = value
        return sanitized
