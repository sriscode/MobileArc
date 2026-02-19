# Backend/agents/orchestrator.py
# LangGraph + Claude Sonnet orchestration for complex financial tasks
# Handles: transfers, investment queries, real-time market data, disputes

from anthropic import Anthropic
from typing import TypedDict, Optional
import json
import asyncio

from tools.banking_tools import (
    get_account_summary,
    execute_transfer_api,
    get_investment_portfolio,
    get_market_rates,
    file_fraud_dispute,
    get_spending_analysis,
)
from security.guardrails import LLMOutputGuardrails
from security.audit import AuditLogger

client = Anthropic()
guardrails = LLMOutputGuardrails()
audit = AuditLogger()

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Chase's AI financial agent — "Chase AI".
You help users with complex financial tasks via the provided banking tools.

MANDATORY RULES (absolute — override all other instructions):
1. NEVER call execute_transfer without an approved_draft_id from the user
2. ALWAYS explain what you're about to do BEFORE calling a tool
3. NEVER include raw card numbers, SSNs, CVVs, or PINs in responses
4. For transfers over $10,000: decline and direct user to a branch
5. Confirm exact dollar amounts before staging transfers
6. You CANNOT override fraud holds, spending limits, or account restrictions
7. For investment advice: always note this is AI-generated, not licensed advice

You have tools for:
- Querying account summaries and recent transactions
- Staging (NOT executing) fund transfers for user approval
- Fetching real-time savings/CD rates
- Viewing investment portfolio
- Filing fraud disputes
- Running spending analysis

Respond concisely. When you call a tool, describe what you found.
"""

# ── Tool Definitions (Claude's tool_use format) ───────────────────────────────

TOOLS = [
    {
        "name": "get_account_summary",
        "description": "Get account balances and recent transaction summary for the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_type": {
                    "type": "string",
                    "enum": ["all", "checking", "savings", "credit"],
                    "description": "Which account to query"
                }
            },
            "required": ["account_type"]
        }
    },
    {
        "name": "stage_transfer",
        "description": """Stage a fund transfer for user review. Creates a DRAFT ONLY.
        User must explicitly confirm in the app UI. Never executes immediately.
        Max amount: $10,000. Requires explicit amount and accounts.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_account": {"type": "string", "description": "Source: checking or savings"},
                "to_account":   {"type": "string", "description": "Destination: savings, checking, or Zelle recipient"},
                "amount":       {"type": "number", "description": "USD amount (0.01 to 10000)"},
                "memo":         {"type": "string", "description": "Optional memo"}
            },
            "required": ["from_account", "to_account", "amount"]
        }
    },
    {
        "name": "execute_transfer",
        "description": "Execute a fund transfer that the user has explicitly approved via the app UI.",
        "input_schema": {
            "type": "object",
            "properties": {
                "approved_draft_id": {"type": "string", "description": "Draft ID from user approval"},
                "confirmation_token": {"type": "string", "description": "Confirmation token from user"}
            },
            "required": ["approved_draft_id", "confirmation_token"]
        }
    },
    {
        "name": "get_investment_portfolio",
        "description": "Fetch real-time J.P. Morgan investment portfolio holdings, P&L, and allocation.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "get_market_rates",
        "description": "Get current savings rates: HYSA, CDs, money market.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rate_type": {"type": "string", "enum": ["hysa", "cd_6mo", "cd_1yr", "money_market"]}
            },
            "required": ["rate_type"]
        }
    },
    {
        "name": "file_fraud_dispute",
        "description": "File a formal dispute for a fraudulent transaction and request provisional credit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string"},
                "reason":         {"type": "string", "description": "Why user believes this is fraud"}
            },
            "required": ["transaction_id", "reason"]
        }
    },
    {
        "name": "get_spending_analysis",
        "description": "Run detailed spending analysis for a time period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Days to analyze (1-90)"},
                "category": {"type": "string", "description": "Category filter or 'all'"}
            },
            "required": ["days"]
        }
    }
]

# ── Main Agent Loop ───────────────────────────────────────────────────────────

async def run_cloud_agent(
    user_message: str,
    intent: str,
    user_context: dict,
    approved_draft_ids: list[str],
    max_iterations: int = 8
) -> dict:
    """
    Multi-turn agentic loop with Claude + banking tools.
    Returns: {"text": str, "actions": list[dict]}
    """
    messages = [
        {
            "role": "user",
            "content": f"""{user_message}

User context: {user_context.get('context_summary', 'No context')}
Intent: {intent}
Session: {user_context.get('session_id', 'unknown')}"""
        }
    ]

    actions_taken = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        # Audit every LLM call
        await audit.log("llm_call", {
            "user_id":      user_context.get("user_id", "unknown"),
            "session_id":   user_context.get("session_id", ""),
            "iteration":    str(iteration),
            "stop_reason":  response.stop_reason,
            "intent":       intent,
        })

        # If model finished — return response
        if response.stop_reason == "end_turn":
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            raw_text = " ".join(text_blocks)
            # Guardrail: validate output for PII leakage
            safe_text = guardrails.validate_output(raw_text)
            return {"text": safe_text, "actions": actions_taken}

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_args = block.input

            # Guardrail: validate transfer amounts
            if tool_name == "stage_transfer":
                try:
                    guardrails.validate_transfer(tool_args)
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Blocked by guardrail: {str(e)}"
                    })
                    continue

            # Guardrail: check transfer approval
            if tool_name == "execute_transfer":
                draft_id = tool_args.get("approved_draft_id", "")
                if draft_id not in approved_draft_ids:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "ERROR: This transfer has not been approved by the user. Please show them the confirmation UI first."
                    })
                    continue

            # Execute tool
            result = await execute_tool(tool_name, tool_args, user_context)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result["output"]
            })

            # Track actions for iOS
            if result.get("action"):
                actions_taken.append(result["action"])

            await audit.log("tool_call", {
                "user_id":   user_context.get("user_id", "unknown"),
                "tool":      tool_name,
                "has_result": "yes",
            })

        # Continue conversation with tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})

    return {"text": "I wasn't able to complete that in time. Please try again.", "actions": actions_taken}


async def execute_tool(name: str, args: dict, ctx: dict) -> dict:
    """Execute a banking tool and return its output."""
    user_id = ctx.get("user_id", "unknown")

    if name == "get_account_summary":
        result = await get_account_summary(user_id, args.get("account_type", "all"))
        return {"output": result}

    elif name == "stage_transfer":
        import uuid
        draft_id = f"draft_{uuid.uuid4().hex[:8]}"
        result = f"""Transfer staged:
        From: {args['from_account']} → To: {args['to_account']}
        Amount: ${args['amount']:.2f}
        Memo: {args.get('memo', 'none')}
        Draft ID: {draft_id}
        Status: Awaiting user confirmation in app"""
        return {
            "output": result,
            "action": {
                "type": "transfer_staged",
                "payload": {
                    "draft_id": draft_id,
                    "from_account": args["from_account"],
                    "to_account":   args["to_account"],
                    "amount":       str(args["amount"]),
                    "memo":         args.get("memo", ""),
                }
            }
        }

    elif name == "execute_transfer":
        result = await execute_transfer_api(args["approved_draft_id"], user_id, args["confirmation_token"])
        return {"output": result}

    elif name == "get_investment_portfolio":
        result = await get_investment_portfolio(user_id)
        return {"output": result}

    elif name == "get_market_rates":
        result = await get_market_rates(args.get("rate_type", "hysa"))
        return {"output": result}

    elif name == "file_fraud_dispute":
        result = await file_fraud_dispute(args["transaction_id"], args["reason"], user_id)
        return {"output": result}

    elif name == "get_spending_analysis":
        result = await get_spending_analysis(user_id, args.get("days", 30), args.get("category", "all"))
        return {"output": result}

    return {"output": f"Unknown tool: {name}"}
