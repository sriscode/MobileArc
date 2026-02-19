# Backend/tools/banking_tools.py
# All banking API integrations for the cloud agent
# Replace mock data with real Chase API calls

import asyncio
import json
import httpx
from datetime import datetime, timedelta
import os

CHASE_API_BASE = os.getenv("CHASE_API_BASE", "https://api.chase.com/v1")
CHASE_API_KEY  = os.getenv("CHASE_API_KEY", "")


async def get_account_summary(user_id: str, account_type: str = "all") -> str:
    """Fetch account balances. Never returns raw card/account numbers."""
    # PRODUCTION: Call Chase Accounts API with OAuth token
    # async with httpx.AsyncClient() as client:
    #     resp = await client.get(f"{CHASE_API_BASE}/accounts", headers=auth_headers(user_id))
    #     data = resp.json()

    # MOCK
    await asyncio.sleep(0.1)
    accounts = {
        "checking": {"balance": 4821.50, "available": 4721.50, "last4": "4821"},
        "savings":  {"balance": 12450.00, "available": 12450.00, "last4": "9034"},
        "credit":   {"balance": -1240.00, "available": 8760.00, "last4": "1337"},
    }

    if account_type != "all" and account_type in accounts:
        a = accounts[account_type]
        return f"{account_type.title()} (••••{a['last4']}): ${a['balance']:.2f} current, ${a['available']:.2f} available"

    lines = []
    for atype, a in accounts.items():
        lines.append(f"• {atype.title()} (••••{a['last4']}): ${a['balance']:.2f}")
    return "Account balances:\n" + "\n".join(lines)


async def execute_transfer_api(draft_id: str, user_id: str, confirmation_token: str) -> str:
    """Execute a previously staged transfer. Requires approval token."""
    await asyncio.sleep(0.2)
    # PRODUCTION: POST to Chase Transfers API
    return f"Transfer {draft_id} executed successfully. Confirmation: TXN_{draft_id[-8:].upper()}"


async def get_investment_portfolio(user_id: str) -> str:
    """Fetch J.P. Morgan investment portfolio."""
    await asyncio.sleep(0.2)
    return """Investment Portfolio — J.P. Morgan Wealth Management:
• Total Value: $24,850.32 (+$340.12 today, +1.39%)
• S&P 500 Index (VFIAX): 45% — $11,182.64 — +0.8% today
• Int'l Equity (VTIAX):  20% — $4,970.06  — +0.3% today
• Bonds (VBTLX):         25% — $6,212.58  — flat
• Cash/MMF:              10% — $2,485.04  — 5.02% APY
YTD Return: +8.2% | Since inception: +34.1%"""


async def get_market_rates(rate_type: str) -> str:
    """Fetch current savings and lending rates."""
    await asyncio.sleep(0.1)
    rates = {
        "hysa":       "Chase HYSA: 0.01% | Best market (SoFi/Marcus): 5.25% | National avg: 0.58%",
        "cd_6mo":     "Chase 6-month CD: 4.25% APY | Best market: 5.40% | Min: $1,000",
        "cd_1yr":     "Chase 1-year CD: 4.50% APY | Best market: 5.35% | Min: $1,000",
        "money_market": "Chase Money Market: 0.01% | Best market: 5.15% | National avg: 0.64%",
    }
    return rates.get(rate_type, "Rate data unavailable")


async def file_fraud_dispute(transaction_id: str, reason: str, user_id: str) -> str:
    """File formal fraud dispute and request provisional credit."""
    await asyncio.sleep(0.3)
    case_id = f"DISP-{transaction_id[-6:].upper()}-{datetime.now().strftime('%m%d')}"
    return f"""Fraud dispute filed:
Case ID: {case_id}
Transaction: {transaction_id}
Reason: {reason}
Provisional credit: Applied within 1-2 business days
Resolution: 10 business days (Visa Zero Liability)
Reference: {case_id}"""


async def get_spending_analysis(user_id: str, days: int, category: str = "all") -> str:
    """Run spending analysis for a period."""
    await asyncio.sleep(0.2)
    return f"""Spending Analysis — Last {days} days:
Total spent: $847.32
vs prior period: +$124.50 (+17.2%)

By category:
• Dining:      $187.15 (22%)
• Groceries:   $203.84 (24%)
• Transport:    $89.40 (11%)
• Shopping:    $156.23 (18%)
• Entertainment: $89.99 (11%)
• Utilities:    $120.71 (14%)

Top merchants: Whole Foods ($187), Chipotle ($89), Amazon ($156)
Insight: Dining up 34% vs last month — 6 more transactions"""
