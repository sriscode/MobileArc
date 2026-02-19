# Backend/api/main.py
# FastAPI server for Chase Agentic AI cloud agent
# Run: uvicorn api.main:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import os

from agents.orchestrator import run_cloud_agent
from security.guardrails import LLMOutputGuardrails
from security.auth import verify_token

app = FastAPI(
    title="Chase Agentic AI Backend",
    description="Cloud agent orchestration for complex financial tasks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app-domain.com"],  # Restrict in production
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Request/Response Models ──────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    query: str = Field(..., max_length=1000)
    intent: str = Field(..., max_length=50)
    context_summary: str = Field(..., max_length=2000)
    approved_draft_ids: list[str] = Field(default_factory=list)
    session_id: str = Field(..., max_length=64)

class AgentAction(BaseModel):
    type: str
    payload: dict[str, str]

class AgentQueryResponse(BaseModel):
    text: str
    actions: list[AgentAction] = []
    session_id: str

class TransferExecuteRequest(BaseModel):
    draft_id: str
    confirmation_token: str

# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(
    request: AgentQueryRequest,
    authorization: str = Header(...)
):
    # Verify iOS auth token
    user_id = verify_token(authorization)

    try:
        result = await run_cloud_agent(
            user_message=request.query,
            intent=request.intent,
            user_context={
                "user_id": user_id,
                "context_summary": request.context_summary,
                "session_id": request.session_id,
            },
            approved_draft_ids=request.approved_draft_ids
        )

        return AgentQueryResponse(
            text=result["text"],
            actions=[AgentAction(**a) for a in result.get("actions", [])],
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/transfer/execute")
async def execute_transfer(
    request: TransferExecuteRequest,
    authorization: str = Header(...)
):
    user_id = verify_token(authorization)

    # Verify the draft exists and was approved by this user
    # In production: check against database
    if not request.draft_id or not request.confirmation_token:
        raise HTTPException(status_code=400, detail="Invalid request")

    # Execute via banking API
    from tools.banking_tools import execute_transfer_api
    result = await execute_transfer_api(
        draft_id=request.draft_id,
        user_id=user_id,
        confirmation_token=request.confirmation_token
    )
    return {"status": "executed", "result": result}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
