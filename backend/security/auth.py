# Backend/security/auth.py
from fastapi import HTTPException
import os

def verify_token(authorization: str) -> str:
    """Verify iOS device token. Returns user_id."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ", 1)[1]
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    # PRODUCTION: Verify JWT, check against user database, validate device binding
    # For development: accept any non-empty token
    return "dev_user_001"
