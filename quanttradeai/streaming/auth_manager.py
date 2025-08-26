"""Authentication and authorization utilities for data providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional


@dataclass
class AuthManager:
    """Simple authentication manager with token refresh support."""

    provider: str
    _token: Optional[str] = None
    _expires_at: datetime = datetime.fromtimestamp(0)

    def _load_credentials(self) -> Dict[str, str]:
        """Load credentials from environment variables."""
        key = os.getenv(f"{self.provider.upper()}_API_KEY", "")
        secret = os.getenv(f"{self.provider.upper()}_API_SECRET", "")
        return {"key": key, "secret": secret}

    def _token_needs_refresh(self) -> bool:
        return datetime.utcnow() >= self._expires_at - timedelta(minutes=5)

    async def _refresh_token(self) -> None:
        creds = self._load_credentials()
        # In production, call provider-specific auth endpoint.
        self._token = creds.get("key")
        self._expires_at = datetime.utcnow() + timedelta(hours=1)

    async def get_auth_headers(self) -> Dict[str, str]:
        if self._token is None or self._token_needs_refresh():
            await self._refresh_token()
        return {"Authorization": f"Bearer {self._token}"}
