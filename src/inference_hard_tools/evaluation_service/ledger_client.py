"""Best-effort measured-artifact registration with the VDP ledger API."""

from __future__ import annotations

from typing import Any, Protocol

import httpx


class LedgerRegistrar(Protocol):
    def register(self, artifact: dict[str, Any]) -> None: ...


class HttpLedgerRegistrar:
    def __init__(
        self, base_url: str, writer_token: str, timeout_seconds: float
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/v1/artifacts"
        self.writer_token = writer_token
        self.timeout_seconds = timeout_seconds

    def register(self, artifact: dict[str, Any]) -> None:
        response = httpx.post(
            self.url,
            json=artifact,
            headers={"X-VDP-Service-Identity": self.writer_token},
            timeout=self.timeout_seconds,
            follow_redirects=False,
        )
        response.raise_for_status()
