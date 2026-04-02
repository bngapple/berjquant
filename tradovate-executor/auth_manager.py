"""
Auth Manager — handles Tradovate authentication for multiple accounts.
Each account gets its own token lifecycle (obtain, renew, expiry tracking).
"""

import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

from config import AccountConfig, AppConfig

logger = logging.getLogger(__name__)


@dataclass
class AuthSession:
    """Live auth state for one Tradovate account."""
    account_config: AccountConfig
    access_token: Optional[str] = None
    md_access_token: Optional[str] = None    # Separate token for market data WS
    expiry_ts: float = 0.0                  # Unix timestamp
    tradovate_account_id: Optional[int] = None  # Numeric account ID for orders
    user_id: Optional[int] = None
    is_authenticated: bool = False

    @property
    def is_expired(self) -> bool:
        return time.time() >= (self.expiry_ts - 60)  # Renew 60s before expiry

    @property
    def name(self) -> str:
        return self.account_config.name


class AuthManager:
    """
    Manages authentication sessions for all configured accounts.
    Handles initial login, token renewal, and provides tokens on demand.
    """

    TOKEN_LIFETIME = 3600  # Tradovate tokens last ~1 hour
    RENEW_MARGIN = 300     # Renew 5 min before expiry

    def __init__(self, config: AppConfig):
        self.config = config
        self.sessions: dict[str, AuthSession] = {}  # keyed by account name
        self._http = httpx.AsyncClient(timeout=15.0)
        self._renew_tasks: dict[str, asyncio.Task] = {}
        # Optional callback: called after every successful token renewal or re-auth.
        # Signature: on_token_renewed(session: AuthSession) -> None (sync or async)
        self.on_token_renewed: Optional[Callable] = None

    async def authenticate_all(self):
        """Authenticate every configured account in parallel."""
        tasks = []
        for acct in self.config.accounts:
            session = AuthSession(account_config=acct)
            self.sessions[acct.name] = session
            tasks.append(self._authenticate(session))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for acct, result in zip(self.config.accounts, results):
            if isinstance(result, Exception):
                logger.error(f"[{acct.name}] Auth FAILED: {result}")
            else:
                logger.info(f"[{acct.name}] Authenticated — account ID: {self.sessions[acct.name].tradovate_account_id}")

    async def _authenticate(self, session: AuthSession, start_renewal: bool = True):
        """POST /auth/accesstokenrequest for one account."""
        url = f"{self.config.rest_url}/auth/accesstokenrequest"
        payload = {
            "name": session.account_config.username,
            "password": session.account_config.password,
            "appId": session.account_config.app_id,
            "appVersion": session.account_config.app_version,
            "deviceId": session.account_config.device_id,
            "cid": session.account_config.cid,
            "sec": session.account_config.sec,
        }
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        if "errorText" in data:
            raise RuntimeError(f"Auth error: {data['errorText']}")

        session.access_token = data["accessToken"]
        session.md_access_token = data.get("mdAccessToken", data["accessToken"])
        session.expiry_ts = time.time() + self.TOKEN_LIFETIME
        session.user_id = data.get("userId")
        session.is_authenticated = True

        # Fetch account ID (needed for order placement)
        await self._fetch_account_id(session)

        # Start auto-renewal loop (skipped when called from within the renewal loop
        # to avoid self-cancellation of the currently running task)
        if start_renewal:
            self._start_renewal(session)

    async def _fetch_account_id(self, session: AuthSession):
        """GET /account/list to find the numeric account ID."""
        url = f"{self.config.rest_url}/account/list"
        headers = {"Authorization": f"Bearer {session.access_token}"}
        resp = await self._http.get(url, headers=headers)
        resp.raise_for_status()
        accounts = resp.json()
        if accounts:
            # Use the first active account
            session.tradovate_account_id = accounts[0]["id"]
            logger.debug(f"[{session.name}] Account ID resolved: {session.tradovate_account_id}")
        else:
            raise RuntimeError(f"[{session.name}] No accounts found")

    def _start_renewal(self, session: AuthSession):
        """Launch background task to renew token before expiry."""
        name = session.name
        if name in self._renew_tasks:
            self._renew_tasks[name].cancel()
        self._renew_tasks[name] = asyncio.create_task(self._renewal_loop(session))

    async def _renewal_loop(self, session: AuthSession):
        """Sleep until close to expiry, then renew."""
        while True:
            sleep_secs = max((session.expiry_ts - time.time()) - self.RENEW_MARGIN, 10)
            await asyncio.sleep(sleep_secs)
            try:
                await self._renew_token(session)
                logger.info(f"[{session.name}] Token renewed")
                self._fire_token_renewed(session)
            except Exception as e:
                logger.error(f"[{session.name}] Token renewal failed: {e}")
                # Re-auth without starting a new renewal task — this loop continues
                try:
                    await self._authenticate(session, start_renewal=False)
                    logger.info(f"[{session.name}] Re-authenticated after renewal failure")
                    self._fire_token_renewed(session)
                except Exception as e2:
                    logger.critical(f"[{session.name}] Re-auth also failed: {e2}")

    def _fire_token_renewed(self, session: AuthSession):
        """Invoke the on_token_renewed callback if set."""
        if not self.on_token_renewed:
            return
        result = self.on_token_renewed(session)
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)

    async def _renew_token(self, session: AuthSession):
        """POST /auth/renewaccesstoken."""
        url = f"{self.config.rest_url}/auth/renewaccesstoken"
        headers = {"Authorization": f"Bearer {session.access_token}"}
        resp = await self._http.post(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        if "errorText" in data:
            raise RuntimeError(data["errorText"])

        session.access_token = data["accessToken"]
        session.md_access_token = data.get("mdAccessToken", data["accessToken"])
        session.expiry_ts = time.time() + self.TOKEN_LIFETIME

    def get_session(self, account_name: str) -> Optional[AuthSession]:
        """Get auth session by account name."""
        return self.sessions.get(account_name)

    def get_master_session(self) -> Optional[AuthSession]:
        """Get the master account's session."""
        master = self.config.master_account
        if master:
            return self.sessions.get(master.name)
        return None

    def get_copy_sessions(self) -> list[AuthSession]:
        """Get all copy account sessions."""
        return [
            self.sessions[a.name]
            for a in self.config.copy_accounts
            if a.name in self.sessions and self.sessions[a.name].is_authenticated
        ]

    def get_all_sessions(self) -> list[AuthSession]:
        """All authenticated sessions."""
        return [s for s in self.sessions.values() if s.is_authenticated]

    async def shutdown(self):
        """Cancel all renewal tasks and close HTTP client."""
        for task in self._renew_tasks.values():
            task.cancel()
        self._renew_tasks.clear()
        await self._http.aclose()
