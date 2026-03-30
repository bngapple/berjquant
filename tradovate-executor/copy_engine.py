"""
Copy Engine — replicates master account fills to all linked copy accounts.
Each copy account gets its own OrderExecutor with per-account sizing.
Failed copies are skipped and logged (no retry).
"""

import asyncio
import logging
from typing import Optional

from auth_manager import AuthManager, AuthSession
from order_executor import OrderExecutor, OrderRecord
from signal_engine import Signal, Side
from config import AppConfig

logger = logging.getLogger(__name__)


class CopyResult:
    """Result of a copy attempt to one account."""

    def __init__(self, account_name: str):
        self.account_name = account_name
        self.success: bool = False
        self.order_record: Optional[OrderRecord] = None
        self.error: Optional[str] = None
        self.contracts: int = 0
        self.skipped: bool = False  # True if 0 contracts (e.g., scaled too small)


class CopyEngine:
    """
    Mirrors master account executions to all copy accounts.
    Uses per-account sizing from AccountConfig.
    """

    def __init__(self, config: AppConfig, auth_manager: AuthManager):
        self.config = config
        self.auth = auth_manager
        self.executors: dict[str, OrderExecutor] = {}  # account_name → executor

    async def initialize(self):
        """Create an OrderExecutor for each copy account."""
        for session in self.auth.get_copy_sessions():
            executor = OrderExecutor(self.config, session)
            self.executors[session.name] = executor
            logger.info(f"[Copy] Executor ready for: {session.name}")

    async def copy_entry(self, signal: Signal) -> list[CopyResult]:
        """
        Replicate an entry signal to all copy accounts.
        Returns list of CopyResults (one per account).
        """
        results = []
        tasks = []

        for acct in self.config.copy_accounts:
            name = acct.name
            executor = self.executors.get(name)
            if not executor:
                result = CopyResult(name)
                result.error = "No executor initialized"
                results.append(result)
                continue

            # Calculate per-account contract size
            contracts = acct.get_contracts(signal.strategy, signal.contracts)

            if contracts <= 0:
                result = CopyResult(name)
                result.skipped = True
                result.contracts = 0
                results.append(result)
                logger.info(f"[Copy][{name}] Skipped — 0 contracts for {signal.strategy}")
                continue

            # Fire copy order (all accounts in parallel)
            tasks.append((name, contracts, executor))

        # Execute all copies concurrently
        if tasks:
            copy_coros = [
                self._execute_copy(name, executor, signal, contracts)
                for name, contracts, executor in tasks
            ]
            copy_results = await asyncio.gather(*copy_coros, return_exceptions=True)

            for (name, contracts, _), result in zip(tasks, copy_results):
                if isinstance(result, Exception):
                    cr = CopyResult(name)
                    cr.error = str(result)
                    cr.contracts = contracts
                    results.append(cr)
                    logger.error(f"[Copy][{name}] FAILED: {result}")
                else:
                    results.append(result)

        # Log summary
        ok = sum(1 for r in results if r.success)
        fail = sum(1 for r in results if r.error)
        skip = sum(1 for r in results if r.skipped)
        logger.info(
            f"[Copy] {signal.strategy} {signal.side.value} → "
            f"{ok} ok, {fail} failed, {skip} skipped"
        )

        return results

    async def _execute_copy(
        self,
        account_name: str,
        executor: OrderExecutor,
        signal: Signal,
        contracts: int,
    ) -> CopyResult:
        """Place entry + bracket on one copy account."""
        result = CopyResult(account_name)
        result.contracts = contracts

        try:
            record = await executor.place_entry_with_bracket(signal, contracts)
            if record and record.order_id:
                result.success = True
                result.order_record = record
                logger.info(f"[Copy][{account_name}] ✓ {signal.side.value} {contracts}")
            else:
                result.error = "Order rejected or no order ID"
        except Exception as e:
            result.error = str(e)

        return result

    async def copy_flatten(self, strategy: str = "") -> list[CopyResult]:
        """Flatten positions on all copy accounts."""
        results = []
        tasks = []

        for name, executor in self.executors.items():
            tasks.append((name, executor))

        flatten_coros = [
            executor.flatten_position(strategy)
            for name, executor in tasks
        ]
        flatten_results = await asyncio.gather(*flatten_coros, return_exceptions=True)

        for (name, _), success in zip(tasks, flatten_results):
            result = CopyResult(name)
            if isinstance(success, Exception):
                result.error = str(success)
            else:
                result.success = bool(success)
            results.append(result)

        return results

    async def cancel_all_on_copies(self):
        """Cancel all working orders on all copy accounts."""
        tasks = [
            executor.cancel_all_orders()
            for executor in self.executors.values()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self):
        """Close all copy executors."""
        for executor in self.executors.values():
            await executor.shutdown()
