"""Save, load, and manage strategy definition files as JSON."""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from strategies.generator import GeneratedStrategy

logger = logging.getLogger(__name__)


class StrategySerializer:
    """Save, load, and manage strategy definition files."""

    def __init__(self, strategies_dir: str | Path = "strategies/saved"):
        """Initialize with directory for storing strategy JSON files."""
        self.strategies_dir = Path(strategies_dir)
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

    def save(self, strategy, tag: str | None = None) -> Path:
        """
        Save a strategy definition to JSON.

        Filename: {strategy.name}_{tag}.json or {strategy.name}_{timestamp}.json
        Returns the path to the saved file.
        """
        suffix = tag if tag else datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy.name}_{suffix}.json"
        filepath = self.strategies_dir / filename

        data = {
            "saved_at": datetime.now().isoformat(),
            "strategy": strategy.to_dict(),
        }

        try:
            filepath.write_text(json.dumps(data, indent=2, default=str))
            logger.info("Saved strategy to %s", filepath)
        except OSError as e:
            logger.error("Failed to save strategy to %s: %s", filepath, e)
            raise

        return filepath

    def load(self, filepath: str | Path):
        """Load a single strategy from a JSON file. Returns GeneratedStrategy."""
        filepath = Path(filepath)

        try:
            data = json.loads(filepath.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load strategy from %s: %s", filepath, e)
            raise

        return GeneratedStrategy.from_dict(data["strategy"])

    def load_all(self) -> list:
        """Load all strategies from the strategies directory."""
        strategies = []
        for fp in sorted(self.strategies_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text())
                # Skip batch/leaderboard files
                if "strategy" in data:
                    strategies.append(GeneratedStrategy.from_dict(data["strategy"]))
            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping %s: %s", fp.name, e)
        return strategies

    def save_batch(self, strategies: list, batch_name: str) -> Path:
        """
        Save a batch of strategies to a single JSON file.

        Useful for saving a generation run or top-N leaderboard.
        Returns path to saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_{batch_name}_{timestamp}.json"
        filepath = self.strategies_dir / filename

        data = {
            "saved_at": datetime.now().isoformat(),
            "batch_name": batch_name,
            "count": len(strategies),
            "strategies": [s.to_dict() for s in strategies],
        }

        try:
            filepath.write_text(json.dumps(data, indent=2, default=str))
            logger.info("Saved batch of %d strategies to %s", len(strategies), filepath)
        except OSError as e:
            logger.error("Failed to save batch to %s: %s", filepath, e)
            raise

        return filepath

    def load_batch(self, filepath: str | Path) -> list:
        """Load a batch of strategies from a batch JSON file."""
        filepath = Path(filepath)

        try:
            data = json.loads(filepath.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load batch from %s: %s", filepath, e)
            raise

        return [GeneratedStrategy.from_dict(s) for s in data["strategies"]]

    def save_leaderboard(self, results: list, filepath: str | Path | None = None) -> Path:
        """
        Save ranked backtest results (strategy + metrics) as a leaderboard JSON.

        Each entry: {"rank": int, "strategy": dict, "metrics": dict}

        results: list of (GeneratedStrategy, PerformanceMetrics) tuples
                 or BacktestResult objects.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.strategies_dir / f"leaderboard_{timestamp}.json"
        else:
            filepath = Path(filepath)

        entries = []
        for rank, item in enumerate(results, start=1):
            if isinstance(item, tuple):
                strategy, metrics = item
                entry = {
                    "rank": rank,
                    "strategy": strategy.to_dict(),
                    "metrics": _metrics_to_dict(metrics),
                }
            else:
                # Assume BacktestResult
                entry = {
                    "rank": rank,
                    "strategy_name": item.strategy_name,
                    "metrics": _metrics_to_dict(item.metrics) if item.metrics else {},
                }
                # If the BacktestResult has a strategy attribute with to_dict, include it
                if hasattr(item, "strategy") and hasattr(item.strategy, "to_dict"):
                    entry["strategy"] = item.strategy.to_dict()
            entries.append(entry)

        data = {
            "saved_at": datetime.now().isoformat(),
            "type": "leaderboard",
            "count": len(entries),
            "entries": entries,
        }

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(json.dumps(data, indent=2, default=str))
            logger.info("Saved leaderboard with %d entries to %s", len(entries), filepath)
        except OSError as e:
            logger.error("Failed to save leaderboard to %s: %s", filepath, e)
            raise

        return filepath

    def load_leaderboard(self, filepath: str | Path) -> list:
        """Load a leaderboard file. Returns list of entry dicts."""
        filepath = Path(filepath)

        try:
            data = json.loads(filepath.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load leaderboard from %s: %s", filepath, e)
            raise

        return data["entries"]

    def list_saved(self) -> list[dict]:
        """List all saved strategy files with metadata (name, date, path)."""
        results = []
        for fp in sorted(self.strategies_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text())
                info = {
                    "path": str(fp),
                    "filename": fp.name,
                    "saved_at": data.get("saved_at", "unknown"),
                    "type": data.get("type", "strategy"),
                }
                # Add strategy name if available
                if "strategy" in data and isinstance(data["strategy"], dict):
                    info["name"] = data["strategy"].get("name", fp.stem)
                elif "batch_name" in data:
                    info["name"] = data["batch_name"]
                    info["type"] = "batch"
                    info["count"] = data.get("count", 0)
                else:
                    info["name"] = fp.stem
                results.append(info)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping %s: %s", fp.name, e)

        return results

    def delete(self, filepath: str | Path) -> bool:
        """Delete a saved strategy file. Returns True if deleted."""
        filepath = Path(filepath)
        try:
            filepath.unlink()
            logger.info("Deleted %s", filepath)
            return True
        except FileNotFoundError:
            logger.warning("File not found: %s", filepath)
            return False
        except OSError as e:
            logger.error("Failed to delete %s: %s", filepath, e)
            return False


def _metrics_to_dict(metrics) -> dict:
    """Convert a PerformanceMetrics dataclass to a JSON-safe dict."""
    if metrics is None:
        return {}
    try:
        return asdict(metrics)
    except TypeError:
        # Fallback if not a dataclass
        return {k: v for k, v in vars(metrics).items() if not k.startswith("_")}
