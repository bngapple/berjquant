import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig
from server.engine_bridge import EngineBridge


def test_app_config_load_reads_strategy_contract_overrides(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "environment": "live",
        "symbol": "MNQM6",
        "session": {"monthly_loss_limit": -1000.0},
        "rsi": {"contracts": 1},
        "ib": {"contracts": 1},
        "mom": {"contracts": 1},
        "accounts": [],
    }))

    cfg = AppConfig.load(str(config_path))

    assert cfg.rsi.contracts == 1
    assert cfg.ib.contracts == 1
    assert cfg.mom.contracts == 1
    assert cfg.session.monthly_loss_limit == -1000.0
    assert cfg.rsi.take_profit_pts == 100.0
    assert cfg.ib.take_profit_pts == 120.0
    assert cfg.mom.take_profit_pts == 100.0


def test_engine_bridge_build_config_reads_strategy_contract_overrides(monkeypatch):
    monkeypatch.setattr(
        "server.engine_bridge.config_store.load_config",
        lambda: {
            "environment": "live",
            "symbol": "MNQM6",
            "session": {"monthly_loss_limit": -1000.0},
            "rsi": {"contracts": 1},
            "ib": {"contracts": 1},
            "mom": {"contracts": 1},
            "accounts": [],
        },
    )

    cfg = EngineBridge()._build_config()

    assert cfg.rsi.contracts == 1
    assert cfg.ib.contracts == 1
    assert cfg.mom.contracts == 1
    assert cfg.session.monthly_loss_limit == -1000.0
