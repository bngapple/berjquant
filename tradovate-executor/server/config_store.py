"""
Config Store — CRUD for accounts in config.json with Fernet password encryption.
"""

import json
import os
from cryptography.fernet import Fernet

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.json")
KEY_FILE = os.environ.get("KEY_FILE", ".secret_key")

_DEFAULT_CONFIG = {
    "environment": "demo",
    "symbol": "MNQM6",
    "accounts": [],
}


def _get_fernet() -> Fernet:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            key = f.read().strip()
        # Fix permissions if they're too open
        if os.stat(KEY_FILE).st_mode & 0o077:
            os.chmod(KEY_FILE, 0o600)
    else:
        key = Fernet.generate_key()
        fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
    return Fernet(key)


def _encrypt(plaintext: str) -> str:
    if not plaintext:
        return plaintext
    return "enc:" + _get_fernet().encrypt(plaintext.encode()).decode()


def _decrypt(stored: str) -> str:
    if not stored:
        return stored
    if stored.startswith("enc:"):
        return _get_fernet().decrypt(stored[4:].encode()).decode()
    return stored


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return json.loads(json.dumps(_DEFAULT_CONFIG))
    with open(CONFIG_PATH) as f:
        data = json.load(f)
    for acct in data.get("accounts", []):
        acct["password"] = _decrypt(acct.get("password", ""))
        acct["sec"] = _decrypt(acct.get("sec", ""))
    return data


def save_config(data: dict):
    save_data = json.loads(json.dumps(data))
    for acct in save_data.get("accounts", []):
        pw = acct.get("password", "")
        if pw and not pw.startswith("enc:"):
            acct["password"] = _encrypt(pw)
        sec = acct.get("sec", "")
        if sec and not sec.startswith("enc:"):
            acct["sec"] = _encrypt(sec)
    with open(CONFIG_PATH, "w") as f:
        json.dump(save_data, f, indent=2)


def get_accounts() -> list[dict]:
    return load_config().get("accounts", [])


def get_account(name: str) -> dict | None:
    for acct in get_accounts():
        if acct["name"] == name:
            return acct
    return None


def add_account(acct: dict):
    config = load_config()
    for existing in config["accounts"]:
        if existing["name"] == acct["name"]:
            raise ValueError(f"Account '{acct['name']}' already exists")
    config["accounts"].append(acct)
    save_config(config)


def update_account(name: str, updates: dict) -> dict:
    config = load_config()
    for acct in config["accounts"]:
        if acct["name"] == name:
            for k, v in updates.items():
                if v is not None:
                    acct[k] = v
            save_config(config)
            return acct
    raise ValueError(f"Account '{name}' not found")


def delete_account(name: str):
    config = load_config()
    config["accounts"] = [a for a in config["accounts"] if a["name"] != name]
    save_config(config)


def mask_account(acct: dict) -> dict:
    masked = dict(acct)
    masked["password"] = "********"
    if masked.get("sec"):
        masked["sec"] = "********"
    # Ensure new fields have defaults
    masked.setdefault("starting_balance", 150000.0)
    masked.setdefault("profit_target", 9000.0)
    masked.setdefault("max_drawdown", -4500.0)
    masked.setdefault("account_type", "eval")
    masked.setdefault("monthly_loss_limit", -4500.0)
    masked.setdefault("min_contracts", 1)
    return masked


def get_environment() -> str:
    return load_config().get("environment", "demo")


def set_environment(env: str):
    config = load_config()
    config["environment"] = env
    save_config(config)
