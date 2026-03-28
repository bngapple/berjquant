import json
import os
import pytest

# Point to test config file to avoid polluting real config
os.environ["CONFIG_PATH"] = "test_config.json"
os.environ["KEY_FILE"] = "test_secret_key"

from server import config_store


@pytest.fixture(autouse=True)
def clean_test_files():
    """Remove test files before and after each test."""
    for f in ["test_config.json", "test_secret_key"]:
        if os.path.exists(f):
            os.remove(f)
    yield
    for f in ["test_config.json", "test_secret_key"]:
        if os.path.exists(f):
            os.remove(f)


class TestConfigStore:
    def test_load_missing_config_returns_defaults(self):
        data = config_store.load_config()
        assert data["environment"] == "demo"
        assert data["symbol"] == "MNQM6"
        assert data["accounts"] == []

    def test_add_and_get_account(self):
        acct = {
            "name": "test-1",
            "username": "user1",
            "password": "pass1",
            "device_id": "dev-1",
            "app_id": "HTFSwing",
            "app_version": "1.0.0",
            "cid": 123,
            "sec": "mysecret",
            "is_master": True,
            "sizing_mode": "mirror",
            "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        }
        config_store.add_account(acct)
        result = config_store.get_account("test-1")
        assert result is not None
        assert result["username"] == "user1"
        assert result["password"] == "pass1"  # Decrypted on read

    def test_passwords_encrypted_on_disk(self):
        config_store.add_account({
            "name": "enc-test",
            "username": "u",
            "password": "secret_password",
            "device_id": "d",
            "app_id": "HTFSwing",
            "app_version": "1.0.0",
            "cid": 0,
            "sec": "api_secret",
            "is_master": False,
            "sizing_mode": "mirror",
            "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        })
        with open("test_config.json") as f:
            raw = json.load(f)
        raw_acct = raw["accounts"][0]
        assert raw_acct["password"].startswith("enc:")
        assert raw_acct["sec"].startswith("enc:")
        assert raw_acct["password"] != "secret_password"

    def test_add_duplicate_name_raises(self):
        config_store.add_account({
            "name": "dup", "username": "u", "password": "p", "device_id": "d",
            "app_id": "HTFSwing", "app_version": "1.0.0", "cid": 0, "sec": "",
            "is_master": False, "sizing_mode": "mirror", "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        })
        with pytest.raises(ValueError, match="already exists"):
            config_store.add_account({
                "name": "dup", "username": "u2", "password": "p2", "device_id": "d2",
                "app_id": "HTFSwing", "app_version": "1.0.0", "cid": 0, "sec": "",
                "is_master": False, "sizing_mode": "mirror", "account_size": 150000.0,
                "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
            })

    def test_update_account(self):
        config_store.add_account({
            "name": "upd", "username": "u", "password": "p", "device_id": "d",
            "app_id": "HTFSwing", "app_version": "1.0.0", "cid": 0, "sec": "",
            "is_master": False, "sizing_mode": "mirror", "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        })
        result = config_store.update_account("upd", {"sizing_mode": "fixed", "account_size": 50000.0})
        assert result["sizing_mode"] == "fixed"
        assert result["account_size"] == 50000.0

    def test_update_nonexistent_raises(self):
        with pytest.raises(ValueError, match="not found"):
            config_store.update_account("nope", {"sizing_mode": "fixed"})

    def test_delete_account(self):
        config_store.add_account({
            "name": "del-me", "username": "u", "password": "p", "device_id": "d",
            "app_id": "HTFSwing", "app_version": "1.0.0", "cid": 0, "sec": "",
            "is_master": False, "sizing_mode": "mirror", "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        })
        config_store.delete_account("del-me")
        assert config_store.get_account("del-me") is None

    def test_mask_account(self):
        acct = {"name": "x", "password": "secret", "sec": "api_key", "username": "u"}
        masked = config_store.mask_account(acct)
        assert masked["password"] == "********"
        assert masked["sec"] == "********"
        assert masked["username"] == "u"

    def test_environment_get_set(self):
        assert config_store.get_environment() == "demo"
        config_store.set_environment("live")
        assert config_store.get_environment() == "live"

    def test_backwards_compat_plaintext_passwords(self):
        with open("test_config.json", "w") as f:
            json.dump({
                "environment": "demo", "symbol": "MNQM6",
                "accounts": [{
                    "name": "plain", "username": "u", "password": "plaintext_pass",
                    "device_id": "d", "app_id": "HTFSwing", "app_version": "1.0.0",
                    "cid": 0, "sec": "plain_sec", "is_master": False,
                    "sizing_mode": "mirror", "account_size": 150000.0,
                    "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
                }]
            }, f)
        acct = config_store.get_account("plain")
        assert acct["password"] == "plaintext_pass"
        assert acct["sec"] == "plain_sec"
