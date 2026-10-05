"""Validated local backend configuration; reading it never provisions a model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from sol_chat.core.settings import Settings
from sol_chat.paths import CONFIG_DIR

HERE = CONFIG_DIR
DEFAULT_CONFIG = HERE / "local.json"


class ConfigError(ValueError):
    """A bounded startup error, without configuration values or tracebacks."""


def local_origin(value):
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ConfigError("Origins must be nonempty strings of at most 200 characters.")
    try:
        parsed = urlsplit(value)
        port = parsed.port
        value.encode("ascii")
    except (ValueError, UnicodeError) as exc:
        raise ConfigError("Invalid local origin.") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.netloc.endswith(":")
        or (port is not None and not 1 <= port <= 65535)
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in value)
    ):
        raise ConfigError("Only explicit HTTP(S) loopback origins are supported.")
    return value.rstrip("/")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError("Duplicate configuration keys are not supported.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ConfigError("Configuration must contain finite JSON numbers.")


def finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError("JSON numbers must be finite.")
    return result


@dataclass(frozen=True)
class BackendConfig:
    host: str
    port: int
    index_path: Path
    allowed_origins: tuple[str, ...]
    max_body_bytes: int
    engine: dict


def validate_config(value, base_dir=HERE):
    if not isinstance(value, dict):
        raise ConfigError("Backend configuration must be a JSON object.")
    fields = {
        "schema_version",
        "host",
        "port",
        "index_path",
        "allowed_origins",
        "max_body_bytes",
        "engine",
    }
    if set(value) - fields:
        raise ConfigError("Unknown backend configuration fields.")
    version = value.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ConfigError("Only backend schema_version 1 is supported.")
    host = value.get("host", "127.0.0.1")
    if host != "127.0.0.1":
        raise ConfigError("SOL binds to 127.0.0.1 only; public hosting is not configured.")
    port = value.get("port", 8765)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ConfigError("port must be an integer from 0 to 65535.")
    body = value.get("max_body_bytes", 32768)
    if isinstance(body, bool) or not isinstance(body, int) or not 1024 <= body <= 65536:
        raise ConfigError("max_body_bytes must be an integer from 1024 to 65536.")
    origins = value.get("allowed_origins", [])
    if not isinstance(origins, list) or len(origins) > 8:
        raise ConfigError("allowed_origins must be a list of at most eight local origins.")
    origins = tuple(dict.fromkeys(local_origin(origin) for origin in origins))
    index = value.get("index_path", "../outputs/auro_guide_corpus/local_rag.sqlite")
    if (
        not isinstance(index, str)
        or not index
        or len(index) > 4096
        or any(ord(char) < 32 for char in index)
    ):
        raise ConfigError("index_path must be a nonempty local file path.")
    if index.replace("\\", "/").startswith("//"):
        raise ConfigError("Network, UNC and device index paths are not supported.")
    try:
        index.encode("utf-8")
        index_path = Path(index)
        if not index_path.is_absolute():
            index_path = Path(base_dir) / index_path
        index_path = index_path.resolve()
    except (OSError, ValueError, UnicodeError) as exc:
        raise ConfigError("Invalid local index_path.") from exc
    engine = value.get("engine", {})
    if not isinstance(engine, dict):
        raise ConfigError("engine must be a JSON object.")
    try:
        settings = Settings.from_dict(engine)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ConfigError(
            "Invalid engine settings; use documented limits and an installed local model."
        ) from exc
    # Frontend contract caps can be tightened, never silently loosened by config.
    if (
        settings.max_question_characters > 1200
        or settings.max_history_messages > 8
        or settings.max_history_characters > 2000
        or settings.max_history_message_characters > 1000
    ):
        raise ConfigError("Engine question/history limits cannot exceed the backend API contract.")
    return BackendConfig(host, port, index_path, origins, body, asdict(settings))


def load_config(path=None):
    """Relative index paths resolve beside the config file, not the shell cwd."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if str(config_path).replace("\\", "/").startswith("//"):
        raise ConfigError("Configuration must be a local file, not a UNC or device path.")
    if path is None and not config_path.exists():
        return validate_config({})
    try:
        with config_path.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ConfigError("Backend configuration exceeds 65536 bytes.")
        value = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=finite_json_float,
        )
    except ConfigError:
        raise
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise ConfigError("Cannot read the backend configuration as a UTF-8 JSON object.") from exc
    return validate_config(value, config_path.resolve().parent)
