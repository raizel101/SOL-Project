"""Serve only four known UI assets, never a user-selected file path."""

from __future__ import annotations

from sol_chat.paths import FRONTEND_DIR

ASSETS = {
    "/chat": (FRONTEND_DIR / "index.html", "text/html; charset=utf-8"),
    "/assets/styles.css": (FRONTEND_DIR / "styles" / "main.css", "text/css; charset=utf-8"),
    "/assets/app.js": (FRONTEND_DIR / "src" / "app.js", "text/javascript; charset=utf-8"),
    "/assets/client-core.js": (
        FRONTEND_DIR / "src" / "client-core.js",
        "text/javascript; charset=utf-8",
    ),
}
UI_PATHS = frozenset(ASSETS)
MAX_ASSET_BYTES = 1_048_576
UI_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


def read_asset(request_path: str) -> tuple[bytes, str]:
    """Exact lookup with a size/encoding bound; no joins or URL decoding."""
    path, media_type = ASSETS[request_path]
    with path.open("rb") as stream:
        body = stream.read(MAX_ASSET_BYTES + 1)
    if len(body) > MAX_ASSET_BYTES:
        raise ValueError("Frontend asset exceeds the size limit.")
    body.decode("utf-8")
    return body, media_type
