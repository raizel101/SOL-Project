"""Bounded, user-displayable engine errors."""

from __future__ import annotations


class EngineError(ValueError):
    """Validation or local-service error that a caller can display safely."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}
