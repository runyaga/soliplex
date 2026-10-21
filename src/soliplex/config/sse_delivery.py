from __future__ import annotations  # forward refs in typing decls

import dataclasses
import enum
import pathlib
import typing

# ============================================================================
#   AG-UI SSE delivery configuration types
# ============================================================================

_BOUND_KEYS = ("max_deltas", "max_bytes", "max_ms")


class InvalidSSEDeliveryConfig(ValueError):
    """Base class: an 'agui_sse_delivery' stanza is invalid"""

    def __init__(self, _config_path, reason: str):
        self._config_path = _config_path
        self.reason = reason

        super().__init__(
            f"Invalid 'agui_sse_delivery' in {_config_path}: {reason}",
        )


class SSEDeliveryNotAMapping(InvalidSSEDeliveryConfig):
    def __init__(self, _config_path):
        super().__init__(_config_path, "must be a mapping")


class SSEDeliveryMissingStrategy(InvalidSSEDeliveryConfig):
    def __init__(self, _config_path):
        super().__init__(_config_path, "'strategy' is required")


class SSEDeliveryUnknownKeys(InvalidSSEDeliveryConfig):
    def __init__(self, _config_path, unknown_keys):
        super().__init__(
            _config_path,
            f"unknown key(s): {sorted(unknown_keys)}",
        )


class SSEDeliveryUnknownStrategy(InvalidSSEDeliveryConfig):
    def __init__(self, _config_path, strategy):
        super().__init__(_config_path, f"unknown strategy: {strategy!r}")


class SSEDeliveryTakesNoBounds(InvalidSSEDeliveryConfig):
    def __init__(self, _config_path):
        super().__init__(_config_path, "strategy 'message' takes no bounds")


class SSEDeliveryMissingBound(InvalidSSEDeliveryConfig):
    def __init__(self, _config_path, key):
        super().__init__(
            _config_path,
            f"'{key}' is required for strategy 'bounded'",
        )


class SSEDeliveryInvalidBound(InvalidSSEDeliveryConfig):
    def __init__(self, _config_path, key):
        super().__init__(_config_path, f"'{key}' must be a positive integer")


class AGUI_SSEDeliveryStrategy(enum.StrEnum):
    MESSAGE = "message"
    BOUNDED = "bounded"


@dataclasses.dataclass(frozen=True, kw_only=True)
class AGUI_SSEDeliveryConfig:
    """How the AG-UI run endpoint groups streamed deltas into SSE events

    'message' applies 'soliplex.agui.sse_delivery.compact_event_stream',
    which merges a message's deltas until any other event arrives
    (normally the message's end).  'bounded' applies
    'soliplex.agui.sse_delivery.coalesce_event_stream', which also sends
    the merged deltas once they hold 'max_deltas' deltas or 'max_bytes'
    bytes, or have been held about 'max_ms' milliseconds.
    """

    strategy: AGUI_SSEDeliveryStrategy
    max_deltas: int | None = None
    max_bytes: int | None = None
    max_ms: int | None = None

    @classmethod
    def from_yaml(
        cls,
        config_path: pathlib.Path,
        config_dict: typing.Any,
    ) -> AGUI_SSEDeliveryConfig:
        if not isinstance(config_dict, dict):
            raise SSEDeliveryNotAMapping(config_path)

        if "strategy" not in config_dict:
            raise SSEDeliveryMissingStrategy(config_path)

        unknown = set(config_dict) - {"strategy", *_BOUND_KEYS}
        if unknown:
            raise SSEDeliveryUnknownKeys(config_path, sorted(unknown))

        try:
            strategy = AGUI_SSEDeliveryStrategy(config_dict["strategy"])
        except ValueError:
            raise SSEDeliveryUnknownStrategy(
                config_path,
                config_dict["strategy"],
            ) from None

        bounds = {
            key: config_dict[key] for key in _BOUND_KEYS if key in config_dict
        }

        if strategy == AGUI_SSEDeliveryStrategy.MESSAGE:
            if bounds:
                raise SSEDeliveryTakesNoBounds(config_path)
            return cls(strategy=strategy)

        for key in _BOUND_KEYS:
            if key not in bounds:
                raise SSEDeliveryMissingBound(config_path, key)
            value = bounds[key]
            if type(value) is not int or value < 1:
                raise SSEDeliveryInvalidBound(config_path, key)

        return cls(strategy=strategy, **bounds)

    @property
    def as_yaml(self) -> dict[str, typing.Any]:
        result = {"strategy": str(self.strategy)}
        if self.strategy == AGUI_SSEDeliveryStrategy.BOUNDED:
            result["max_deltas"] = self.max_deltas
            result["max_bytes"] = self.max_bytes
            result["max_ms"] = self.max_ms
        return result


DEFAULT_SSE_DELIVERY = AGUI_SSEDeliveryConfig(
    strategy=AGUI_SSEDeliveryStrategy.MESSAGE,
)
