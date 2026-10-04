"""Typed episode-local lifecycle signals for Habitat DAgger."""


class DaggerEpisodeAbort(RuntimeError):
    """End one DAgger episode without turning it into a worker failure."""

    def __init__(self, reason: str) -> None:
        normalized = str(reason or "unknown").strip() or "unknown"
        self.reason = normalized
        super().__init__(normalized)


__all__ = ["DaggerEpisodeAbort"]
