from __future__ import annotations


WEALTH_BUCKETS = ["constraint", "low", "middle", "high"]
INCOME_STATES = ["low", "high"]
AGG_STATES = ["recession", "normal", "boom"]


class LLMUnavailable(RuntimeError):
    pass
