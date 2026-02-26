"""Budget tracking and enforcement for CPU pod sessions."""
import json
import time

from .config import (
    COST_PER_HOUR,
    BUDGET_LIMIT,
    BUDGET_WARN_PCT,
    SESSION_FILE,
)


class BudgetTracker:
    """Tracks session cost and enforces spending limits."""

    def __init__(self):
        self.start_time = None
        self._load()

    def _load(self):
        if SESSION_FILE.exists():
            try:
                data = json.loads(SESSION_FILE.read_text())
                self.start_time = data.get("start_time")
            except (json.JSONDecodeError, KeyError):
                self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed_hours(self):
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 3600.0

    def current_cost(self):
        return self.elapsed_hours() * COST_PER_HOUR

    def remaining_budget(self):
        return max(0.0, BUDGET_LIMIT - self.current_cost())

    def max_remaining_seconds(self):
        remaining_usd = self.remaining_budget()
        return (remaining_usd / COST_PER_HOUR) * 3600.0

    def check(self):
        cost = self.current_cost()
        elapsed = self.elapsed_hours()

        if cost >= BUDGET_LIMIT:
            return False, (
                f"BUDGET EXCEEDED: ${cost:.2f} / ${BUDGET_LIMIT:.2f} "
                f"({elapsed:.1f}h). Tear down the pod NOW."
            )

        if cost >= BUDGET_LIMIT * BUDGET_WARN_PCT:
            return True, (
                f"WARNING: ${cost:.2f} / ${BUDGET_LIMIT:.2f} "
                f"({elapsed:.1f}h, ${self.remaining_budget():.2f} remaining). "
                f"Consider tearing down soon."
            )

        return True, (
            f"Budget: ${cost:.2f} / ${BUDGET_LIMIT:.2f} "
            f"({elapsed:.1f}h, ${self.remaining_budget():.2f} remaining)"
        )

    def status_line(self):
        _, msg = self.check()
        return msg
