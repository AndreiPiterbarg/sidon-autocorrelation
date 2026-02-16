"""Budget tracking and enforcement for RunPod sessions."""
import json
import time

from .config import (
    COST_PER_HOUR,
    BUDGET_LIMIT,
    BUDGET_WARN_PCT,
    SESSION_FILE,
)


class BudgetTracker:
    """Tracks session cost and enforces spending limits.

    Persists start time to SESSION_FILE so cost survives across CLI invocations.
    Cost = elapsed_hours * COST_PER_HOUR.
    """

    def __init__(self):
        self.start_time = None
        self._load()

    def _load(self):
        """Load session start time from file."""
        if SESSION_FILE.exists():
            try:
                data = json.loads(SESSION_FILE.read_text())
                self.start_time = data.get("start_time")
            except (json.JSONDecodeError, KeyError):
                self.start_time = None

    def start(self):
        """Record session start time."""
        self.start_time = time.time()

    def elapsed_hours(self):
        """Return elapsed time in hours since session start."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 3600.0

    def current_cost(self):
        """Return current session cost in USD."""
        return self.elapsed_hours() * COST_PER_HOUR

    def remaining_budget(self):
        """Return remaining budget in USD."""
        return max(0.0, BUDGET_LIMIT - self.current_cost())

    def max_remaining_seconds(self):
        """Return max seconds of compute remaining within budget."""
        remaining_usd = self.remaining_budget()
        return (remaining_usd / COST_PER_HOUR) * 3600.0

    def check(self):
        """Check budget status. Returns (ok, message).

        ok=True: can proceed (may include warning).
        ok=False: budget exceeded, must stop.
        """
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
        """Return a one-line budget status string."""
        _, msg = self.check()
        return msg
