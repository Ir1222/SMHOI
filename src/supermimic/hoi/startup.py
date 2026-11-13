"""Online startup state only; Trainer.run remains the sole training loop."""

from dataclasses import asdict, dataclass


@dataclass
class StartupProgress:
    phase: str = "collect"
    wm_updates: int = 0
    alternating_updates: int = 0
    waiting_since: int | None = None

    def ready(self, cfg, iteration, windows, starts):
        if cfg.execution == "integration":
            return False
        enough = windows >= cfg.startup.min_windows and starts >= cfg.startup.min_policy_starts
        if not enough:
            if self.waiting_since is None:
                self.waiting_since = 0 if self.phase == "collect" else iteration
            if iteration - self.waiting_since >= cfg.startup.max_collect_iterations:
                raise RuntimeError(
                    f"Online Replay not ready: windows8={windows}/{cfg.startup.min_windows}, "
                    f"policy_starts={starts}/{cfg.startup.min_policy_starts}, phase={self.phase}. "
                    "Inspect reset geometry, episode lengths and robot_failure/clip_end/timeout "
                    "counts in startup.json; no task-success/contact coverage gate is used."
                )
            return False
        self.waiting_since = None
        if iteration < cfg.schedule.warmup_iterations:
            return False
        if self.phase == "collect":
            self.phase = "wm_bootstrap"
        return True

    def completed(self, cfg):
        self.wm_updates += 1
        if self.phase == "alternating":
            self.alternating_updates += 1
        elif self.wm_updates >= cfg.startup.wm_only_updates:
            self.phase = "alternating"

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state, cfg, num_updates):
        if set(state) != set(cls().__dict__):
            raise ValueError("Incompatible startup state fields")
        progress = cls(**state)
        if progress.phase not in ("collect", "wm_bootstrap", "alternating"):
            raise ValueError("Invalid startup phase")
        if any(
            type(v) is not int or v < 0 for v in (progress.wm_updates, progress.alternating_updates)
        ):
            raise ValueError("Invalid startup update counters")
        if progress.waiting_since is not None and (
            type(progress.waiting_since) is not int or progress.waiting_since < 0
        ):
            raise ValueError("Invalid startup wait counter")
        if progress.wm_updates != num_updates:
            raise ValueError("Startup/Trainer update counts differ")
        expected_alternating = max(0, progress.wm_updates - cfg.startup.wm_only_updates)
        if progress.alternating_updates != expected_alternating:
            raise ValueError("Invalid alternating update count")
        if (progress.phase == "alternating") != (
            progress.wm_updates >= cfg.startup.wm_only_updates
        ):
            raise ValueError("Startup phase disagrees with completed WM updates")
        if progress.phase == "collect" and progress.wm_updates:
            raise ValueError("Collection phase cannot contain updates")
        return progress
