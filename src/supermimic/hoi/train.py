"""smHOI-track: the approved HOI components in the original SuperMimic lifecycle."""

import json
import traceback

import torch
import tyro

from supermimic.hoi.training import HOITrainer
from supermimic.hoi.training_config import HOITrainConfig
from supermimic.hoi.training_diagnostics import retained_contact_report
from supermimic.hoi.training_safety import SolverGuard


def execute(cfg):
    """Bounded integration by default. No P7-A/B permission is inferred."""
    with SolverGuard() as guard:
        trainer = HOITrainer(cfg, guard)
        try:
            trainer.run()
            (trainer.log_dir / "physical_diagnostics.json").write_text(
                json.dumps(retained_contact_report(trainer), indent=2, allow_nan=False) + "\n"
            )
        except BaseException:
            # Preserve the actual failed state; never save it as resumable training.
            with (trainer.log_dir / "failure_state.pt").open("xb") as stream:
                torch.save(
                    {
                        k: getattr(trainer.env.raw.sim.data, k).detach().cpu().clone()
                        for k in ("qpos", "qvel", "qacc", "ctrl", "overflow")
                    },
                    stream,
                )
            (trainer.log_dir / "failure.txt").write_text(traceback.format_exc())
            raise
        finally:
            (trainer.log_dir / "guard.json").write_text(
                json.dumps(guard.evidence(), indent=2) + "\n"
            )
            (trainer.log_dir / "updates.json").write_text(
                json.dumps(trainer.learning_safety.steps) + "\n"
            )
            trainer.close()


def main():
    execute(tyro.cli(HOITrainConfig))


if __name__ == "__main__":
    main()
