"""Fail-fast safety in the actual HOI process, including captured solver graphs."""

from collections import Counter
from contextlib import ExitStack
from functools import wraps

import torch

from supermimic.hoi.contact import assert_contact_capacity
from supermimic.hoi.solver_precision import bracket_precision_guard


def finite(name, *values):
    for value in values:
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite {name}; HOI execution aborted")
        elif isinstance(value, dict):
            finite(name, *value.values())
        elif isinstance(value, (tuple, list)):
            finite(name, *value)
        elif isinstance(value, float):
            import math

            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite {name}; HOI execution aborted")


class SolverGuard:
    def __init__(self):
        self.launches = Counter()
        self.graph_launches = 0
        self.substeps = 0
        self.checked_world_substeps = 0
        self.solver_iteration_histogram = Counter()
        self.attestations = []
        self.active = False

    def __enter__(self):
        import warp as wp
        from mujoco_warp._src import solver

        self.stack = ExitStack()
        self.identity = self.stack.enter_context(bracket_precision_guard())
        self.factory = solver._linesearch_iterative_kernel
        for name in ("launch", "launch_tiled", "capture_launch"):
            original = getattr(wp, name)

            def observed(*args, _original=original, _name=name, **kwargs):
                if _name == "capture_launch":
                    self.graph_launches += 1
                else:
                    kernel = kwargs.get("kernel", args[0] if args else None)
                    key = str(kernel.key)
                    if "linesearch" in key:
                        self.launches[key] += 1
                return _original(*args, **kwargs)

            setattr(wp, name, observed)
            self.stack.callback(setattr, wp, name, original)
        self.active = True
        return self

    def __exit__(self, *exc):
        self.active = False
        return self.stack.__exit__(*exc)

    def attest(self, env, reason):
        from mujoco_warp import OverflowType
        from mujoco_warp._src import solver

        if not self.active or solver._linesearch_iterative_kernel is not self.factory:
            raise RuntimeError("HOI solver guard no longer active")
        if not any("smhoi_p6_linesearch_precision" in k for k in self.launches):
            raise RuntimeError("Guarded line-search kernel was not observed during capture")
        if any(k.startswith("_linesearch_iterative_kernel__") for k in self.launches):
            raise RuntimeError("Unguarded iterative line-search kernel observed")
        if env.sim.step_graph is None or int(env.sim.wp_model.opt.warn_overflow) != int(
            OverflowType.ALL
        ):
            raise RuntimeError("HOI requires captured steps and visible overflow warnings")
        self.attestations.append(
            {
                "reason": reason,
                "guarded_kernel_launches": dict(self.launches),
                "graph_launches": self.graph_launches,
            }
        )

    def monitor(self, env):
        from mujoco_warp._src import solver

        if not self.active or solver._linesearch_iterative_kernel is not self.factory:
            raise RuntimeError("HOI solver guard lost during execution")
        assert_contact_capacity(env)  # Checks every world's complete overflow bitmask.
        finite(
            "physical solve",
            *(
                getattr(env.sim.data, k)[:]  # mjlab TorchArray proxy -> actual Tensor view.
                for k in (
                    "qpos",
                    "qvel",
                    "qacc",
                    "qacc_warmstart",
                    "ctrl",
                    "actuator_force",
                    "xfrc_applied",
                    "qfrc_applied",
                )
            ),
        )
        if env.sim.data.xfrc_applied.count_nonzero() or env.sim.data.qfrc_applied.count_nonzero():
            raise RuntimeError("External body/generalized wrench in HOI training profile")
        action = env.action_manager.get_term("joint_pos")
        finite("executed action", action.executed_raw, action._processed_actions)
        if not torch.equal(
            action._processed_actions, action.executed_raw * action._scale + action._offset
        ):
            raise RuntimeError("Executed action differs from the native actuator map")
        self.solver_iteration_histogram.update(env.sim.data.solver_niter[:].cpu().tolist())
        self.substeps += 1
        self.checked_world_substeps += env.num_envs
        return torch.zeros(env.num_envs, device=env.device)

    def evidence(self):
        return {
            **self.identity,
            "active": self.active,
            "kernel_launches": dict(self.launches),
            "graph_launches": self.graph_launches,
            "substeps": self.substeps,
            "checked_world_substeps": self.checked_world_substeps,
            "solver_iteration_histogram": dict(self.solver_iteration_histogram),
            "attestations": self.attestations,
        }


class LearningSafety:
    """Hooks on the original optimizers; no optimizer or training algorithm fork."""

    def __init__(self, wm_trainer, policy_trainer):
        wm_trainer.numerics_check = finite
        policy_trainer.numerics_check = finite
        self.steps = {"wm": 0, "actor": 0, "critic": 0}
        self.gradient_norms = {}
        self.handles = []
        for label, module, opt in (
            ("wm", wm_trainer.wm, wm_trainer.opt),
            ("actor", policy_trainer.policy, policy_trainer.opt),
            ("critic", policy_trainer.critic, policy_trainer.critic_opt),
        ):
            for parameter in module.parameters():

                def check_gradient(gradient, name=label):
                    finite(name + " gradient before clipping", gradient)
                    return gradient

                self.handles.append(parameter.register_hook(check_gradient))

            def before(optimizer, args, kwargs, name=label, model=module):
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients:
                    raise RuntimeError(f"No gradients for {name} optimizer")
                self.gradient_norms[name] = float(torch.stack([g.norm() for g in gradients]).norm())
                finite(
                    name + " weights/gradients",
                    model.state_dict(),
                    [p.grad for p in model.parameters() if p.grad is not None],
                    list(optimizer.state.values()),
                )

            def after(optimizer, args, kwargs, name=label, model=module):
                finite(
                    name + " updated weights/optimizer",
                    model.state_dict(),
                    list(optimizer.state.values()),
                )
                self.steps[name] += 1

            self.handles.extend(
                (opt.register_step_pre_hook(before), opt.register_step_post_hook(after))
            )
        for instance, name in (
            (wm_trainer, "loss_on"),
            (policy_trainer, "actor_loss"),
            (policy_trainer, "critic_batch"),
        ):
            original = getattr(instance, name)

            @wraps(original)
            def checked(*args, _original=original, _name=name, **kwargs):
                result = _original(*args, **kwargs)
                finite(_name, result)
                return result

            setattr(instance, name, checked)
