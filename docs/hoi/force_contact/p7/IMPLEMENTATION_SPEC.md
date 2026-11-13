# P7.0 readiness audit scope

Authority: the current P7.0 user request at baseline
`a6b876f392f69709d9835eb0e4d664bfcfe75b8d`. The Master Plan remains unchanged.
Native PD reference replay and independent PD carrying remain P6 open problems;
their success is not a prerequisite for a learned-controller feasibility pilot.

This stage audits existing implementation and runs bounded software/physical
preflight checks. It does not implement or launch full P7-A/B, train a carrying
policy, promote assets, commit, or push. Backward-only checks have zero optimizer
updates. Synthetic gradient fixtures are identified separately from physical data.

| Requirement | Existing implementation/evidence | P7.0 work and acceptance |
|---|---|---|
| Plans and baseline | Master §12; P0 architecture/file map; P3 cleanup; P5 spec; P6 closure | Read current code; distinguish historical recommendations from implemented behavior. Preserve all old evidence. |
| Versioned experimental inputs | P6 `single_body.json`, `reference_candidate_v1`, original Part B | Pin hashes; retain `training_ready=false`; use original intent and baseline contact parameters. |
| Reset/state/action | P1 scene/state, native action affine map; P2 recorder | Two-world, at most 12-control-step Actor rollouts per reset segment; verify state84 and actual action use, zero external wrench, no mid-episode state writes. |
| Observation sufficiency | Actor930/goal680, Critic788; HOI state84 | Perturb robot and Box inputs independently. Report missing Box feedback without silently changing network dimensions. |
| Sensing/replay | P1/P2 filtered sensors and interval capture; P6 calibration | Reuse calibration; record four physical substeps, independently threshold L/R, verify reference t/t+1 and sample eight-step windows. |
| WM/loss/imagination | Joint166/82; P5 adapters | Shapes, finite backward, existing compiled parity, strict weight roundtrip. No predictive-readiness claim from synthetic or initialized WM. |
| Guard and invalid states | Hash-pinned opt-in guard; P1 capacity assertion | Check installed source hash and actual launched kernels during the Actor/HOI preflight. Check every substep for overflow and nonfinite state; stop immediately. Training-entry activation remains a separate gate if no HOI trainer exists. |
| Lifecycle | Robot-only Trainer save/load; joint weights-only checkpoint | Test reusable strict loading and report missing integrated resume/evaluation. |
| Pilot/protocol | Existing BPTT mainline; installed rsl_rl PPO | Prepare a bounded PPO proposal as requested, explicitly separate from Master §12 BPTT. No second optimizer loop is implemented here. Define A–F physical outcomes separately. |

New files are confined to this P7 documentation directory, a P7 readiness script,
and fresh evidence under `outputs/hoi/contact_supervision/p7_readiness/`.
Production source, dependency files, solver defaults, P6 artifacts, and references
are unchanged. The experimental scene reuses P6's candidate constructor, including
its explicit missing-V3 provenance exception; this is not production validation.

Predeclared checks: exact hashes/labels/action identity; reset roundtrip atol=1e-6;
finite state/action/loss/gradients; zero overflow flags at every solve; eight-step
windows stay within episode/clip; strict checkpoint loads and identical outputs.
No physical contact, lift, or carry pass is required from an untrained Actor.
Any observed invalid state ends that run, with failure artifacts retained.

The final findings, exact commands, limitations and pilot proposal belong in
`P7_TRAINING_READINESS.md`. A smoke pass does not authorize P7.1 execution.
