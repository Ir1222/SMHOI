# Bounded training validation record — 2026-11-14

Execution authority is the sole Master Plan §1.1 and the current user request.
This is an experiment log, not an additional plan. Baseline c6225c1c0a66591dbb42e52760cc1119aad3414d.
User files `.vscode/` and the untracked local refined-reference directory are preserved.
No applicable AGENTS.md was found in the repository/ancestor directories.

Hypothesis 1: reset overlap comes from mapping/origin errors. Independent named
NPZ pose assignment on all 840 frames exactly matches cache qpos/contact distances;
native body FK differs from stored reference by at most 8.36e-8 m. Two-world
round trips checked at 0/406/460/839. Hypothesis rejected: fixed reference/native
collision geometry already overlaps. Keep reference/assets/reset unchanged.
Detailed source: logs/hoi_validation_20261114/reset_source_v1/reset_source.json.
This does not certify geometry or justify reference/asset changes without approval.

Implementation: add JSONL real-update/gradient, physical prediction/prior, and
checkpoint timing/size evidence around existing lifecycle hooks. Prediction
sampling preserves Torch/CUDA RNG and is explicitly in-sample, not held-out.
No algorithm, optimizer, architecture, reward or native simulation changes.

Every GPU-using command is run by scripts/hoi/validation_command.py. Its ledger
logs/hoi_validation_20261114/autoresearch.jsonl conservatively charges the entire
child lifetime, including CPU work, against 3600 seconds. Failures are retained.

Small v1 completed 64 vector steps / 128 transitions: 21 WM, 19 Actor, 380 Critic
Adam steps; internal WM-only startup transitioned at iteration 16. Midpoint32
resume completed iteration64 with equal counters and RNG, finite losses/gradients.
Final trajectories are NOT bitwise equal (qpos max delta 5.13e-5). A separate
exact-load probe compared all 573 tensors / 221113058 bytes with zero differences.
Repeating no-update continuation from that exact state already differs on step1
(qpos 6.85e-7, qvel 2.96e-5, identical clean/executed actions), isolating native
GPU physical nondeterminism. Retain the failed bitwise comparison; do not loosen
its criterion or claim deterministic trajectory resume. No state-loader fix is
supported by this evidence.

v2 correction: the shared console looked for the Robot-only `nse_1step` metric,
while HOI deliberately names it `weighted_component_mean_1step`. Report the
correct named metric and use n/a for unmeasured quantities instead of nan.
Only shared log formatting changed; Trainer.run update/control semantics remain.
Add CPU signed-geometry samples of actual pre-reset reached states at log times.
These are sparse samples, not complete substep penetration-duration measurements.

Medium v2 (16 env, batch64, H16, two WM minibatches, 10 startup updates) completed
96 iterations in 57.99 charged seconds: 60 WM, 20 Actor, 400 Critic Adam steps.
Source archives v1/v2 preserve exact implementation fingerprints for old evidence.

Scale64 solver10 failed again with final failure-context capture: last complete
iteration9, world63, reference frame454, episode age9, ITERATIONS512. Preserve both
failure reproductions. Hypothesis2: the solver iteration resource ceiling prevents
scaling. Raising only this ceiling to50 allows250 iterations with840 WM,34 Actor,
680 Critic steps. Keep the configurable ceiling and default50 for HOI; physical
scene calibration and Robot-only defaults remain10. Do not reinterpret this as a
penetration fix.

Final implementation commit6799072 reran64/batch256/H32 from scratch for250 steps,
normal save at268907781 bytes. Internal startup completed at164. Complete-state
restore compared573 tensors /268654120 tensor bytes with zero differences; repeated
native continuation still differs (step1 qpos1.19e-7, qvel9.54e-6). Strict real
resume continues from250 using the identical full config, including max_hours.


The same-config64 resume completed iteration350: cumulative WM1240 / Actor74 /
Critic1480 Adam steps. Exhaustive CPU Replay audit checked14036 eligible eight-step
windows and all16135 retained transitions. A separate start-frame0/830,0.24-second
boundary experiment observed four timeouts and four clip ends and checked their
distinct bootstrap semantics. The actual trained Actor play at frame0 failed at
step56. Frozen WM Robot eight-step normalized error improved versus zero residual
on that single episode (.1014 versus .6991), while Box position degraded (.01695m
versus .000983m). Do not promote WM/Actor/task success on this evidence.

Hypothesis3: the strong Critic loss/value drift is a TD(lambda) implementation
error. Frozen-snapshot independent per-trajectory double-precision recurrence
matches targets (max absolute difference6.84e-5, rtol/atol1e-5), with valid masks
and no optimizer updates. This does not explain the observed instability: Critic
loss reaches448.6768 at349; extreme imagined rewards/features are also present.
Do not alter reward, targets, algorithm or claim the cause is proven.

A separate interface bug is confirmed: inherited set_lr overwrote explicitly
configured Critic LR with Actor LR. An HOI-only ratio-preserving override fixes
this; equal initial rates preserve the existing arithmetic path and Robot-only
behavior is untouched. Three synthetic ratio tests plus four relevant regressions
passed. Initial rates in config/README/full script remain2e-4.

Automatic approval review rejected a combined change that would additionally
lower Critic default to2e-5. None of that rejected command executed. A concrete
single-parameter scratch comparison is documented in CRITIC_LR_PROPOSAL.md and
awaits explicit user approval; no lower-LR physical experiment has run.

Current-source v5 equal-rate engineering regression completed48 vector steps with
WM14 / Actor12 / Critic240 Adam steps, then strict24->48 resume and physical Actor
play. Immediate restore again compares all573 tensors bitwise. Play fails at65;
this small configuration is an interface regression, not comparable policy-quality
evidence against the64-world run. Source archives preserve the old fingerprints;
no loader contract or historic artifact was rewritten.
