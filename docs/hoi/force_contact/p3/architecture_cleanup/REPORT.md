# P3 architecture cleanup report — 2026-11-11

**Overall acceptance: PARTIAL. Architecture/software contracts: PASS.**

The final joint model uses input166/output82, physical84 and canonical86. The
active Contact Head, contact input/logit/probability feedback, Contact BCE and
classifier diagnostics are removed. Reference `object_contact[t]`, Replay labels,
P1/P2 sensing and historical evidence are preserved. No P4 work, physical data
collection, optimizer update or Joint WM training was performed. Learned predictive
validation remains **DEFERRED to P7-A**; this is not a trained-model acceptance.

Two checks prevent overall PASS: the external robot checkpoint still exceeds the
approved FP32 multi-step tolerance, and one unchanged P1 configuration test fails
because its historical V3 provenance document is absent. Neither criterion was
relaxed. The user explicitly authorized implementation and a commit with “yes,
implement, and commit after change”; this commit records the partial result and
does not authorize P4 or claim research readiness.

## Baseline and implementation

Branch: `main`. Starting HEAD and verified GitHub main:
`7251bf33f95fb4e2ee75f45a306f66fbadac85d1`. The complete current Master Plan and
historical P0/P1/P2/P3 reports were read before implementation. The Master Plan
SHA256 is `500a59dd17644284f012e1844c3f64e98287d7d84e0bda9f2ee0f4272733d75d`;
it is unchanged. The independent [specification](IMPLEMENTATION_SPEC.md) records
precedence, baseline, traceability, risks and implementation adjustment A1.

The shared original `DynamicsMLP`, autoregressive `WorldModel.window_loss` and
`WorldModelTrainer.update` are reused. No original robot WM, Trainer.run, Replay,
Actor/Critic, imagination, default configuration or dependency source was edited.
The joint adapter now passes physical targets and `a_exec` only. Termination remains
the original separate head/BCE, using physical failure and reference root at t+1;
it is not a contact classifier. An internal empty auxiliary mapping enables the
existing physical metric hook; callers cannot supply auxiliary WM targets.

| Contract | Final value / behavior |
|---|---|
| Physical state | robot71 + object13 = 84 |
| Canonical state | robot71 + object15 = 86 |
| Input order | robot0:71, object71:86, support86:104, action104:133, xi133:166 |
| Output order | robot70 + object12 = 82 physical residuals |
| Object residual | dv in heading frame, domega in body frame, dp in heading frame, dtheta in body frame; 3D each |
| Pose/twist | Environment-local root position, wxyz quaternion, root-origin world linear velocity, body angular velocity |
| Canonical linear velocity | Absolute object velocity rotated into robot heading; no robot-velocity subtraction |
| Integration | Existing velocity-first constant-velocity prior, body rotation increment on right; control dt .020 s |
| Support / xi | Support18 recomputed each step; xi33 unchanged and episode-constant |
| Measurement timing | Existing .005 s physics, decimation4, .020 s control; no timing change |
| Reference labels | Exact existing binary labels and t/t+1 Replay pairs retained outside WM loss |
| Force data | Existing N units, hand-on-box sign, contemporaneous box-frame substep aggregation retained as diagnostics/provenance |
| Final checkpoint | `hoi-joint-physical-wm-v3`, strict dimensions/config/timing/asset/reference/tensor validation; no optimizer resume |

Robot migration maps named robot/action/xi columns, zeros new object/support input
columns and object output rows, and preserves termination tensors. Strict legacy
force migration preserves all 82 physical rows, discards force rows82:88 and force
statistics, and preserves physical/termination coefficients. Transitional167/83
contact checkpoints and raw fit snapshots are rejected. No `strict=False` migration
is used. All source checkpoints remain intact.

The contact collector/fitter now exit before experiment imports or output creation.
Their original source and the removed contact-head tests remain available at the
baseline Git revision. The saved historical force-readiness interpreter is isolated
in `legacy_validation.py`; its AST is identical to the old implementation and its
saved conclusion is unchanged. Dormant force-statistics utilities remain available
for old evidence, outside the active WM. The physical metric helper is retained as
a P7-A interface; it was not run for learned predictive evaluation here.

## Verification and numeric evidence

Exact commands and exclusions are in [VALIDATION.md](VALIDATION.md). Evidence root:
`outputs/hoi/contact_supervision/architecture_cleanup/`.

| Check | Result | Evidence |
|---|---|---|
| First targeted run, before A1 | 12 passed / 15 failed; missing provenance documents | `tests_01.log` |
| Final core contracts | **28 passed** | `tests_02.log` |
| Additional P1/P2 + original robot/FK regression | **28 passed / 1 failed** | `regression_01.log` |
| Strict legacy force166/88 → physical166/82, horizons1–8 | **PASS**, max absolute difference 0 at every step | `migration_01/results.json` |
| Final strict checkpoint round-trip | **PASS**, all tensors exact; prediction difference 0 | same |
| External original robot checkpoint, FP32 | **PARTIAL**, horizon1 passes; horizons2–8 fail original tolerance | same |
| External robot FP64 diagnostic | Every horizon difference 0; not a replacement for FP32 acceptance | same |
| Original robot window loss/metrics/gradients vs accepted P2 source | **PASS**, exact equality | Included in core run, `test_wm_reuse.py` |
| Lint / format | **PASS**, 16 changed/new Python files | `static_checks.json` |
| Historical files and dependency preservation | **PASS**, no unexpected changes | `preservation.json`, `baseline.json` |

Final selected pytest runs total **56 passed / 1 failed**, with no pytest skips.
This is not a full-suite claim. The initial failed run is preserved separately;
its counts are not added to the final total. The existing TorchScript deprecation
warning appeared in both suites and the migration probe.

The external robot source is
`/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt`, SHA256
`16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7`.
With **atol=rtol=2e-6**, its measured FP32 maximum absolute error is:

| Horizon | Max absolute error | Within tolerance |
|---|---:|---|
| 1 | 3.725290298461914e-08 | PASS |
| 2 | 7.62939453125e-06 | FAIL |
| 3 | 1.33514404296875e-05 | FAIL |
| 4 | 1.33514404296875e-05 | FAIL |
| 5 | 1.9073486328125e-05 | FAIL |
| 6 | 1.621246337890625e-05 | FAIL |
| 7 | 1.621246337890625e-05 | FAIL |
| 8 | 2.288818359375e-05 | FAIL |

The per-element tolerance decision is recorded separately from the maximum error.
FP64 equality is consistent with accumulation differences after widening the input
matrix; it does not establish FP32 equivalence. The external source also lacks full
training Git/dataset provenance, so this is only an architecture compatibility
candidate. The local `logs/hdmi_robot_only/ckpt_0012000.pt` remains absent.

The strict historical force source is
`force_model_validation/migration_01/diagnostic_joint_weights.pt`, SHA256
`6246b001cb14a4966a90f6a99815b2fb45172b82f9cf57a867e6cdd0ce95d56b`.
The new initialization artifact is
`architecture_cleanup/migration_01/migrated_physical_weights.pt`, SHA256
`a226a40cf7a049fd67845fc473579b49d91a422ceaa05b7a11a205d847517360`.
It is 126,926 bytes, uses the final schema, records zero new updates and P7-A
deferred validation, and carries no inherited readiness claim. Source-to-source
rollout comparison uses existing data as inputs; it does not measure prediction
accuracy against physical next-state targets.

## Limitations and adjustment A1

The checkout lacks `SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md` and all three historical
refined-reference JSON sidecars. No exact copies were found locally or in Git; no
replacement documents were fabricated. The accepted M6.5 asset manifest and refined
NPZ bytes are present and hash-verified.

For SupportContext only, the exact accepted manifest is now pinned before structural
validation. Every available source must still match its recorded hash; the only
permitted missing source is the exact historical V3 planning filename/hash. A
modified manifest or any other missing/changed source is rejected. Geometry and
support encoding are unchanged. This narrowly documented provenance adaptation
enables software construction; it does not repair missing evidence or relax the
physical scene loader. The latter still fails
`test_exact_pair_filters_and_two_primary_columns` before any simulation stepping.
Restoring the exact original V3 document would address that prerequisite; a substitute
provenance policy would need its own review.

An explicit migration/software-only reference adapter verifies the historical
refined NPZ hash and array schema without claiming absent sidecar validation. The
normal `ReferenceClip` and `p3_common.reference()` retain full sidecar requirements.
The original accepted Part B hash `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`
and historical refined hash `f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7`
remain distinct. There was no data switch or relabeling.

The Master Plan still contains duplicate historical force-era P5–P8 sections and
residual Force Head wording. Current §§1/2/4/8/14/17 and the user's explicit design
decisions govern this cleanup; the Master Plan was not rewritten. M6.5 physical
hand/support FAILs, `training_ready=false` and `M6_5_NO_GO_M7` remain unresolved.
Failed contact capture remains FAIL with one complete transition, two saved partial
substeps and solver flag512 (ITERATIONS), not a buffer-saturation conclusion.
Historical P3 remains PARTIAL. No new readiness gate or acceptance threshold was
invented. P7-A still requires separate approval and its own prerequisite disposition.

## Preservation, diff review and commit scope

The [file inventory](CHANGED_FILES.md) lists every changed/new/deleted path. Of 534
baseline files, **519 are byte-identical** and exactly **15** are intentional source/
test modifications or removal. There are zero unexpected changes. This includes
262 historical documentation files, 127 existing output files, both asset manifests,
all six original robot WM files, all four original training files, all 11 environment
files and the user's untracked `.vscode/settings.json`. All ten audited dependency
source hashes and the external robot checkpoint hash are unchanged.

Commit scope is 30 paths: 15 existing source/test changes, two new Python files,
four independent documents and nine evidence artifacts. The diff of the 15 existing
paths contains 309 added and 1,357 removed lines; historical evidence is added only
under the new cleanup directory. Before staging, the only unrelated working-tree
entry was the pre-existing untracked `.vscode/`.

All pre-existing P0/P1/P2/P3 documentation, raw archives, snapshots and checkpoints
present at baseline are preserved, including the old P3 `REPORT.md` and
`IMPLEMENTATION_SPEC.md`, `revision_task0` evidence, failed contact collection and
both transitional migration checkpoints. No accepted data, reference, physical
asset, sensor, recorder or Replay implementation was edited.

The complete source/test diff, new documents, JSON evidence and raw test failures
were reviewed. Source/document whitespace checks pass. The two raw failed-pytest
logs retain 50 traceback lines with trailing spaces; these are evidence formatting,
not code defects, and are not rewritten. No unrelated files are staged. Commit
authorization is the user's latest explicit instruction; the commit does not mark
the partial checks as passed. The commit ID and final Git status are returned in
the accompanying response rather than embedded self-referentially in this report.

**Stop after this P3 commit and report for review. No push or next stage.**
