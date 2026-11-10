# P0 dependency and ContactSensor audit

Status: inspected installed implementation and ran isolated dependency probes. This is not P1 scene calibration. Sources below are local files under `.venv/lib/python3.12/site-packages/`; their SHA256 values are recorded in [dependencies.json](dependencies.json).

## Installed and locked environment

| Component | Installed | Lock / provenance |
|---|---|---|
| Python | 3.12.8 | target `.venv/bin/python`, Anaconda build |
| mjlab | 1.6.0 | `8bb2bc9e187797689ec82c78c35019afd2fd0376` in pyproject, uv.lock and installed direct_url.json |
| MuJoCo | 3.16.0 | matches uv.lock |
| MuJoCo Warp | 3.16.0 | matches uv.lock |
| Warp | 1.18.0 | matches uv.lock |
| Torch | 2.14.1+cu130 | matches uv.lock |
| NumPy | 2.5.4 | matches uv.lock |

The source carry checkout pins mjlab v1.4.0; its APIs/defaults cannot establish target behavior. No packages were installed or upgraded. The available GPU was an RTX 4090 D with roughly 24 GiB; the isolated probe used only small sensor buffers. No training was launched.

## Match expansion and actual collision bodies

`mjlab/sensor/contact_sensor.py:260` expands one native sensor per `(primary, field)`; body matching includes all geoms attached directly to that body. Subtree matching also includes descendants. Entity scope prefixes names as `entity/local_name`. Geom/body regexes resolve within the specified entity; without an entity, a pattern is treated as a literal name. Primary matches are separate output columns, not automatically summed across the hand group.

The compiled, selected native G1 has these enabled capsule colliders (enum type 3):

| Side | Geom | Actual parent body | Radius / capsule half length (m) | Local center (m) |
|---|---|---|---|---|
| Left | left_wrist_collision | left_wrist_pitch_link | .035 / .035 | (.025,0,0) |
| Left | left_hand_collision | left_wrist_yaw_link | .035 / .0412310563 | (.11,-.01,0) |
| Right | right_wrist_collision | right_wrist_pitch_link | .035 / .035 | (.025,0,0) |
| Right | right_hand_collision | right_wrist_yaw_link | .035 / .0412310563 | (.11,+.01,0) |

All have contype/conaffinity 1/1, condim 1, priority 0 and friction `(1,.005,.0001)` after current robot configuration. Full orientations and dimensions are in [audit_checks.json](audit_checks.json). Collision shapes are capsules, not visual meshes. The M6.5 sensor contract lists all four parent bodies. Source carry code's yaw-only sensor follows its different HDMI collider adaptation and would omit selected native wrist collisions here.

For P1, the audit-supported configuration is two independent sensors, each with that side's pitch and yaw primary bodies, secondary `body=bread_box` in the box entity, `secondary_policy="error"`, `reduce="netforce"`, `num_slots=1`, `history_length=4`, fields at least `found, force`. Exact entity names must be asserted against the constructed scene. Each side then has P=2 primary columns: force `[E,2,3]`, history `[E,2,4,3]`; adding both primary force vectors yields that side's net force. This is a file-change recommendation, not an implemented sensor.

The single secondary body includes bottom, side1–4 and both handles. Matching just one secondary geom loses the other six; expanding a secondary geom regex is not a union filter. `secondary_policy="first"` silently selects the first match; `"any"` drops the filter entirely, even when a secondary was supplied. `"error"` rejects ambiguous geom/body matches. The subtree secondary path does not perform the same body-name expansion; exact body matching is sufficient for the one rigid bread-box body.

## Native netforce semantics

`mujoco_warp/_src/sensor.py:2230–2403` matches each contact against both object orders and records direction. It filters `ContactType.CONSTRAINT` and indexes by `contact.worldid`. For a regular primary/secondary order, direction is +1; reversed order is −1.

`_sensor_contact`, netforce branch around lines 2507–2605, performs the following:

1. Extract solved constraint/contact force, including applicable friction components, with `support.contact_force_fn`.
2. Multiply by pair direction and rotate using the transpose of the native contact frame.
3. Sum vectors over matched contacts for each primary. This is world-frame net force in newtons, not impulse and not the sum of normal-force magnitudes.
4. Return raw matched-contact count in `found`. Torque is summed and shifted to a force-magnitude-weighted contact centroid. `dist` is set to zero; `normal` and `tangent` are identity-axis placeholders for netforce and cannot diagnose penetration or surface normals.

Matching is bounded by `contact_sensor_maxmatch` (mjlab default 64). `found` reports raw count, but the sum is capped at that limit, with overflow recorded. `num_slots>1` is not a way to obtain more netforce contacts: the first slot carries the result and remaining slots are zero in this Warp implementation. Use one slot explicitly. Net vectors can cancel, so nonzero `found`, net force above a threshold, and summed normal load are distinct measurements.

The target robot-only environment uses `nconmax=35`, `njmax=250`, and suppresses overflow printf via `QuietSimulationCfg`. P1 must size/check capacities and inspect overflow flags under its actual scene, not infer valid labels from quiet logs. This audit does not change defaults.

## Force sign and coordinates

The native netforce is **force exerted by primary on secondary**. With hand primary and box secondary it is force on the box; force on the hand is its negative. A generic sphere-plane probe gave `(0,0,-11.8195)` N for ball→floor and its exact opposite for floor→ball. MuJoCo CPU vs Warp sensor result differed by only `3.052e-8` N; Warp eager and captured sensor evaluation matched. See [force_sign_probe.json](force_sign_probe.json) and [force_probe.log](force_probe.log).

This probe reused an existing CPU constraint solution to test Warp `sensor_acc`, in an ephemeral fixture. It does not validate full Warp stepping, actual hand placement, all seven box contacts, transient collisions, or scene reset/isolation. Those remain P1 tests.

For box-local components, `f_box = R(q_box_wxyz)^T f_world`. Do not negate again merely because primary is a hand. The missing remainder of the master plan prevents confirming whether interval labels rotate each substep into its contemporaneous box frame, or rotate an interval-world average into a specified interval anchor frame. Existing history stores forces, not four object poses. That convention must be explicit before P2; using the post-step box pose silently would introduce a different target.

## Four-substep timing

`mjlab/envs/manager_based_rl_env.py:386–498`, `scene/scene.py:197–207`, and `mujoco_warp/_src/forward.py:2188–2239` establish this order:

```text
process action once
repeat 4 times:
    apply action -> write scene -> mjwarp.step
        forward dynamics -> solve constraints -> sensor_acc -> integrate
    scene.update(.005) -> ContactSensor.update -> roll/insert history
    substep metrics
termination -> reward -> metrics -> step/interval events
record_pre_reset(done IDs) -> reset done IDs (sensor histories zeroed)
forward all worlds (recomputes instantaneous sensordata)
advance commands -> camera/raycast sense -> observations
record_post_reset(done IDs) -> record_post_step(non-reset IDs)
```

Physics dt is .005 s, control dt .020 s. History is updated exactly once after each of the four substeps. The force corresponds to the solved force used in that substep; derived positions are the pre-integration quantities. qpos/qvel already hold the integrated end state. Four samples are stored newest-first, index 0 being substep 4. The final forward is not a fifth history sample.

For current, non-reset worlds, history survives the post-loop forward, whereas instantaneous `sensordata` is recomputed there. `Sensor.data` caches a concatenated result until update/reset invalidates it, so instantaneous `data.force` after forward can depend on whether it was first read before or after forward. History is the stable interval source. The history dictionary replaces each force buffer using `.roll` every update; consumers must reacquire it after the interval or clone for retained labels, not hold an old tensor pointer.

Terminal force histories must be copied during `record_pre_reset`, before scene reset clears selected rows. Nonterminal histories can be copied in `record_post_step`. Neither callback sees all substeps individually. Global manual resets also clear history; no extra initial forward sample should be counted as an action sample.

The built-in history includes force, torque and dist only. There is **no found history**. A transient can have `found=0` on the final substep while earlier forces survive in the four-sample history. Force-bearing transient detection can inspect per-substep force norms, with an explicit threshold; counting all geometric/constraint contacts through the interval needs separate bookkeeping. Netforce cancellation can hide individual contact load. Dist history under netforce is not a substitute.

An installed-class synthetic probe inserted 1,2,3,4 and read `[4,3,2,1]`; reset of env 0 zeroed its history and preserved env 1. This verifies bookkeeping, not physical parallel-world calibration. Evidence: [audit_checks.json](audit_checks.json).

## CUDA graph boundary

`mjlab/sim/sim.py:354–386` captures Warp step, forward, and reset independently. `scene.update` and PyTorch contact history updates execute outside those captured graphs once per substep. Contact sensors are calculated within forward/step; `sim.sense()` is for camera/raycast context, not the native contact pipeline. Replacing model/data arrays requires graph recapture; runtime edits must not leave captured pointers stale. No need to patch installed dependencies has been established. A full HOI step/reset graph test remains P1 work.

## Root, COM and quaternion conventions

Current robot state reads raw qpos/qvel and subtracts the environment origin from root position. MuJoCo free-joint translation velocity is root-origin linear velocity in world coordinates; angular qvel is body-local. `mjlab/entity/data.py:105–152` converts world angular velocity to local on root writes and implements `v_root = v_COM − omega_world × (R*com_offset)`. Root COM and link velocities are separate properties.

A nonidentity-rotation, offset-COM numeric probe verified qvel linear equals the finite-difference root translation and `v_COM − omega_world × (p_COM−p_root)`; world angular velocity equals `R*omega_body`. It also verified COM and root velocities differ. See [audit_checks.json](audit_checks.json). Quaternions are wxyz; current integration uses right multiplication `q_next = q * exp(dt*omega_body + correction_body)`. The future 13-D object layout must explicitly document its angular frame and preserve root/COM conversion; no HOI state implementation exists yet.
