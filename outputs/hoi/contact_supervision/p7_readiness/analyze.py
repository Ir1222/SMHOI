"""Offline smoke metrics and forward-only reset contact classification."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "outputs/hoi/contact_supervision/p6_closure/scripts"))
from closure_common import cpu_model, indices
from common import pair_category

from supermimic.hoi.contact_validation import actual_contact
from supermimic.hoi.reference import ReferenceClip
from supermimic.hoi.state import ReferenceResetCache

out = Path(__file__).parent
rows = [json.loads(x) for x in (out / "physical_final/transitions.jsonl").read_text().splitlines()]
calibration = json.loads(
    (ROOT / "outputs/hoi/contact_supervision/contact_validation/calibration.json").read_text()
)
for row in rows:
    contact = actual_contact(
        np.asarray(row["force_raw_world"]),
        calibration["thresholds_N_LR"],
        calibration["occupancy_cutoffs_LR"],
    )
    assert np.array_equal(contact["hands"], row["hands"])
    assert np.array_equal(contact["occupancy"], row["occupancy"])
a = np.asarray([x["action"] for x in rows])
s = np.asarray([x["state"] for x in rows])
sn = np.asarray([x["next_state"] for x in rows])
h = np.asarray([x["hands"] for x in rows])
f = np.asarray([x["frame"] for x in rows])
r = np.asarray([x["reward"]["total"] for x in rows])
intent = np.asarray([x["intent"] for x in rows], dtype=bool)[..., 1]
cont = np.all(f[1:] == f[:-1] + 1, axis=1)
summary = {
    "world_transitions": int(h.shape[0] * h.shape[1]),
    "action_abs_max": float(abs(a).max()),
    "action_within_episode_delta_rms": float(np.sqrt(np.mean(np.diff(a, axis=0)[cont] ** 2))),
    "robot_joint_step_delta_rms_rad": float(np.sqrt(np.mean((sn[..., 7:36] - s[..., 7:36]) ** 2))),
    "box_step_delta_max_m": float(np.linalg.norm(sn[..., 71:74] - s[..., 71:74], axis=-1).max()),
    "hand_contact_counts": h.sum((0, 1)).tolist(),
    "any_contact_count": int(h.any(-1).sum()),
    "bilateral_contact_count": int(h.all(-1).sum()),
    "intent_on_count": int(intent.sum()),
    "intent_off_count": int((~intent).sum()),
    "physical_reward_range": [float(r.min()), float(r.max())],
    "physical_reward_mean": float(r.mean()),
    "clip_end_transitions": int(np.asarray([x["clip_end"] for x in rows]).sum()),
    "optimized": False,
    "contact_recomputed_from_retained_raw_substeps": True,
}
model = cpu_model("single_body")
bq, bv, names, jq, jv = indices(model)
resets = []
for reference in (
    ROOT / "outputs/hoi/asset_audit/part_b/reference/reference.npz",
    ROOT / "outputs/hoi/contact_supervision/p6_closure/reference_candidate_v1/reference.npz",
):
    cache = ReferenceResetCache(ReferenceClip(reference), tuple(names), "cpu")
    for frame in [0, 258] if "candidate" not in str(reference) else [0, 258, 548, 827, 838]:
        state = cache.frames(torch.tensor([frame]))[0].numpy()
        data = mujoco.MjData(model)
        data.qpos[:7], data.qpos[jq], data.qpos[bq : bq + 7] = state[:7], state[7:36], state[71:78]
        data.qvel[:6], data.qvel[jv], data.qvel[bv : bv + 6] = (
            state[36:42],
            state[42:71],
            state[78:84],
        )
        mujoco.mj_forward(model, data)
        contacts = [
            {
                "distance_m": float(c.dist),
                "geom1": model.geom(c.geom1).name,
                "geom2": model.geom(c.geom2).name,
                "category": pair_category(model, c.geom1, c.geom2) or "non_box",
            }
            for c in data.contact
            if c.dist < 0
        ]
        resets.append(
            {
                "reference": str(reference.relative_to(ROOT)),
                "frame": frame,
                "negative_contacts": contacts,
            }
        )
summary["reset_overlap_scope"] = "CPU forward only; no stepping, no reset pose correction"
summary["reset_negative_contacts"] = resets
(out / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "reset_negative_contacts"}, indent=2))
