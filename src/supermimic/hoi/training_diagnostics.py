"""Offline export of retained P2 measurements; no reward or optimizer dependency."""

import json

from supermimic.hoi.contact_validation import actual_contact, confusion
from supermimic.hoi.training_scene import CALIBRATION


def retained_contact_report(trainer):
    buffer = trainer.buffer
    mask = buffer.fields["transition_valid"] & buffer.valid
    raw = buffer.fields["force_raw_world"][mask].cpu().numpy()
    calibration = json.loads(CALIBRATION.read_text())
    measured = actual_contact(
        raw, calibration["thresholds_N_LR"], calibration["occupancy_cutoffs_LR"]
    )
    reference = buffer.fields["intent_ref"][mask, 1].cpu().numpy().astype(bool)
    return {
        "scope": "retained valid Replay intervals; diagnostic only, unchanged reached-frame labels",
        "rows": len(raw),
        "thresholds_N_LR": calibration["thresholds_N_LR"],
        "occupancy_cutoffs_LR": calibration["occupancy_cutoffs_LR"],
        "peak_net_force_N": float(measured["magnitude_N"].max()) if len(raw) else None,
        "any": confusion(reference, measured["any"]),
        "bilateral": confusion(reference, measured["bilateral"]),
        "overflow_count": int(trainer.env.raw.sim.data.overflow.count_nonzero()),
        "reference_training_ready": trainer.identity["reference_training_ready"],
    }
