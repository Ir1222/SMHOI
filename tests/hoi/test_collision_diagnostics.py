import pytest

from supermimic.hoi.contact_validation import penetration_summary


def test_manifold_multiplicity_does_not_multiply_duration():
    contacts = [
        {"sample": i, "dist": d, "category": c}
        for i, d, c in [
            (0, -0.002, "left"),
            (0, -0.004, "left"),
            (1, 0.001, "left"),
            (3, -0.0001, "support"),
        ]
    ]
    r = penetration_summary(contacts, 5)
    assert r["left"]["contact_records"] == 3 and r["left"]["contact_samples"] == 2
    assert r["hand_box"]["0.001"]["duration_s"] == 0.005
    assert r["hand_box"]["0.001"]["frequency_fraction"] == 0.2
    assert r["hand_box"]["max_penetration_m"] == 0.004
    assert r["hand_box"]["mean_depth_during_contact_m"] == pytest.approx(0.002)
    assert r["support"]["contact_records"] == 1 and r["right"]["contact_records"] == 0
    assert r["right"]["min_signed_distance_m"] is None


def test_persistent_penetration_and_gaps():
    cs = [{"sample": i, "dist": -0.002, "category": "right"} for i in list(range(20)) + [21]]
    r = penetration_summary(cs, 23)["right"]["0.001"]
    assert r["episodes"] == 2 and r["max_contiguous_s"] == 0.1 and r["duration_s"] == 0.105


def test_invalid_contact_sample_rejected():
    with pytest.raises(ValueError):
        penetration_summary([{"sample": 3, "dist": -0.1, "category": "left"}], 3)
