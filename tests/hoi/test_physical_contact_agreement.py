import numpy as np
import pytest

from supermimic.hoi.contact_validation import actual_contact, agreement, confusion


def test_strict_threshold_occupancy_and_net_cancellation():
    raw = np.zeros((4, 4, 2, 2, 3))
    raw[0, :, 0, 0, 0] = 1.0  # equal threshold is OFF
    raw[1, 0, 0, 0, 0] = 2.0  # one solve occupancy=.25 is OFF
    raw[2, :2, 0, 0, 0] = 2.0  # two solves ON
    raw[3, :, 0, 0, 0] = 2.0
    raw[3, :, 0, 1, 0] = -2.0  # opposing loads cancel
    x = actual_contact(raw, [1, 1], [0.25, 0.25])
    assert x["hands"][:, 0].tolist() == [False, False, True, False]
    assert not x["bilateral"].any()
    assert x["occupancy"][:, 0].tolist() == [0, 0.25, 0.5, 0]


def test_confusion_undefined_and_class_imbalance():
    x = confusion(np.array([0, 0, 1, 1], bool), np.array([0, 1, 0, 1], bool))
    assert x["matrix_ref_rows_sim_columns"] == [[1, 1], [1, 1]]
    assert all(x[k] == 0.5 for k in ["accuracy", "precision", "recall", "F1", "balanced_accuracy"])
    y = confusion(np.array([0, 0], bool), np.array([0, 0], bool))
    assert y["accuracy"] == 1 and y["precision"] is None and y["recall"] is None
    assert y["balanced_accuracy"] is None and y["F1"] is None


def test_alignment_shared_mask_and_episode_boundaries():
    raw = np.zeros((4, 4, 2, 2, 3))
    raw[1:, :, 0, 0, 0] = 2
    raw[2:, :, 1, 0, 0] = 2
    labels = np.array([0, 1, 0, 1, 0], bool)
    x = agreement(
        raw,
        labels,
        np.arange(4),
        np.arange(1, 5),
        np.array([0, 0, 1, 1]),
        np.array([1, 0, 1, 1], bool),
        [1, 1],
        [0.25, 0.25],
    )
    assert x["valid_rows"] == 3 and x["invalid_rows"] == [1]
    assert x["any"]["matrix_ref_rows_sim_columns"] == [[0, 1], [1, 1]]
    assert x["any"]["count"] == x["bilateral"]["count"] == 3
    assert x["left"]["onsets"] == 0  # reset boundary cannot invent onset
    assert x["left"]["OFF"]["contact_duration_s"] == 0.02
    assert x["left"]["ON"]["force_bearing_substep_duration_s"] == 0.02
    with pytest.raises(ValueError, match="Misaligned"):
        agreement(
            raw,
            labels,
            np.arange(4),
            np.arange(4),
            np.zeros(4),
            np.ones(4, bool),
            [1, 1],
            [0.25, 0.25],
        )


@pytest.mark.parametrize(
    "tau,rho", [([0, 1], [0.25, 0.25]), ([1, 1], [1, 0.25]), ([np.nan, 1], [0.25, 0.25])]
)
def test_invalid_definition_rejected(tau, rho):
    with pytest.raises(ValueError):
        actual_contact(np.zeros((1, 4, 2, 2, 3)), tau, rho)


def test_duplicate_reference_frames_in_same_episode_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        agreement(
            np.zeros((2, 4, 2, 2, 3)),
            np.zeros(4, bool),
            np.array([0, 0]),
            np.array([1, 1]),
            np.zeros(2),
            np.ones(2, bool),
            [1, 1],
            [0.25, 0.25],
        )
