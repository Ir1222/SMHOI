"""Raw contact contracts; physical scenario evidence is produced by the calibration CLI."""

import pytest
import torch
from mjlab.scene import Scene

from supermimic.hoi.contact import HAND_BODIES, SENSOR_NAMES, force_box_local
from supermimic.hoi.scene import PhysicalSceneCfg, make_physical_env_cfg
from supermimic.kinematics.rotations import quat_apply, quat_from_rotvec


def test_exact_pair_filters_and_two_primary_columns():
    cfg = make_physical_env_cfg(PhysicalSceneCfg(num_envs=1))
    scene = Scene(cfg.scene, "cpu")
    model = scene.compile()
    box_id = model.body("box/bread_box").id
    for name, (side, bodies) in zip(SENSOR_NAMES, HAND_BODIES.items()):
        sensor = scene[name]
        assert set(sensor.primary_names) == set(bodies)
        assert sensor.cfg.history_length == 4
        assert sensor.cfg.num_slots == 1 and sensor.cfg.reduce == "netforce"
        assert sensor.cfg.secondary_policy == "error"
        for body in bodies:
            sid = model.sensor(f"{name}_{body}_force").id
            assert model.sensor_objid[sid] == model.body("robot/" + body).id
            assert model.sensor_refid[sid] == box_id
    # A regex resolving to seven geoms must fail rather than silently choose one.
    cfg.scene.sensors[0].secondary.mode = "geom"
    cfg.scene.sensors[0].secondary.pattern = ".*"
    with pytest.raises(ValueError, match="multiple|matched|resolved"):
        Scene(cfg.scene, "cpu")


def test_box_force_rotation_broadcast_and_sign():
    quat = quat_from_rotvec(torch.tensor([[0.4, -0.3, 0.7], [0.2, 0.5, -0.6]]))
    world = torch.tensor(
        [[[5.0, -2.0, 1.0], [-3.0, 1.0, 4.0]], [[2.0, 3.0, 4.0], [0.0, -1.0, 5.0]]]
    )
    local = force_box_local(world, quat[:, None, :])
    torch.testing.assert_close(
        quat_apply(quat[:, None].expand(2, 2, 4), local), world, atol=2e-5, rtol=2e-6
    )
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(force_box_local(world, identity), world)
