import torch

from supermimic.hoi.support_context import SupportContext
from supermimic.kinematics.rotations import quat_apply, quat_from_yaw, quat_mul


def test_support_context_is_separate_time_varying_and_frame_covariant():
    support = SupportContext().double()
    s = torch.zeros(3, 84, dtype=torch.float64)
    s[:, 3] = s[:, 74] = 1
    base = support(s)
    assert base.shape == (3, 18)
    translated = s.clone()
    translated[:, 0] = 1
    assert not torch.equal(base, support(translated))
    yaw = quat_from_yaw(torch.tensor(0.7, dtype=s.dtype))
    shift = torch.tensor([1.0, -2.0, 0.0], dtype=s.dtype)
    for p, q in ((0, 3), (71, 74)):
        s[:, p : p + 3] = quat_apply(yaw.expand(3, 4), s[:, p : p + 3]) + shift
        s[:, q : q + 4] = quat_mul(yaw.expand(3, 4), s[:, q : q + 4])
    support.positions.copy_(quat_apply(yaw.expand(2, 4), support.positions) + shift)
    support.quaternions.copy_(quat_mul(yaw.expand(2, 4), support.quaternions))
    torch.testing.assert_close(support(s), base, atol=1e-9, rtol=0)
    s.requires_grad_()
    support(s).sum().backward()
    assert torch.isfinite(s.grad).all() and s.grad[:, :3].abs().sum() > 0
