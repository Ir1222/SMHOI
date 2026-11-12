import sys

sys.path.insert(0, "outputs/hoi/contact_supervision/contact_validation/scripts")
import numpy as np
from calibrate_contacts import Placements
from common import cpu_contacts, physical_env, save
from scipy.optimize import least_squares

from supermimic.hoi.state import JointStateIO

E, _, _ = physical_env()
P = Placements(E, JointStateIO(E))
m = P.model
s = P.base.copy()
s[71:78] = [0.35, 0, 2.87, 1, 0, 0, 0]
fits = []
for side, sign in [("left", 1), ("right", -1)]:
    names = [
        n
        for n in P.io.joint_names
        if n.startswith(side + "_") and any(w in n for w in ["shoulder", "elbow", "wrist"])
    ]
    ids = [7 + P.io.joint_names.index(n) for n in names]
    lim = np.array([m.jnt_range[m.joint("robot/" + n).id] for n in names])
    init = s[ids].copy()
    g = m.geom("robot/" + side + "_hand_collision").id

    def f(q, ids=ids, g=g, sign=sign, init=init):
        s[ids] = q
        P.forward(s)
        a = P.data.geom_xmat[g].reshape(3, 3)[:, 2]
        target = [0.35, sign * (0.17 + 0.035 + m.geom_size[g, 1] * abs(a[1]) - 0.0005), 3.0]
        return np.r_[P.data.geom_xpos[g] - target, 0.00001 * (q - init)]

    sol = least_squares(
        f,
        init,
        bounds=(lim[:, 0] + 1e-5, lim[:, 1] - 1e-5),
        max_nfev=2000,
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )
    s[ids] = sol.x
    fits.append(f(sol.x).tolist())
P.forward(s)
c = cpu_contacts(m, P.data, 0)
print("FIT", fits, "contacts", [(a["category"], a["dist"], a["names"]) for a in c])
save(
    __import__("pathlib").Path("/tmp/p6_ik_probe.json"), {"state84": s, "contacts": c, "fits": fits}
)
E.close()
