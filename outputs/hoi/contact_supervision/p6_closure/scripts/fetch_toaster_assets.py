"""Read-only pinned upstream binary/mesh fetch; verify Git object hashes."""

import hashlib
import json
import urllib.request
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "upstream"
rows = json.loads((base / "toaster_asset_tree.json").read_text())
for row in rows:
    path = base / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(
        "https://raw.githubusercontent.com/shuayyy/humanoid-amp-ppo/"
        "b5087b7c8b3a123135b094b9cece3b2cefef26bc/" + row["path"],
        timeout=60,
    ).read()
    digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    assert digest == row["sha"], (path, digest)
    if path.exists():
        assert path.read_bytes() == data
    else:
        path.write_bytes(data)
    print(row["path"], len(data), digest, flush=True)
