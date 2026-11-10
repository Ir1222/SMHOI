"""Exercise actual HTTP/EGL rendering and browser controls, without a desktop GLX server."""
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_browser_controls():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to test browser JavaScript controls')
    subprocess.run([node, 'tests/hoi_browser_controls.mjs'], cwd=ROOT, check=True)


def test_browser_http_rendering(tmp_path):
    # Port 0 asks the OS for an available local port. The test reads the announced URL.
    log_path = tmp_path / 'browser.txt'
    with log_path.open('w') as log:
        proc = subprocess.Popen([sys.executable, '-u', 'scripts/hoi/replay_reference.py',
                                 '--display-only', '--browser', '--port', '0'], cwd=ROOT,
                                stdout=log, stderr=log, env={**os.environ, 'MUJOCO_GL': 'egl'})
        try:
            address = None
            for _ in range(200):
                content = log_path.read_text()
                if 'Local reference viewer: ' in content:
                    address = content.split('Local reference viewer: ')[-1].splitlines()[0]
                    break
                if proc.poll() is not None:
                    pytest.fail(content)
                time.sleep(.05)
            assert address is not None, log_path.read_text()
            with urlopen(address, timeout=5) as response:
                assert b'id="toggle"' in response.read()
            with urlopen(address + '/player.mjs', timeout=5) as response:
                assert b'bindControls' in response.read()
            for frame, view, contact in [(0, 'overview', False), (260, 'contact', True),
                                         (400, 'follow', True), (550, 'contact', False),
                                         (839, 'overview', False)]:
                with urlopen(f'{address}/frame?index={frame}&view={view}&colliders=1', timeout=10) as response:
                    result = json.load(response)
                assert result['frame'] == frame and result['time_s'] == frame/50
                assert result['desired_contact'] == contact and not result['training_ready']
                im = Image.open(io.BytesIO(base64.b64decode(result['image'])))
                assert im.size == (1200, 720)
                assert len(np.unique(np.asarray(im.resize((20, 12))).reshape(-1, 3), axis=0)) > 10
            for path in ['/frame?index=-1', '/frame?index=840', '/frame?index=1&view=bad']:
                with pytest.raises(HTTPError) as error:
                    urlopen(address + path, timeout=5)
                assert error.value.code == 400
        except URLError:
            pytest.fail(log_path.read_text())
        finally:
            proc.terminate()
            proc.wait(timeout=10)
