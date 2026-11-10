"""Local browser controls for the same EGL-rendered, display-only reference scene."""
from __future__ import annotations

import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mujoco
from PIL import Image

from supermimic.hoi.reference_viewer import ReferenceScene


class BrowserServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, scene: ReferenceScene, port=8765):
        self.scene = scene
        self.renderer = mujoco.Renderer(scene.model, height=720, width=1200)
        super().__init__(('127.0.0.1', port), Handler)

    def server_close(self):
        self.renderer.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/player.mjs'):
            name = 'index.html' if parsed.path == '/' else 'player.mjs'
            body = (Path(__file__).parent / 'viewer' / name).read_bytes()
            kind = 'text/html' if name.endswith('.html') else 'text/javascript'
        elif parsed.path == '/frame':
            try:
                query = parse_qs(parsed.query)
                frame = int(query['index'][0])
                view = query.get('view', ['overview'])[0]
                if view not in ('overview', 'follow', 'contact'):
                    raise ValueError('Unknown camera view')
                collision = query.get('colliders', ['0'])[0] == '1'
                scene = self.server.scene
                scene.set_frame(frame)
                scene.option.geomgroup[3] = int(collision)
                scene.option.geomgroup[2] = int(not collision)
                renderer = self.server.renderer
                renderer.update_scene(scene.data, camera=scene.camera(
                    closeup=view == 'contact', follow=view == 'follow'), scene_option=scene.option)
                scene.add_lines(renderer.scene)
                image = Image.fromarray(renderer.render())
                encoded = io.BytesIO()
                image.save(encoded, format='JPEG', quality=90)
                body = json.dumps({'frame': frame, 'time_s': frame / scene.clip.fps,
                                   'desired_contact': scene.desired_contact,
                                   'eef_target_m': scene.distances.tolist(),
                                   'training_ready': scene.asset['training_ready'],
                                   'status': scene.status_text(),
                                   'image': base64.b64encode(encoded.getvalue()).decode()}).encode()
                kind = 'application/json'
            except (ValueError, IndexError, KeyError) as error:
                self.send_error(400, str(error))
                return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)


def run_browser(scene: ReferenceScene, port=8765):
    # Single request thread owns the EGL context; all requests use the same pose/forward method.
    with BrowserServer(scene, port) as server:
        print(f'Local reference viewer: http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
