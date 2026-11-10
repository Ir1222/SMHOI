"""Display-only MuJoCo reference inspection. Never advances physics."""
from __future__ import annotations

import json
import math
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.hoi.asset_manifest import apply_hand_profile, build_audit_scene, load_manifest
from supermimic.hoi.reference import DEFAULT_ASSET, ReferenceClip, sha256

# These are inspection bookmarks, not a motion schema or phase boundaries.
BOOKMARKS = {0: 'initial', 260: 'first desired contact', 280: 'sustained contact',
             298: 'lift inspection', 400: 'carry inspection', 530: 'placement inspection',
             549: 'last desired contact', 550: 'release intent', 839: 'final'}
SIDE_COLORS = ((0.1, 0.85, 0.95, 1), (0.95, 0.25, 0.65, 1))


def points_world(position, quaternion_wxyz, local_points):
    """One offline transform for both target and EEF markers; input arrays untouched."""
    return np.asarray(position) + Rotation.from_quat(
        quaternion_wxyz, scalar_first=True).apply(local_points)


@dataclass
class Playback:
    n_frames: int
    fps: float
    frame: int = 0
    paused: bool = True
    speed: float = 1.0
    remainder: float = 0.0

    def advance(self, elapsed_s: float):
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError('Elapsed time must be finite and nonnegative')
        if self.paused:
            return
        samples = self.remainder + elapsed_s * self.fps * self.speed
        steps = math.floor(samples + 1e-10)
        self.remainder = max(0., samples - steps)
        self.frame = min(self.n_frames - 1, self.frame + steps)
        if self.frame == self.n_frames - 1:
            self.paused, self.remainder = True, 0.

    def command(self, command: str):
        if command == 'toggle':
            self.paused = not self.paused
        elif command in ('next', 'previous'):
            self.frame = int(np.clip(self.frame + (1 if command == 'next' else -1),
                                     0, self.n_frames - 1))
            self.paused, self.remainder = True, 0.
        elif command == 'reset':
            self.frame, self.paused, self.remainder = 0, True, 0.
        elif command in ('faster', 'slower'):
            self.speed = float(np.clip(self.speed * (2 if command == 'faster' else .5), .125, 8))
        else:
            raise ValueError(command)


# GLFW key codes; callbacks only queue input, avoiding concurrent data mutations.
KEY_COMMANDS = {32: 'toggle', 262: 'next', 263: 'previous', 82: 'reset',
                259: 'reset', 61: 'faster', 334: 'faster', 93: 'faster',
                45: 'slower', 333: 'slower', 91: 'slower'}


class ReferenceScene:
    def __init__(self, clip: ReferenceClip, asset_path: Path = DEFAULT_ASSET,
                 *, display_only: bool = False):
        if not display_only:
            raise ValueError('Explicit display_only=True is required; this is not a training scene')
        self.clip = clip
        self.asset_path = Path(asset_path).resolve()
        if sha256(self.asset_path) != clip.manifest['sources']['asset']['sha256']:
            raise ValueError('Asset manifest differs from the processed reference contract')
        self.asset = load_manifest(self.asset_path)
        # Preserve all gate values, including training_ready=false.
        from mjlab.asset_zoo.robots import get_g1_robot_cfg
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

        if sha256(G1_XML) != self.asset['robot']['native_asset']['sha256']:
            raise ValueError('Installed G1 XML differs from reviewed native asset')
        cfg = get_g1_robot_cfg()
        spec = cfg.spec_fn()
        for collision in cfg.collisions:
            collision.edit_spec(spec)
        apply_hand_profile(spec, self.asset)
        scene = build_audit_scene(self.asset, clip.arrays['object_pos_w'][0],
                                  clip.arrays['object_quat_w'][0])
        spec.worldbody.add_geom(name='display_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, .02], rgba=[.22, .25, .29, 1],
                                contype=0, conaffinity=0, group=0)
        spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1],
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[.8, .8, .8])
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 800
        spec.visual.headlight.ambient = [.45, .45, .45]
        box = scene.body(self.asset['object']['name'])
        for i, side in enumerate(('left', 'right')):
            box.add_site(name=f'{side}_target_marker', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                         pos=self.asset['contact']['target_root_m'][i],
                         size=[.012, 0, 0], rgba=SIDE_COLORS[i], group=4)
        for g in self.asset['object']['geoms']:
            scene.geom(g['name']).rgba = [.76, .48, .19, .72]
        for i, support in enumerate(self.asset['supports']):
            scene.geom(support['geom']['name']).rgba = [.26 + .1*i, .42, .57, 1]
        spec.attach(scene, prefix="", suffix="", frame=spec.worldbody.add_frame())
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        if self.model.nu or self.model.neq or self.model.nmocap:
            raise ValueError('Inspection scene must have no actuators, constraints or mocap')
        self.root_adr = self.model.jnt_qposadr[self.model.joint('floating_base_joint').id]
        self.box_adr = self.model.jnt_qposadr[self.model.joint('bread_box_root').id]
        self.joint_ids = [self.model.joint(str(n)).id for n in clip.arrays['joint_names']]
        if len(set(self.joint_ids)) != 29:
            raise ValueError('Expected 29 unique mapped joints')
        if any(self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in self.joint_ids):
            raise ValueError('Expected named hinge joints')
        self.joint_adrs = self.model.jnt_qposadr[self.joint_ids]
        names = clip.arrays['body_names'].tolist()
        self.root_index = names.index('pelvis')
        self.body_ids = [self.model.body(n).id for n in names]
        self.wrist_indices = [names.index(n) for n in self.asset['contact']['wrist_bodies']]
        self.target_ids = [self.model.site(f'{s}_target_marker').id for s in ('left', 'right')]
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[[0, 2]] = 1  # asset geoms, native visual meshes
        self.option.sitegroup[:] = 0
        self.option.sitegroup[4] = 1
        self.frame = 0
        self.set_frame(0)

    def set_frame(self, index: int):
        f = self.clip.frame(index)
        self.frame = index
        d, m = self.data, self.model
        root = self.root_adr
        d.qpos[root:root+3] = f['body_pos_w'][self.root_index]
        d.qpos[root+3:root+7] = f['body_quat_w'][self.root_index]
        d.qpos[self.joint_adrs] = f['joint_pos']
        box = self.box_adr
        d.qpos[box:box+3] = f['object_pos_w']
        d.qpos[box+3:box+7] = f['object_quat_w']
        self.eef_reference = points_world(
            f['body_pos_w'][self.wrist_indices], f['body_quat_w'][self.wrist_indices],
            self.asset['contact']['eef_offset_wrist_m'])
        self.target_reference = points_world(f['object_pos_w'], f['object_quat_w'],
                                             self.asset['contact']['target_root_m'])
        # These buffers are display state, never a physical trajectory or velocity reference.
        d.qvel[:] = 0
        d.qacc_warmstart[:] = 0
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        d.time = index / self.clip.fps
        mujoco.mj_forward(m, d)
        self.distances = np.linalg.norm(d.site_xpos[self.target_ids] - self.eef_reference, axis=1)
        self.desired_contact = bool(f['object_contact'])

    def camera(self, closeup: bool = False, follow: bool = False):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        if closeup:
            camera.lookat[:] = self.clip.arrays['object_pos_w'][self.frame] + [0, 0, .12]
            camera.distance, camera.azimuth, camera.elevation = .95, 145, -28
        elif follow:
            camera.lookat[:] = self.data.xpos[self.model.body('pelvis').id] * [1, 1, 0] + [0, 0, .8]
            camera.distance, camera.azimuth, camera.elevation = 2.8, 135, -18
        else:
            camera.lookat[:] = [0, -.15, .75]
            camera.distance, camera.azimuth, camera.elevation = 5.7, 110, -20
        return camera

    def status_text(self, playback: Playback | None = None):
        motion = '' if playback is None else (
            f' | {"PAUSED" if playback.paused else "PLAYING"} | {playback.speed:g}x')
        return (f'KINEMATIC REFERENCE | training_ready={str(self.asset["training_ready"]).lower()}\n'
                f'Frame {self.frame}/839 | t={self.frame/self.clip.fps:.2f} s | 50 Hz{motion}\n'
                f'Desired contact: {"ON" if self.desired_contact else "OFF"} (intent only)\n'
                f'EEF-target L {self.distances[0]*1000:.2f} mm | R {self.distances[1]*1000:.2f} mm\n'
                'L cyan / R pink | large sphere: target | small sphere: EEF reference')

    def add_lines(self, scene):
        for i, color in enumerate(SIDE_COLORS):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                               np.zeros(3), np.eye(3).ravel(), np.asarray(color))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2,
                                self.data.site_xpos[self.target_ids[i]], self.eef_reference[i])
            scene.ngeom += 1
            marker = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([.008, 0, 0]), self.eef_reference[i],
                               np.eye(3).ravel(), np.asarray(color))
            scene.ngeom += 1


def run_interactive(scene: ReferenceScene, *, autoplay=False, close_after_s=None):
    """Passive GUI has no physics thread; reference samples are assigned on the main thread."""
    import mujoco.viewer

    events = queue.SimpleQueue()
    playback = Playback(scene.clip.n_frames, scene.clip.fps, paused=not autoplay)
    controls = 'Space play/pause | Left/Right frame | +/- speed | R reset | 1 overview 2 follow 3 closeup | C colliders'
    view_mode = 1
    with mujoco.viewer.launch_passive(scene.model, scene.data, key_callback=events.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[:] = scene.option.geomgroup
            viewer.opt.sitegroup[:] = scene.option.sitegroup
            cam = scene.camera()
            for key in ('lookat', 'distance', 'azimuth', 'elevation'):
                setattr(viewer.cam, key, getattr(cam, key))
        started = previous = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            playback.advance(now - previous)
            previous = now
            while not events.empty():
                key = events.get()
                if key in KEY_COMMANDS:
                    playback.command(KEY_COMMANDS[key])
                elif key in (49, 50, 51):
                    view_mode = key - 48
                    with viewer.lock():
                        cam = scene.camera(closeup=view_mode == 3, follow=view_mode == 2)
                        for attr in ('lookat', 'distance', 'azimuth', 'elevation'):
                            setattr(viewer.cam, attr, getattr(cam, attr))
                elif key == 67:
                    with viewer.lock():
                        viewer.opt.geomgroup[3] = 1 - viewer.opt.geomgroup[3]
                        viewer.opt.geomgroup[2] = 1 - viewer.opt.geomgroup[3]
            with viewer.lock():
                scene.set_frame(playback.frame)
                if view_mode != 1:
                    viewer.cam.lookat[:] = scene.camera(closeup=view_mode == 3, follow=view_mode == 2).lookat
                viewer.user_scn.ngeom = 0
                scene.add_lines(viewer.user_scn)
            viewer.set_texts((None, None, scene.status_text(playback) + '\n' + controls, ''))
            viewer.sync()
            if close_after_s is not None and now - started >= close_after_s:
                break
            time.sleep(.005)
    return {'final_frame': playback.frame, 'speed': playback.speed, 'paused': playback.paused}


def inspect_all_frames(scene: ReferenceScene):
    """Check the display FK/markers independently against every stored reference pose."""
    rows = []
    for i in range(scene.clip.n_frames):
        scene.set_frame(i)
        f = scene.clip.frame(i)
        pos_error = np.linalg.norm(scene.data.xpos[scene.body_ids] - f['body_pos_w'], axis=1)
        actual_rot = Rotation.from_quat(scene.data.xquat[scene.body_ids], scalar_first=True)
        reference_rot = Rotation.from_quat(f['body_quat_w'], scalar_first=True)
        orientation_error = (actual_rot.inv() * reference_rot).magnitude()
        obj_id = scene.model.body(scene.asset['object']['name']).id
        object_error = np.linalg.norm(scene.data.xpos[obj_id] - f['object_pos_w'])
        object_rotation_error = (
            Rotation.from_quat(scene.data.xquat[obj_id], scalar_first=True).inv()
            * Rotation.from_quat(f['object_quat_w'], scalar_first=True)).magnitude()
        target_error = np.max(np.linalg.norm(
            scene.data.site_xpos[scene.target_ids] - scene.target_reference, axis=1))
        wrist_ids = [scene.body_ids[j] for j in scene.wrist_indices]
        eef_fk = points_world(scene.data.xpos[wrist_ids], scene.data.xquat[wrist_ids],
                              scene.asset['contact']['eef_offset_wrist_m'])
        eef_error = np.max(np.linalg.norm(eef_fk - scene.eef_reference, axis=1))
        support_error = max(np.linalg.norm(scene.data.xpos[scene.model.body(s['name']).id]
                                          - s['root_pos_world']) for s in scene.asset['supports'])
        support_rotation_error = max((
            Rotation.from_quat(scene.data.xquat[scene.model.body(s['name']).id], scalar_first=True).inv()
            * Rotation.from_quat(s['root_quat_wxyz'], scalar_first=True)).magnitude()
            for s in scene.asset['supports'])
        rows.append({'frame': i, 'time_s': i/scene.clip.fps, 'desired_contact': scene.desired_contact,
                     'robot_position_max_error_m': float(pos_error.max()),
                     'robot_rotation_max_error_rad': float(orientation_error.max()),
                     'object_position_error_m': float(object_error),
                     'object_rotation_error_rad': float(object_rotation_error),
                     'target_transform_error_m': float(target_error),
                     'eef_fk_vs_reference_error_m': float(eef_error),
                     'support_position_error_m': float(support_error),
                     'support_rotation_error_rad': float(support_rotation_error),
                     'left_eef_target_m': float(scene.distances[0]),
                     'right_eef_target_m': float(scene.distances[1])})
    limits = {'robot_position_max_error_m': 2e-6, 'robot_rotation_max_error_rad': 2e-6,
              'object_position_error_m': 1e-12, 'object_rotation_error_rad': 1e-12, 'target_transform_error_m': 1e-7,
              'eef_fk_vs_reference_error_m': 2e-6, 'support_position_error_m': 1e-12, 'support_rotation_error_rad': 1e-12}
    maxima = {key: max(r[key] for r in rows) for key in limits}
    if any(maxima[key] > limit for key, limit in limits.items()):
        raise ValueError(f'Display transform validation failed: {maxima}')
    a = scene.clip.arrays
    def angular_steps(q):
        rotations = Rotation.from_quat(q, scalar_first=True)
        return (rotations[:-1].inv() * rotations[1:]).magnitude()
    joint_steps = np.abs(np.diff(a['joint_pos'].astype(float), axis=0))
    continuity = {
        'max_joint_step_rad': float(joint_steps.max()),
        'max_joint_step_destination_frame': int(np.unravel_index(joint_steps.argmax(), joint_steps.shape)[0]+1),
        'max_pelvis_step_m': float(np.linalg.norm(np.diff(a['body_pos_w'][:, scene.root_index], axis=0), axis=1).max()),
        'max_object_step_m': float(np.linalg.norm(np.diff(a['object_pos_w'], axis=0), axis=1).max()),
        'max_pelvis_rotation_step_rad': float(angular_steps(a['body_quat_w'][:, scene.root_index]).max()),
        'max_object_rotation_step_rad': float(angular_steps(a['object_quat_w']).max()),
        'note': 'Measurements of original adjacent samples, not smoothing or a physical-motion acceptance threshold.',
    }
    return rows, {'status': 'PASS', 'frames_checked': len(rows), 'maxima': maxima,
                  'tolerances': limits, 'continuity': continuity}


def render_headless(scene: ReferenceScene, output: Path, frames, *, video=False):
    """Use the same scene/pose method as the GUI; export auditable frames and optional 50 Hz video."""
    import csv
    import subprocess

    from PIL import Image, ImageDraw, ImageFont

    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite inspection output: {output}')
    for frame in frames:
        scene.clip._check_index(frame)
    rows, summary = inspect_all_frames(scene)
    output.mkdir(parents=True, exist_ok=False)
    summary.update({'reference_sha256': sha256(scene.clip.path),
                    'asset_manifest_sha256': sha256(scene.asset_path),
                    'mode': 'kinematic_display_only', 'training_ready': False,
                    'rendered_frames': list(frames), 'physics_steps': 0,
                    'bookmarks_are_phase_labels': False})
    (output / 'inspection.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'frames.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    thumbnails = []
    with mujoco.Renderer(scene.model, height=720, width=1200) as renderer:
        def render(closeup=False):
            renderer.update_scene(scene.data, camera=scene.camera(closeup=closeup),
                                  scene_option=scene.option)
            scene.add_lines(renderer.scene)
            im = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(im, 'RGBA')
            draw.rectangle((0, 0, 1200, 126), fill=(16, 22, 30, 230))
            draw.multiline_text((14, 8), scene.status_text(), font=font, fill='white', spacing=3)
            return im
        for frame in frames:
            scene.set_frame(frame)
            for closeup, name in [(False, 'overview'), (True, 'contact')]:
                im = render(closeup)
                im.save(output / f'frame_{frame:04d}_{name}.png')
                if not closeup:
                    thumbnails.append(im.resize((600, 360)))
        sheet = Image.new('RGB', (1800, math.ceil(len(thumbnails)/3)*360), (20, 25, 30))
        for i, im in enumerate(thumbnails):
            sheet.paste(im, ((i % 3)*600, (i // 3)*360))
        sheet.save(output / 'overview_montage.png')
        if video:
            command = ['ffmpeg', '-nostdin', '-n', '-loglevel', 'error', '-f', 'rawvideo',
                       '-pix_fmt', 'rgb24', '-s', '1200x720', '-r', str(scene.clip.fps),
                       '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22',
                       str(output / 'reference_50hz.mp4')]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for i in range(scene.clip.n_frames):
                    scene.set_frame(i)
                    process.stdin.write(np.asarray(render()).tobytes())
            finally:
                process.stdin.close()
                code = process.wait()
            if code:
                raise RuntimeError(f'ffmpeg failed: {code}')
    return summary
