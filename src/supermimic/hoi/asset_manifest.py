"""Validated static asset construction for offline audit only.

This module does not load, transform or generate motion references. The audit
manifest currently fails physical-contact/support gates; it is not trainable.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from supermimic.hoi.provenance import effective_sources, research_authority
from supermimic.hoi.provenance import verify_sources as verify_source_records


def validate_manifest(m: dict, verify_sources: bool = True) -> None:
    if m['schema_version'] != 1 or m['units']['quaternion'] != 'wxyz':
        raise ValueError('Unsupported schema or quaternion order')
    obj = m['object']
    if obj['joint'] != 'free' or not obj['dynamic'] or len(obj['geoms']) != 7:
        raise ValueError('Expected one free rigid body and seven collision boxes')
    if len({g['name'] for g in obj['geoms']}) != 7:
        raise ValueError('Duplicate geom name')
    inertia = obj['inertia']
    if inertia['mass_kg'] <= 0 or np.any(np.asarray(inertia['diagonal_kg_m2']) <= 0):
        raise ValueError('Nonpositive mass/inertia')
    values = np.asarray(inertia['diagonal_kg_m2'])
    if 2 * values.max() > values.sum():
        raise ValueError('Invalid inertia triangle inequality')
    geoms = obj['geoms'] + [s['geom'] for s in m['supports']]
    for g in geoms:
        if g['type'] != 'box' or np.asarray(g['size']).shape != (3,) or np.any(np.asarray(g['size']) <= 0):
            raise ValueError('Invalid box half extents')
        for key, shape in [('pos', (3,)), ('quat', (4,)), ('friction', (3,))]:
            a = np.asarray(g[key])
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError('Invalid geom ' + key)
        if abs(np.linalg.norm(g['quat']) - 1) > 1e-6 or g['condim'] not in (1, 3, 4, 6):
            raise ValueError('Invalid quaternion/condim')
    if len(m['supports']) != 2 or not all(s['fixed'] for s in m['supports']):
        raise ValueError('Expected two fixed supports')
    for s in m['supports']:
        if np.asarray(s['root_pos_world']).shape != (3,) or abs(np.linalg.norm(s['root_quat_wxyz']) - 1) > 1e-6:
            raise ValueError('Invalid support pose')
    for key in ('target_root_m', 'eef_offset_wrist_m'):
        if np.asarray(m['contact'][key]).shape != (2, 3):
            raise ValueError('Invalid contact contract')
    if m['training_ready'] and any(g['status'] == 'FAIL' for g in m['gates'].values()):
        raise ValueError('Cannot approve training with failed geometry gates')
    if verify_sources:
        verify_source_records(m['sources'])
        if 'research_authority' in m and m['research_authority'] != research_authority():
            raise ValueError('Manifest research authority mismatch')


def load_manifest(path: str | Path, verify_sources: bool = True) -> dict:
    m = json.loads(Path(path).read_text())
    validate_manifest(m, verify_sources)
    # Return an explicit migrated view; never rewrite historical asset bytes/gates.
    m['sources'] = effective_sources(m['sources'])
    m['research_authority'] = research_authority()
    return m


def _add_geom(body, g):
    keys = ('pos', 'quat', 'size', 'friction', 'solref', 'solimp', 'contype',
            'conaffinity', 'condim', 'priority', 'group', 'margin', 'gap', 'solmix')
    return body.add_geom(name=g['name'], type=mujoco.mjtGeom.mjGEOM_BOX,
                         **{k: g[k] for k in keys if k in g})


def build_audit_scene(m: dict, object_pos, object_quat, origin=(0., 0., 0.)) -> mujoco.MjSpec:
    """Construct one free box and two fixed supports with a shared world offset."""
    validate_manifest(m, verify_sources=False)
    spec = mujoco.MjSpec()
    offset = np.asarray(origin, dtype=float)
    obj = m['object']
    b = spec.worldbody.add_body(name=obj['name'], pos=np.asarray(object_pos) + offset,
                               quat=object_quat)
    b.add_freejoint(name='bread_box_root')
    b.explicitinertial = True
    b.mass = obj['inertia']['mass_kg']
    b.ipos = obj['inertia']['com_pos_root_m']
    b.iquat = obj['inertia']['principal_axes_wxyz']
    b.inertia = obj['inertia']['diagonal_kg_m2']
    for g in obj['geoms']:
        _add_geom(b, g)
    for s in m['supports']:
        body = spec.worldbody.add_body(name=s['name'], pos=np.asarray(s['root_pos_world']) + offset,
                                      quat=s['root_quat_wxyz'])
        _add_geom(body, s['geom'])
    return spec


def apply_hand_profile(spec: mujoco.MjSpec, m: dict, profile: str | None = None) -> None:
    """Explicit in-memory audit option; never modifies a foot or source XML."""
    name = m['robot']['selected_profile'] if profile is None else profile
    cfg = m['robot']['profiles'][name]
    if name == 'native_unchanged':
        return
    for geom_name in cfg['disable_geoms']:
        g = spec.geom(geom_name)
        g.contype = g.conaffinity = 0
    for g in cfg['geoms']:
        _add_geom(spec.body(g['body']), g)
