"""Inspect only the processed Part B NPZ with the reviewed static assets."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        default=Path('outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz'))
    parser.add_argument('--profile', choices=['refined-single-body', 'p6-original', 'p6-single-body'],
                        default='refined-single-body')
    parser.add_argument('--asset-manifest', type=Path,
                        default=Path('data/assets/hdmi_bread_box/asset_manifest_m6_5.json'))
    parser.add_argument('--display-only', action='store_true', required=True,
                        help='Acknowledge kinematic inspection; cannot approve physical/training gates')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--browser', action='store_true', help='Local EGL-rendered browser viewer')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 260, 280, 298, 400, 530, 549, 550, 839])
    parser.add_argument('--video', action='store_true', help='Export every frame at 50 Hz (headless)')
    parser.add_argument('--autoplay', action='store_true')
    parser.add_argument('--close-after-s', type=float, help='Bounded GUI smoke run')
    args = parser.parse_args()
    if args.browser and args.headless:
        parser.error('--browser and --headless are mutually exclusive')
    if args.browser:
        if args.autoplay or args.close_after_s is not None:
            parser.error('Browser playback is controlled in the page')
        os.environ.setdefault('MUJOCO_GL', 'egl')
    if args.headless:
        if args.output_dir is None:
            parser.error('--headless requires a new --output-dir')
        os.environ.setdefault('MUJOCO_GL', 'egl')
    elif args.video or args.output_dir:
        parser.error('--video/--output-dir require --headless')
    from supermimic.hoi.reference import ReferenceClip
    from supermimic.hoi.reference_viewer import ReferenceScene, render_headless, run_interactive

    scene = ReferenceScene(ReferenceClip(args.reference), args.asset_manifest,
                           display_only=args.display_only, profile=args.profile)
    print(scene.status_text(), flush=True)
    if args.browser:
        from supermimic.hoi.reference_browser import run_browser
        result = run_browser(scene, args.port)
    elif args.headless:
        result = render_headless(scene, args.output_dir, args.frames, video=args.video)
    else:
        result = run_interactive(scene, autoplay=args.autoplay, close_after_s=args.close_after_s)
    print(result)


if __name__ == '__main__':
    main()
