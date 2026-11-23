#!/usr/bin/env python3
"""Drive out and reverse over the same ground using LocalControl and the obstacle grid."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from scene_pipeline.contracts import write_json
from scene_pipeline.rhizome_nav import obstacle

REPO = Path(__file__).resolve().parents[1]


def score(log_path, distance, arrival, shuttle=None):
    directions = {key: dict(stop_ticks=0, grid_stop_ticks=0, violation_cell_ticks=0,
                             max_violation_cells=0, first_stop_s=None, first_violation_s=None,
                             first_violation_distance_m=None,
                             hole_cell_ticks=None, obstacle_cell_ticks=None)
                  for key in ('fwd', 'rev')}
    result = dict(step_ticks=0, nav_ticks=0, scored_predictions=0, obstacles=directions,
                  reference_time_s=None, peak_abs_z_error_m=None, peak_z_error_time_s=None,
                  turnaround_time_s=None, turnaround_z_error_m=None, final_z_error_m=None,
                  first_violation_s=None, first_violation_distance_m=None, travel_distance_m=0.,
                  legs=[], last_step_time_s=None, last_leg=None, max_distance_from_start_m=None,
                  final_distance_from_start_m=None, out_and_back_completed=False)
    reference = None
    previous_xy = None
    if not log_path.is_file():
        return result
    with log_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            t = row['simulation_time']
            value = row['output']
            if row['op'] == 'nav_infer':
                result['nav_ticks'] += 1
                prediction = value['prediction']
                if not prediction:
                    continue
                result['scored_predictions'] += 1
                for i, (direction, counts) in enumerate(directions.items()):
                    verdict = obstacle(prediction, direction)
                    violations = verdict['violations']
                    if violations and counts['first_violation_s'] is None:
                        counts.update(first_violation_s=t, first_violation_distance_m=result['travel_distance_m'])
                        if result['first_violation_s'] is None:
                            result.update(first_violation_s=t, first_violation_distance_m=result['travel_distance_m'])
                    counts['violation_cell_ticks'] += violations
                    counts['max_violation_cells'] = max(counts['max_violation_cells'], violations)
                    counts['grid_stop_ticks'] += int(verdict['grid_stop'])
                    if 'violation_cells' in value:
                        for kind in ('hole', 'obstacle'):
                            field = kind + '_cell_ticks'
                            counts[field] = (counts[field] or 0) + value['violation_cells'][kind][i]
                    if verdict['blocked']:
                        counts['stop_ticks'] += 1
                        if counts['first_stop_s'] is None:
                            counts['first_stop_s'] = t
                continue
            if row['op'] != 'step':
                continue
            loc = value['localization']
            result.update(step_ticks=result['step_ticks'] + 1, last_step_time_s=t, last_leg=loc['leg'])
            if round(t * 100) < 100:
                continue
            x, y, z = row['input']['position']
            difference = loc['pose']['z'] - z
            if reference is None:
                reference = (x, y, difference)
                result.update(reference_time_s=t, max_distance_from_start_m=0.)
            if previous_xy is not None:
                result['travel_distance_m'] += math.hypot(x - previous_xy[0], y - previous_xy[1])
            previous_xy = (x, y)
            error = difference - reference[2]
            if loc['leg'] in ('out', 'back'):
                legs = result['legs']
                index = loc.get('leg_index', int(loc['leg'] == 'back'))
                if not legs or index != legs[-1]['index']:
                    legs.append(dict(index=index, direction=loc['leg'], start_s=t,
                                     start_distance_m=result['travel_distance_m'],
                                     peak_abs_z_error_m=abs(error), peak_z_error_m=error, peak_time_s=t))
                segment = legs[-1]
                if abs(error) > segment['peak_abs_z_error_m']:
                    segment.update(peak_abs_z_error_m=abs(error), peak_z_error_m=error, peak_time_s=t)
                segment.update(end_s=t, end_distance_m=result['travel_distance_m'], end_z_error_m=error)
            if result['peak_abs_z_error_m'] is None or abs(error) > result['peak_abs_z_error_m']:
                result.update(peak_abs_z_error_m=abs(error), peak_z_error_time_s=t)
            travel = math.hypot(x - reference[0], y - reference[1])
            result['max_distance_from_start_m'] = max(result['max_distance_from_start_m'], travel)
            result.update(final_distance_from_start_m=travel, final_z_error_m=error)
            if loc['leg'] == 'back' and result['turnaround_time_s'] is None:
                result.update(turnaround_time_s=t, turnaround_z_error_m=error)
    furthest = distance + shuttle[0] if shuttle else distance
    endpoint = distance if shuttle else 0.
    segments = 2 * shuttle[1] + int(distance > 0) if shuttle else 2
    tolerance = min(furthest / 2, arrival + .25)
    result['out_and_back_completed'] = (
        result['last_leg'] == 'done' and len(result['legs']) == segments and
        result['max_distance_from_start_m'] >= furthest - tolerance and
        abs(result['final_distance_from_start_m'] - endpoint) <= tolerance)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=REPO / 'outputs/localization_courses_balanced/flat')
    parser.add_argument('--config', type=Path, default=REPO / 'examples/agriculture/localization_drive.json')
    parser.add_argument('--output', type=Path, help='Fresh rollout directory')
    parser.add_argument('--mode', choices=('truth', 'odometry'))
    parser.add_argument('--chassis-pitch-deg', type=json.loads, help='Physical pitch in degrees, or JSON [[time, degrees], ...]')
    pitches = parser.add_mutually_exclusive_group()
    pitches.add_argument('--reverse-pitch-deg', type=float, help='Physical pitch on each commanded reverse leg')
    pitches.add_argument('--turnaround-pitch-deg', type=float, help='Hold this physical pitch from the start of the reverse leg')
    parser.add_argument('--package', type=Path, help='CPU obstacle-grid package directory')
    parser.add_argument('--host-build', type=Path, help='Verified rhizome host build')
    parser.add_argument('--shuttle', nargs=2, metavar=('LEG', 'CYCLES'), help='Drive LEG metres forward and back CYCLES times')
    parser.add_argument('--distance', '--outbound', type=float, help='Outbound distance in metres')
    parser.add_argument('--speed', type=float, help='LocalControl speed limit in m/s')
    parser.add_argument('--seconds', type=float, help='Override simulation duration for truncated or extended runs')
    parser.add_argument('--timeout', type=float, default=7200., help='Wall time budget in seconds')
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve()
    if not (scene / 'scene.mjz').is_file():
        parser.error(f'Missing {scene}/scene.mjz. Build the courses with scripts/build_localization_courses.py')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output.expanduser().resolve() if args.output else REPO / 'outputs' / ('localization_' + stamp)
    config = json.loads(args.config.read_text())
    ctrl = config['controller']
    if 'arm_commands' not in config:
        # The Gen3 home-pitch tools dig into the soil while reversing.
        profile = json.loads((scene / 'robot_profile.json').read_text())
        config['arm_commands'] = {arm['id']: [dict(time=0, yaw_rad=0., pitch_rad=0.)]
                                  for arm in profile.get('arms', [])}
    if args.host_build:
        ctrl['host_build'] = str(args.host_build.resolve())
    if args.package:
        ctrl['nav_model']['package'] = str(args.package.resolve())
    if args.chassis_pitch_deg is not None:
        ctrl.setdefault('localization', {})['chassis_pitch_deg'] = args.chassis_pitch_deg
    if args.reverse_pitch_deg is not None:
        ctrl.setdefault('localization', {}).pop('turnaround_pitch_deg', None)
        ctrl['localization']['reverse_pitch_deg'] = args.reverse_pitch_deg
    if args.turnaround_pitch_deg is not None:
        ctrl.setdefault('localization', {}).pop('reverse_pitch_deg', None)
        ctrl.setdefault('localization', {})['turnaround_pitch_deg'] = args.turnaround_pitch_deg
    if args.mode:
        ctrl.setdefault('localization', {})['mode'] = args.mode
    if args.shuttle:
        try:
            leg, cycles = float(args.shuttle[0]), int(args.shuttle[1])
        except ValueError:
            parser.error('--shuttle requires a distance and an integer cycle count')
        if not math.isfinite(leg) or leg <= 0 or cycles < 1:
            parser.error('--shuttle requires a positive finite distance and a positive cycle count')
        ctrl['shuttle'] = [leg, cycles]
    shuttle = ctrl.get('shuttle')
    distance = 0. if args.shuttle else ctrl.get('out_and_back_m', 0. if shuttle else 40.)
    if args.distance is not None:
        distance = args.distance
    speed = ctrl.get('parameters', {}).get('auto_speed_max_ms', .45) if args.speed is None else args.speed
    if not math.isfinite(distance) or (distance < 0 or distance == 0 and not shuttle) or not .3 <= speed <= .6:
        parser.error('Choose a finite outbound distance (zero allowed with --shuttle) and a speed in [0.3, 0.6] m/s')
    ctrl['out_and_back_m'] = distance
    ctrl.setdefault('parameters', {})['auto_speed_max_ms'] = speed
    route_length = distance + 2 * shuttle[0] * shuttle[1] if shuttle else 2 * distance
    budget = math.ceil(1.2 * route_length / speed + (2 * shuttle[1] if shuttle else 0) + 10)
    seconds = args.seconds if args.seconds is not None else config.get('seconds', budget)
    os.chdir(REPO)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('LP_NUM_THREADS', '2')
    from scene_pipeline.agriculture_drive import run
    report = run(scene, output, config=config, seconds=seconds, video=False, timeout=args.timeout)
    metrics = score(output / 'rhizome/messages.jsonl', distance, ctrl['parameters'].get('arrival_distance', .25), shuttle)
    summary = dict(output=str(output), simulation_passed=report['passed'], failure=report['failure'], **metrics)
    write_json(output / 'localization_score.json', summary)
    print(json.dumps(summary, indent=2))
    if not report['passed']:
        return 1
    return 0 if metrics['out_and_back_completed'] or args.seconds is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
