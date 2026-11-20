"""Correlate native field, trajectory and motor publications with MuJoCo samples."""
from collections import Counter
import json
import math
from pathlib import Path


def _ns(stamp):
    return stamp['secs'] * 1_000_000_000 + stamp['nsecs']


def _refs(value):
    return tuple(value[axis]['position_ref'] for axis in ('pitch', 'yaw'))


def _read_messages(stream):
    slots = [dict(detections={}, fields={}, selected={}, activated={}, commands={}, ticks=[])
             for _ in (0, 1)]
    pending = [None, None]
    for line in stream:
        msg = json.loads(line)
        topic, _, slot = msg['header']['topic'].rpartition('/')
        if slot not in ('0', '1'):
            continue
        slot = int(slot)
        data = slots[slot]
        stamp = _ns(msg['header']['timestamp'])
        if topic == '/vision/crop_infs':
            data['detections'][_ns(msg['camera_timestamp'])] = (stamp, Counter(
                detection['class_label'] for detection in msg['detections']))
        elif topic == '/trajectory/field':
            data['fields'][msg['capture_ns']] = msg['published_ns']
        elif topic == '/trajectory/selected':
            data['selected'][msg['start_ns']] = (msg['source_capture_ns'], stamp,
                [point['pitch_action'] for point in msg['points']])
        elif topic == '/trajectory/activated':
            data['activated'][msg['start_ns']] = (msg['source_capture_ns'], msg['activation_ns'])
        elif topic == '/arm_planner/plan':
            data['commands'][stamp] = _refs(msg)
            pending[slot] = stamp
        elif topic == '/trajectory/status':
            # Each trajectory worker publishes its command before its status on the same socket.
            data['ticks'].append((pending[slot], msg['plan_start_ns'], msg['idle'], msg['index'],
                                  (msg['target_pitch'], msg['target_yaw'])))
            pending[slot] = None
    return slots


def _analyze_arm(data, name, rows, start, end):
    def active(stamp):
        return start <= stamp <= end

    has_actual = bool(rows) and all('applied_commands' in row and 'end_measured' in row for row in rows)
    if rows and 'camera_capture_ns' in rows[0]:
        sent = {row['camera_capture_ns'] for row in rows if row['camera_capture_ns'] and active(row['camera_capture_ns'])}
    else:
        sent = {row['frame_ns'] for i, row in enumerate(rows)
            if i and active(row['frame_ns']) and row['camera_frames'][name] > rows[i-1]['camera_frames'][name]}
    det, fields = data['detections'], data['fields']
    captures = {ts for ts in fields.keys() & det.keys() if active(ts)}
    selected, activated, commands = data['selected'], data['activated'], data['commands']
    matched = {plan: capture for plan, (capture, ts) in activated.items()
               if capture in captures and capture in sent and active(ts) and plan in selected
               and selected[plan][0] == capture and active(selected[plan][1])}
    counts = dict(sent_frames=len(sent), inference_frames=sum(map(active, det)),
                  field_frames=sum(map(active, fields)), paired_captures=len(captures),
                  selected_plans=sum(active(v[1]) for v in selected.values()),
                  activations=sum(active(v[1]) for v in activated.values()),
                  matched_activations=len(matched), emitted_commands=sum(map(active, commands)),
                  idle_ticks=0, unmatched_plan_ticks=0, activated_reference_ticks=0,
                  commanded_strikes=0, applied_strikes=0 if has_actual else None,
                  modified_reference_steps=0 if has_actual else None, matched_native_request_steps=0,
                  matched_applied_steps=0, applied_evidence_available=has_actual)
    labels = Counter()
    for ts in captures:
        labels.update(det[ts][1])
    counts['detections_by_class'] = dict(labels)

    requests = {}
    unmodified = set()
    for row in rows:
        command_ns = row.get('command_ns', {}).get(name)
        if command_ns not in commands or not active(command_ns):
            continue
        requested = tuple(row['commands'][name][axis] for axis in ('pitch', 'yaw'))
        if requested != commands[command_ns]:
            raise ValueError(f'{name}: world references differ from native command {command_ns}')
        exact = False
        if has_actual:
            if row['end_time_s'] <= row['time_s']:
                raise ValueError('Motor reference intervals must have positive duration')
            applied = row['applied_commands'][name]
            exact = all(math.isclose(applied[axis], value, rel_tol=0, abs_tol=1e-7)
                        for axis, value in zip(('pitch', 'yaw'), requested))
            if exact:
                unmodified.add(command_ns)
            else:
                counts['modified_reference_steps'] += 1
        requests.setdefault(command_ns, []).append((row, exact))

    was_striking = False
    applied_in_strike = False
    best = None
    best_rank = None
    first_chain = None
    for tick, plan, idle, index, targets in data['ticks']:
        if tick is None or tick > end:
            continue
        entry = selected.get(plan)
        activated_plan = entry and plan in activated and activated[plan][0] == entry[0]
        valid_index = activated_plan and not idle and 0 < index <= len(entry[2])
        refs = commands[tick]
        verified = valid_index and all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(targets, refs))
        striking = bool(verified and entry[2][index-1] == 1)
        if tick < start:
            was_striking = striking
            applied_in_strike = striking
            continue
        if idle:
            counts['idle_ticks'] += 1
        elif not verified:
            counts['unmatched_plan_ticks'] += 1
        else:
            counts['activated_reference_ticks'] += 1
        if striking and not was_striking:
            counts['commanded_strikes'] += 1
            applied_in_strike = False
        if striking and tick in unmodified and not applied_in_strike:
            counts['applied_strikes'] += 1
            applied_in_strike = True
        was_striking = striking
        if not verified or plan not in matched:
            continue
        capture = matched[plan]
        chain = dict(capture_ns=capture, field_published_ns=fields[capture], detection_published_ns=det[capture][0], detections=dict(det[capture][1]),
                     selected_ns=entry[1], plan_start_ns=plan, activation_ns=activated[plan][1],
                     command_ns=tick, pitch_action=entry[2][index-1], point_index=index-1)
        if first_chain is None:
            first_chain = chain
        for row, exact in requests.get(tick, []):
            counts['matched_native_request_steps'] += 1
            if not has_actual:
                continue
            counts['matched_applied_steps'] += int(exact)
            reference = row['applied_commands'][name]
            before = row['measured'][name]
            after = row['end_measured'][name]
            approach = max(abs(target-before[axis]) - abs(target-after[axis])
                           for axis, target in reference.items())
            rank = (exact, approach > 1e-5, striking, bool(det[capture][1]), approach)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = dict(**chain, world_step_s=row['time_s'], next_step_s=row['end_time_s'],
                            requested_rad=row['commands'][name], reference_rad=reference,
                            measured_rad={a: before[a] for a in reference},
                            next_measured_rad={a: after[a] for a in reference},
                            unmodified_reference=exact, moved_toward_reference=rank[1])
    counts['native_chain_example'] = first_chain
    counts['applied_chain_example'] = best
    return counts


def analyze(output, arm_names, start_ns, end_ns):
    """Use capture/plan/command IDs for exact joins over the active row pass."""
    root = Path(output)
    with (root/'world.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(arm_names) != 4 or start_ns > end_ns:
        raise ValueError('Expected four native arm slots and an ordered pass interval')
    arms = {}
    for device in (1, 2):
        with (root/str(device)/'messages.jsonl').open() as stream:
            slots = _read_messages(stream)
        for slot, data in enumerate(slots):
            name = arm_names[(device-1)*2+slot]
            arms[name] = _analyze_arm(data, name, rows, start_ns, end_ns)
    return dict(arms=arms, all_arms_traced=all(a['matched_applied_steps'] for a in arms.values()))
