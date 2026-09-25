"""Post-run weeding diagnostics for unassigned plants in a generated field.

No truth from this module enters perception/control. Capsule volumes measure
near-stem tool contacts; they are not botanical damage or removal models.
"""
import html
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .agriculture import sha256
from .contracts import read_json,write_json
from .weeding import arm_transform,swept_clearance


def analyze(root):
    root=Path(root);manifest=read_json(root/'manifest.json');rig=read_json(root/'rig.json')
    config=read_json(root/'drive_config.json')['controller']
    names=config.get('physical_arms',[config.get('physical_arm')])
    arms={a['id']:a for a in rig['profile']['arms'] if a['id'] in names}
    plants=[dict(id=name,category=p['category'],position_m=p['position_m'],
                 radius_m=.06 if p['category']=='crop' else .015,
                 height_m=.15 if p['category']=='crop' else .04)
            for name,p in manifest['instances'].items() if p['category'] in ('crop','weed')]
    tree=cKDTree([p['position_m'][:2] for p in plants]) if plants else None
    by_category={label:[p for p in plants if p['category']==label] for label in ('crop','weed')}
    trees={label:cKDTree([p['position_m'][:2] for p in values]) for label,values in by_category.items() if values}
    def associate(point,label):
        label=label.lower()
        if label not in trees:return None
        distance,index=trees[label].query(point[:2],distance_upper_bound=.06)
        return by_category[label][index]['id'] if np.isfinite(distance) else None
    reports={};all_events=[];message_hashes={}
    with h5py.File(root/'data.h5','r') as data:
        for name,arm in arms.items():
            records={p['id']:dict(category=p['category'],detected_at=None,tracked_at=None,selected_at=None,
                                  first_contact_at=None,ordered_contact_at=None) for p in plants}
            states={};flags={};pose=None;prediction_count=0;track_count=0
            path=root/'rhizome'/name/'messages.jsonl' if 'physical_arms' in config else root/'rhizome/messages.jsonl'
            message_hashes[name]=sha256(path)
            def mark(plant,stage,t):
                if plant is not None and records[plant][stage] is None:records[plant][stage]=t
            for line in path.read_text().splitlines():
                message=json.loads(line);t=message['simulation_time'];request=message['input'];reply=message['output']
                if message['op']=='step':
                    pose=arm_transform(request,arm);native=reply['arm'];state=native['state']
                    states[state]=states.get(state,0)+1
                    for d in native.get('diagnostics',[]):
                        for flag in ('in_reach_circle','in_reach_triangle','is_valid_candidate','strike_ready','is_locked_target'):
                            if d.get(flag):flags[flag]=flags.get(flag,0)+1
                        if d.get('is_locked_target'):
                            point=pose[0]+pose[1]@np.array([d[k] for k in ('x','y','z')])
                            mark(associate(point,d['class_label']),'selected_at',t)
                elif message['op'] in ('observe','infer') and request['camera']==arm['camera']:
                    detections=request['detections'] if message['op']=='observe' else reply['prediction']['detections']
                    prediction_count+=len(detections);track_count+=len(reply['tracks']['detections'])
                    for d in detections:
                        point=np.array([d['keypoint']['kp'][k] for k in ('x','y','z')])
                        if message['op']=='infer':
                            if pose is None:continue
                            point=pose[0]+pose[1]@point
                        mark(associate(point,d['class_label']),'detected_at',t)
                    for d in reply['tracks']['detections']:
                        point=np.array([d['keypoint']['kp'][k] for k in ('x','y','z')])
                        mark(associate(point,d['class_label']),'tracked_at',t)
            stream='weeding_tool_'+name if 'physical_arms' in config else 'weeding_tool'
            tool=data[stream];times=tool['time'][:];positions=tool['position_world'][:]
            rotations=tool['rotation_world'][:];halves=tool['half_size_m'][:];active=set()
            for index,t in enumerate(times):
                previous=max(0,index-1);p0,p1=positions[previous],positions[index]
                # Enclose every orientation of the swept box and all capsule
                # radii. This broad phase cannot omit a capsule intersection.
                center=(p0+p1)/2
                radius=np.linalg.norm(halves[index])+np.linalg.norm(p1-p0)/2+.06+.001
                candidates=tree.query_ball_point(center[:2],radius) if tree is not None else []
                contacts=set()
                for candidate in candidates:
                    plant=plants[candidate]
                    clearance=swept_clearance(p0,rotations[previous],p1,rotations[index],halves[index],plant)
                    if clearance>0:continue
                    plant_id=plant['id'];contacts.add(plant_id);mark(plant_id,'first_contact_at',float(t))
                    if plant_id not in active:
                        record=records[plant_id]
                        ordered=(all(record[k] is not None for k in ('detected_at','tracked_at','selected_at')) and
                                 record['detected_at']<=record['tracked_at']<=record['selected_at']<=times[previous])
                        if ordered:mark(plant_id,'ordered_contact_at',float(t))
                        all_events.append(dict(arm=name,plant=plant_id,category=plant['category'],time=float(t),
                                               clearance_m=clearance,ordered=bool(ordered)))
                active=contacts
            def identifiers(category,key):
                return [p for p,r in records.items() if r['category']==category and r[key] is not None]
            reports[name]=dict(camera=arm['camera'],prediction_samples=prediction_count,track_samples=track_count,
                native_state_samples=states,native_flag_samples=flags,tool_pose_samples=len(times),
                detected_weeds=identifiers('weed','detected_at'),tracked_weeds=identifiers('weed','tracked_at'),
                selected_weeds=identifiers('weed','selected_at'),contacted_weeds=identifiers('weed','first_contact_at'),
                ordered_weed_contacts=identifiers('weed','ordered_contact_at'),contacted_crops=identifiers('crop','first_contact_at'),
                plants={p:r for p,r in records.items() if any(v is not None for k,v in r.items() if k!='category')})
    def union(key):return sorted({p for a in reports.values() for p in a[key]})
    report=dict(schema_version=1,perception=config['perception'],field_counts={k:len(v) for k,v in by_category.items()},
        unique_detected_weeds=union('detected_weeds'),unique_selected_weeds=union('selected_weeds'),
        unique_contacted_weeds=union('contacted_weeds'),unique_ordered_weed_contacts=union('ordered_weed_contacts'),
        unique_contacted_crops=union('contacted_crops'),arms=reports,
        interpretation='Diagnostic near-stem capsule/box contacts; unassigned field plants, no whole-field pass threshold or biological removal',
        geometry=dict(weed_radius_m=.015,weed_axis_height_m=.04,crop_radius_m=.06,crop_axis_height_m=.15,
                      sweep_tolerance_m=.001,tool_pose_hz=500,association_radius_xy_m=.06),
        limitations=['Plant association uses nearest same-class stem in XY within 6 cm; close soybean neighbours can be ambiguous',
                     'The drive path covers only part of the field; counts are not whole-field recall or efficacy',
                     'Crop/weed capsule sizes are fixed benchmark tolerances, not measured plant geometry'],
        provenance=dict(scorer_sha256=sha256(__file__),geometry_scorer_sha256=sha256(Path(__file__).with_name('weeding.py')),
                        data_sha256=sha256(root/'data.h5'),manifest_sha256=sha256(root/'manifest.json'),messages_sha256=message_hashes))
    write_json(root/'field_weeding_report.json',report)
    with (root/'field_weeding_events.jsonl').open('w') as f:
        for event in sorted(all_events,key=lambda e:e['time']):f.write(json.dumps(event)+'\n')
    rows=[]
    for name,a in reports.items():
        cells=[html.escape(name),html.escape(a['camera']),*(str(len(a[k])) for k in
            ('detected_weeds','selected_weeds','ordered_weed_contacts','contacted_crops'))]
        rows.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    (root/'field_weeding.html').write_text('<!doctype html><meta charset="utf-8"><title>Field weeding diagnostics</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:12px;text-align:left;border-bottom:1px solid #ccc}video{width:100%}</style>'
        '<h1>Field weeding: '+html.escape(config['perception'])+'</h1>'
        '<p><a href="index.html">Run inspection</a> · <a href="field_weeding_report.json">Detailed report</a> · '
        '<a href="field_weeding_events.jsonl">Contact events</a></p>'
        '<p>Unique plants per arm; contacts are measured from swept tool geometry. Ordered contacts require prior detection, tracking and selection. '
        'Multiple arms may observe the same plant.</p><table><tr>'+''.join('<th>'+h+'</th>' for h in
        ('Arm','Camera','Detected weeds','Selected weeds','Ordered weed contacts','Crop contacts'))+'</tr>'+''.join(rows)+'</table>'
        '<p>Fixed near-stem proxy volumes; no biological removal. Close crop associations can be ambiguous. The path covers part of the field.</p>'
        '<video controls src="replay.mp4"></video>')
    return report
