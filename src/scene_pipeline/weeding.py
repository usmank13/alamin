"""Controlled scenes and independent geometric weeding acceptance.

Plant truth belongs to scene construction and evaluation, never neural inputs.
Contacts describe a tool/proxy intersection, not biological weed removal.
"""
import copy
import json
import math
from pathlib import Path
import shutil

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim_harness.export import export_mujoco
from sim_harness.scene import _read_spec
from .agriculture import (compile_environment, compose_robot, load_bundle, preview,
                          sha256)
from .contracts import PipelineError, read_json, write_json
from .rhizome_perception import quaternion_rotation


def controlled_scene(scene, config, output):
    """Relocate two existing Inicio plants into a flat, reproducible field."""
    scene=Path(scene);output=Path(output)
    if output.exists():raise PipelineError('OUTPUT_EXISTS',str(output))
    config=read_json(config) if isinstance(config,(str,Path)) else copy.deepcopy(config)
    if not isinstance(config,dict):raise PipelineError('WEEDING_CONFIG','Expected a scenario configuration object')
    profile=read_json(scene/'robot_profile.json')
    multi='arms' in config
    names=config.get('arms',[config.get('arm')])
    if (not isinstance(names,list) or not names or not all(isinstance(n,str) for n in names)
            or len(set(names))!=len(names) or (multi and 'arm' in config)):
        raise PipelineError('WEEDING_CONFIG','Select unique physical arms')
    arms=[next((a for a in profile.get('arms',[]) if a['id']==name),None) for name in names]
    if any(a is None for a in arms):raise PipelineError('WEEDING_CONFIG','Select physical arms from the source scene')
    arm=arms[0]
    plants=config.get('plants',[])
    if (not isinstance(plants,list) or len(plants)!=2*len(arms) or not all(isinstance(p,dict) for p in plants)
            or len({p.get('id') for p in plants})!=len(plants)):
        raise PipelineError('WEEDING_CONFIG','Specify one weed and one crop per arm with distinct IDs')
    for name in names:
        pair=[p for p in plants if p.get('arm',name if not multi else None)==name]
        if len(pair)!=2 or {p.get('category') for p in pair}!={'weed','crop'}:
            raise PipelineError('WEEDING_CONFIG','Each arm requires an assigned weed and crop')
    source=load_bundle(scene/'bundle');bundle=copy.deepcopy(source)
    selected=[]
    for plant in plants:
        candidates=[i for i in source['instances'] if i['category']==plant['category']
                    and (not plant.get('source_id') or i['id']==plant['source_id'])]
        if not candidates:raise PipelineError('WEEDING_CONFIG','No matching source plant')
        selected.append(candidates[0])
        position=np.asarray(plant['position_m'],dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all() or position[2]!=0:
            raise PipelineError('WEEDING_CONFIG','Plants require finite XYZ positions on the flat z=0 terrain')
        if np.any(np.abs(position[:2])>np.array([2.7,1.7])):
            raise PipelineError('WEEDING_CONFIG','Plant volumes must fit inside the 6 by 4 metre field')
        if not isinstance(plant['id'],str) or not __import__('re').fullmatch(r'[A-Za-z][A-Za-z0-9_]*',plant['id']):
            raise PipelineError('WEEDING_CONFIG','Invalid plant ID')
        for key in ('radius_m','height_m'):
            if not isinstance(plant.get(key),(float,int)) or not 0<plant[key]<=.3:
                raise PipelineError('WEEDING_CONFIG',f'{key} must be positive and at most 0.3 m')
    output.mkdir(parents=True);root=output/'bundle'
    shutil.copytree(scene/'bundle',root)
    bounds=np.array([[-3.,-2.],[3.,2.]])
    bundle['terrain']=dict(file='terrain.npz',bounds_m=bounds.tolist(),grid_order='rows +Y, columns +X',
                           cell_m=[.05,.05],resolved=dict(mode='flat',elevation_m=0))
    np.savez_compressed(root/'terrain.npz',elevation_m=np.zeros((81,121),dtype=np.float32))
    bundle['sha256']['terrain.npz']=sha256(root/'terrain.npz')
    bundle['instances']=[];bundle['rows']=[]
    for placement,item in zip(plants,selected):
        item=copy.deepcopy(item);delta=np.asarray(placement['position_m'])-item['position_m']
        item.update(id=placement['id'],position_m=placement['position_m'],row_id=None,
                    source_instance_id=item['id'])
        for index,mesh in enumerate(item['meshes']):
            with np.load(root/mesh['file'],allow_pickle=False) as values:
                vertices=values['vertices']+delta;faces=values['faces']
            filename=f'meshes/{item["id"]}_{index}.npz'
            np.savez_compressed(root/filename,vertices=vertices,faces=faces)
            mesh.update(file=filename,bounds_m=[vertices.min(0).tolist(),vertices.max(0).tolist()])
            bundle['sha256'][filename]=sha256(root/filename)
        bundle['instances'].append(item)
    bundle['input_config']['simulation']={}
    bundle['provenance']['controlled_weeding']=dict(source_scene_sha256=sha256(scene/'scene.mjz'),config=config)
    write_json(root/'bundle.json',bundle)
    environment,manifest=compile_environment(bundle,root)
    manifest['weeding_benchmark']=dict(schema_version=1,arm=arm['id'],camera=arm['camera'],plants=plants,
        association_radius_m=.06,sweep_tolerance_m=.001,
        interpretation='Vertical capsule targets/protection volumes and swept MuJoCo tool box; no plant forces, removal or biological efficacy')
    if multi:
        manifest['weeding_benchmark'].pop('arm');manifest['weeding_benchmark'].pop('camera')
        manifest['weeding_benchmark'].update(schema_version=2,arms=names,cameras={a['id']:a['camera'] for a in arms})
    # Hold all tools above soil during initial perception; motion is native after that.
    for payload in profile['arms']:
        profile['initial_joint_positions'][payload['pitch_joint']]=-.2
        profile['initial_controls'][payload['pitch_joint']]=-.2
    spawn=config.get('spawn',dict(position_m=[-.5,0],yaw_rad=0))
    robot=_read_spec(scene/'robot.mjz')
    spec,model,data=compose_robot(environment,robot,profile,spawn,bounds)
    export_mujoco(environment,mujoco.MjData(environment.compile()),output/'environment.mjz')
    from .gen3 import initialize,robot_preview
    rm=robot.compile();rd=mujoco.MjData(rm);initialize(rm,rd,profile)
    export_mujoco(robot,rd,output/'robot.mjz');export_mujoco(spec,data,output/'scene.mjz')
    write_json(output/'robot_profile.json',profile);write_json(output/'manifest.json',manifest)
    write_json(output/'generation.json',dict(domain='agriculture',spawn=spawn,provenance=bundle['provenance'],
        limitations=bundle['limitations'],robot_dynamics=profile['provenance'],weeding_benchmark=config))
    # Both runs use exactly the same scene, camera, drive path and native parameters.
    for mode in ('oracle_visible','neural'):
        controller=dict(root=config.get('rhizome_root','~/Desktop/code/rhizome'),
            waypoints=config.get('waypoints',[dict(xy=[-.5,0]),dict(xy=[.3,0])]),
            parameters=dict(auto_speed_max_ms=.12,arrival_distance=.04),
            perception=mode,physical_arm=arm['id'],targeting_diagnostics=True)
        if multi:
            controller.pop('physical_arm');controller['physical_arms']=names
        if mode=='neural':
            controller.update(python=config.get('inference_python','~/Desktop/code/ml-aigen-tools/venv312/bin/python'),
                neural=dict(package=config.get('model_package','/data/models/edge_packages/multicrop_td_wzmle4cf_grid'),
                            crop_species=next(p['species'] for p in selected if p['category']=='crop'),device='cpu'))
        write_json(output/(mode+'.json'),dict(seconds=config.get('seconds',10),
            camera=dict(width=320,height=180,names=[a['camera'] for a in arms]),controller=controller))
    robot_preview(output,model,data);preview(output,model,data)
    report=dict(passed=True,kind='controlled_weeding_scene',plants=len(plants),arms=names,output=str(output))
    write_json(output/'validation.json',report)
    return report


def segment_box_distance(a,b,half):
    """Exact Euclidean distance between a line segment and axis-aligned box."""
    a=np.asarray(a);d=np.asarray(b)-a;half=np.asarray(half)
    cuts=[0.,1.]
    for axis in range(3):
        if abs(d[axis])>1e-15:
            cuts.extend(float(t) for t in ((-half[axis]-a[axis])/d[axis],(half[axis]-a[axis])/d[axis]) if 0<t<1)
    cuts=sorted(cuts);candidates=list(cuts)
    for low,high in zip(cuts,cuts[1:]):
        mid=a+d*((low+high)/2);active=np.abs(mid)>half
        if not active.any():return 0.
        offset=a[active]-np.sign(mid[active])*half[active]
        denom=float(d[active]@d[active])
        if denom:candidates.append(float(np.clip(-offset@d[active]/denom,low,high)))
    return float(min(np.linalg.norm(np.maximum(np.abs(a+d*t)-half,0)) for t in candidates))


def swept_clearance(p0,r0,p1,r1,half,plant,tolerance=.001):
    """Conservative capsule/box clearance with a <= tolerance/2 error bound.

    Sample the interpolated rigid pose densely enough that every box point
    moves <= tolerance between samples. Inflating by half that bound prevents
    translational or rotational tunnelling between the 500 Hz poses.
    """
    relative=Rotation.from_matrix(r0.T@r1).as_rotvec()
    motion=float(np.linalg.norm(p1-p0)+np.linalg.norm(half)*np.linalg.norm(relative))
    count=max(1,math.ceil(motion/tolerance))
    a=np.asarray(plant['position_m']);b=a+[0,0,plant['height_m']]
    best=math.inf
    for t in np.linspace(0,1,count+1):
        rotation=r0@Rotation.from_rotvec(relative*t).as_matrix();position=p0+(p1-p0)*t
        distance=segment_box_distance(rotation.T@(a-position),rotation.T@(b-position),half)
        best=min(best,distance-plant['radius_m'])
    return best-motion/(2*count)


def arm_transform(message,arm):
    rotation=quaternion_rotation(message['quaternion'])
    return (np.asarray(message['position'])+rotation@arm['position_chassis_m'],
            rotation@quaternion_rotation(arm['quaternion_chassis_wxyz']))


def evaluate(root,manifest,arm,*,simulation_passed,messages_path=None,tool_stream='weeding_tool',prefix=''):
    """Score captured truth after the run; never feed scoring data to control."""
    root=Path(root);benchmark=manifest['weeding_benchmark'];plants=benchmark['plants']
    records={p['id']:dict(category=p['category'],detected_at=None,tracked_at=None,selected_at=None,
        first_contact_at=None,contact_after_selection_at=None,min_clearance_m=None) for p in plants}
    events=[];frames=[];flag_counts={};last_state=None
    steps=[]
    def associate(point,label):
        candidates=[p for p in plants if p['category']==label.lower()]
        if not candidates:return None
        plant=min(candidates,key=lambda p:np.linalg.norm(np.asarray(p['position_m'])[:2]-point[:2]))
        return plant if np.linalg.norm(np.asarray(plant['position_m'])[:2]-point[:2])<=benchmark['association_radius_m'] else None
    def mark(plant,stage,t,**extra):
        if plant is not None and records[plant['id']][stage] is None:
            records[plant['id']][stage]=t
            events.append(dict(time=t,plant=plant['id'],event=stage,**extra))
    messages=Path(messages_path) if messages_path else root/'rhizome/messages.jsonl'
    if messages.exists():
        for line in messages.read_text().splitlines():
            message=json.loads(line);t=message['simulation_time'];request=message['input'];result=message['output']
            if message['op']=='step':
                steps.append(message)
                if 'arm' not in result:continue
                origin,rotation=arm_transform(request,arm);native=result['arm']
                if native['state']!=last_state:
                    events.append(dict(time=t,event='planner_state',state=native['state']));last_state=native['state']
                targets=[]
                for d in native.get('diagnostics',[]):
                    point=origin+rotation@np.array([d['x'],d['y'],d['z']])
                    labels=[name for name in ('in_reach_circle','in_reach_triangle','is_valid_candidate','strike_ready','is_locked_target') if d[name]]
                    for name in labels:flag_counts[name]=flag_counts.get(name,0)+1
                    if d['is_locked_target']:
                        mark(associate(point,d['class_label']),'selected_at',t,track_id=d['track_id'])
                    targets.append({**d,'position_world':point.tolist()})
                frames.append(dict(time=t,origin=origin.tolist(),rotation=rotation.tolist(),targets=targets,
                    state=native['state'],target_id=native['target_id'],commanded=native['commanded'],measured=native['measured'],
                    workspace=native.get('workspace')))
            elif message['op'] in ('observe','infer') and request['camera']==arm['camera']:
                if message['op']=='observe':detections=request['detections'];transform=None
                else:
                    detections=result['prediction']['detections']
                    transform=arm_transform(steps[-1]['input'],arm) if steps else None
                for d in detections:
                    point=np.array([d['keypoint']['kp'][k] for k in ('x','y','z')])
                    if transform is not None:point=transform[0]+transform[1]@point
                    mark(associate(point,d['class_label']),'detected_at',t)
                for d in result['tracks']['detections']:
                    point=np.array([d['keypoint']['kp'][k] for k in ('x','y','z')])
                    mark(associate(point,d['class_label']),'tracked_at',t,track_id=d['track_id'])
    with h5py.File(root/'data.h5','r') as recording:
        tool=recording.get(tool_stream)
        if tool is not None:
            times=tool['time'][:];positions=tool['position_world'][:];rotations=tool['rotation_world'][:];halves=tool['half_size_m'][:]
            active={p['id']:False for p in plants}
            for index,t in enumerate(times):
                previous=max(0,index-1)
                for plant in plants:
                    clearance=swept_clearance(positions[previous],rotations[previous],positions[index],rotations[index],
                        halves[index],plant,benchmark['sweep_tolerance_m'])
                    record=records[plant['id']]
                    record['min_clearance_m']=min(record['min_clearance_m'] if record['min_clearance_m'] is not None else math.inf,clearance)
                    contact=clearance<=0
                    if contact:
                        mark(plant,'first_contact_at',float(t))
                        if (not active[plant['id']] and record['selected_at'] is not None
                                and times[previous]>=record['selected_at']):
                            mark(plant,'contact_after_selection_at',float(t))
                    if contact!=active[plant['id']]:
                        events.append(dict(time=float(t),event='contact_enter' if contact else 'contact_exit',plant=plant['id'],clearance_m=clearance))
                    active[plant['id']]=contact
            for frame in frames:
                index=min(int(np.searchsorted(times,frame['time'])),len(times)-1)
                frame['tool_position']=positions[index].tolist();frame['tool_rotation']=rotations[index].tolist();frame['tool_half_size']=halves[index].tolist()
        for name in (arm['camera'],):
            group=recording.get('camera_'+name)
            if group is not None:
                capture_times=group['time'][:];camera_positions=group['position_world'][:];camera_rotations=group['rotation_world'][:]
                rig=read_json(root/'rig.json')['cameras'][name];inverse=np.linalg.inv(rig['K'])
                for frame in frames:
                    index=max(0,int(np.searchsorted(capture_times,frame['time'],side='right'))-1)
                    optical=camera_rotations[index]@np.diag([1,-1,-1]);center=camera_positions[index];polygon=[]
                    for uv in ((0,0),(rig['width'],0),(rig['width'],rig['height']),(0,rig['height'])):
                        ray=optical@inverse@[*uv,1]
                        if ray[2]<-1e-6:polygon.append((center-ray*center[2]/ray[2]).tolist())
                    frame['camera_footprint']=polygon
    weed=next(p for p in plants if p['category']=='weed' and p.get('arm',arm['id'])==arm['id'])
    w=records[weed['id']]
    stages=dict(detection=w['detected_at'] is not None,tracking=w['tracked_at'] is not None,
        selection=w['selected_at'] is not None,weed_contact=w['contact_after_selection_at'] is not None,
        crop_avoided=all(r['first_contact_at'] is None for r in records.values() if r['category']=='crop'))
    ordered=all(w[key] is not None for key in ('detected_at','tracked_at','selected_at','contact_after_selection_at'))
    if ordered:ordered=w['detected_at']<=w['tracked_at']<=w['selected_at']<=w['contact_after_selection_at']
    report=dict(schema_version=1,passed=bool(simulation_passed and all(stages.values()) and ordered),stages=stages,ordered=bool(ordered),
        plants=records,diagnostic_flag_samples=flag_counts,
        failure_reasons=[key for key,passed in stages.items() if not passed]+([] if simulation_passed else ['simulation'])
                        +(['stage_order'] if all(stages.values()) and not ordered else []),
        native_elimination_is_not_ground_truth=True,biological_removal_modeled=False,
        geometry=dict(tool='MuJoCo tool collision box',plant_volumes='vertical capsules',
                      sweep_tolerance_m=benchmark['sweep_tolerance_m'],sampling_hz=500),
        association_radius_m=benchmark['association_radius_m'],
        provenance=dict(scorer_sha256=sha256(__file__),data_sha256=sha256(root/'data.h5'),
                        messages_sha256=sha256(messages) if messages.exists() else None))
    write_json(root/(prefix+'weeding_report.json'),report)
    with (root/(prefix+'weeding_events.jsonl')).open('w') as stream:
        for event in sorted(events,key=lambda e:e['time']):stream.write(json.dumps(event,allow_nan=False)+'\n')
    write_json(root/(prefix+'weeding_trace.json'),dict(benchmark=benchmark,arm=arm,frames=frames))
    write_inspection(root,report,benchmark,arm,frames,prefix=prefix)
    return report


def write_inspection(root,report,benchmark,arm,frames,*,prefix=''):
    """Self-contained timeline; truth and planner overlays have separate labels."""
    payload=json.dumps(dict(report=report,benchmark=benchmark,arm=arm,frames=frames),allow_nan=False).replace('<','\\u003c')
    template=Path(__file__).with_name('weeding_view.html').read_text()
    if prefix:template=template.replace('<p><a href="index.html">','<p><a href="weeding.html">All arms</a> · <a href="index.html">')
    for name in ('weeding_report.json','weeding_events.jsonl','weeding_trace.json'):
        template=template.replace(name,prefix+name)
    (Path(root)/(prefix+'weeding.html')).write_text(template.replace('__WEEDING_DATA__',payload))


def evaluate_multi(root,manifest,arms,*,simulation_passed):
    """Require each assigned weed to be struck by its own tool; protect every crop."""
    root=Path(root)
    reports={name:evaluate(root,manifest,arm,simulation_passed=simulation_passed,
        messages_path=root/'rhizome'/name/'messages.jsonl',tool_stream='weeding_tool_'+name,prefix=name+'_')
        for name,arm in arms.items()}
    report=dict(schema_version=2,passed=all(r['passed'] for r in reports.values()),arms=reports,
        failure_reasons={name:r['failure_reasons'] for name,r in reports.items() if not r['passed']},
        crop_protection='Every active tool is checked against every crop volume',
        target_identity='Track IDs are local to each arm; geometric association determines plant identity',
        biological_removal_modeled=False)
    write_json(root/'weeding_report.json',report)
    write_multi_inspection(root,manifest,arms,report)
    return report


def write_multi_inspection(root,manifest,arms,report):
    import html
    root=Path(root);reports=report['arms']
    rows=[]
    for name,arm in arms.items():
        r=reports[name];weed=next(p for p in manifest['weeding_benchmark']['plants'] if p['category']=='weed' and p.get('arm',manifest['weeding_benchmark'].get('arm'))==name)
        result=r['plants'][weed['id']]
        cells=[f'<a href="{html.escape(name)}_weeding.html">{html.escape(name)}</a>',html.escape(arm['camera']),
            'PASS' if r['passed'] else 'FAIL',*('—' if result[k] is None else f"{result[k]:.3f}" for k in ('detected_at','tracked_at','selected_at','contact_after_selection_at')),
            'yes' if r['stages']['crop_avoided'] else 'NO']
        rows.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    (root/'weeding.html').write_text('<!doctype html><meta charset="utf-8"><title>Multi-arm weeding acceptance</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;background:#151a20;color:#eee}a{color:#8cf}'
        'td,th{padding:12px;text-align:left;border-bottom:1px solid #456}video{width:100%}</style>'
        '<h1>Multi-arm weeding acceptance: '+('PASS' if report['passed'] else 'FAIL')+'</h1>'
        '<p>Each arm must detect, track, select and contact its assigned weed in order. Every tool must avoid every crop. '
        'Times are simulation seconds. Select an arm for its interactive timeline.</p>'
        '<p><a href="weeding_report.json">Acceptance report</a> · <a href="index.html">Run inspection</a></p>'
        '<table><tr>'+''.join('<th>'+h+'</th>' for h in ('Arm','Camera','Result','Detection','Tracking','Selection','Contact','Crops avoided'))+
        '</tr>'+''.join(rows)+'</table><h2>Recorded physics</h2><video controls src="replay.mp4"></video>'
        '<p>Contacts are geometric proxy intersections. Plants are not removed; hardware dynamics and camera calibration remain approximate.</p>')
