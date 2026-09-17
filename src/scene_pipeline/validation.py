"""Layered checks. Contact-filtered physics and geometric clearance are distinct."""
import math
from pathlib import Path

import mujoco
import numpy as np

from .contracts import read_json, write_json, digest
from .registry import POLICY, fingerprint, TOP_BANDS_M, DOOR_CLEAR_WIDTH_M, MASS_BANDS_KG


def load(root):
    return mujoco.MjSpec.from_zip(str(Path(root)/'scene.mjz')).compile()


def aabbs(model,data):
    rotations=data.geom_xmat.reshape(-1,3,3)
    centers=data.geom_xpos+np.einsum('nij,nj->ni',rotations,model.geom_aabb[:,:3])
    half=np.einsum('nij,nj->ni',abs(rotations),model.geom_aabb[:,3:])
    return centers-half,centers+half


def geometric_overlap(model,data):
    """Narrow-phase distances bypass contact masks, after a 3D broad phase.

    Same rigid body geoms may overlap by construction; visual geometry excluded.
    This is sampled collision geometry, not a proof about arbitrary render meshes.
    """
    lo,hi=aabbs(model,data)
    colliders=np.flatnonzero((model.geom_contype!=0)|(model.geom_conaffinity!=0))
    worst=0.; pair=None
    for n,i in enumerate(colliders):
        candidates=colliders[n+1:]
        candidates=candidates[(model.geom_bodyid[candidates]!=model.geom_bodyid[i]) & np.all(lo[candidates]<hi[i],axis=1) & np.all(hi[candidates]>lo[i],axis=1)]
        for j in candidates:
            # Geoms attached to different fixed bodies form one rigid component.
            if model.body_weldid[model.geom_bodyid[i]]==model.body_weldid[model.geom_bodyid[j]]:
                continue
            distance=mujoco.mj_geomDistance(model,data,int(i),int(j),.01,None)
            if distance < -worst:
                worst=-float(distance); pair=[model.geom(int(i)).name,model.geom(int(j)).name]
    return worst,pair


def probe(model, manifest=None, *, settle=True, geometric=True):
    data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    initial=data.qpos.copy(); initial_pos=data.xipos.copy()
    initial_quat=data.xquat.copy(); joint_drift=np.zeros(model.nq)
    body_drift=0.; body_angle=0.; penetration=0.; finite=True
    warnings=np.zeros(len(data.warning),dtype=int)
    if settle:
        for step in range(round(POLICY['settle_s']/model.opt.timestep)):
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all(): finite=False; break
            joint_drift=np.maximum(joint_drift,abs(data.qpos-initial))
            body_drift=max(body_drift,float(np.linalg.norm(data.xipos-initial_pos,axis=1).max()))
            dots=np.clip(abs(np.einsum('ij,ij->i',data.xquat,initial_quat)),0,1)
            body_angle=max(body_angle,float((2*np.arccos(dots)).max()))
            penetration=max([penetration]+[-float(c.dist) for c in data.contact])
            warnings=np.maximum(warnings,data.warning.number)
    clock_ok=not settle or abs(data.time-POLICY['settle_s'])<1e-8
    scalar_pass=True
    for j in range(model.njnt):
        kind=model.jnt_type[j]; adr=model.jnt_qposadr[j]
        if kind==mujoco.mjtJoint.mjJNT_HINGE:
            scalar_pass &= joint_drift[adr] <= POLICY['hinge_drift_rad']
        elif kind==mujoco.mjtJoint.mjJNT_SLIDE:
            scalar_pass &= joint_drift[adr] <= POLICY['slide_drift_m']
    mujoco.mj_resetData(model,data); mujoco.mj_forward(model,data)
    closed,closed_pair=geometric_overlap(model,data) if geometric else (None,None)
    sweeps=[]; prerequisites={}
    if manifest:
        for name,instance in manifest['instances'].items():
            prefix=name+'/' if name else ''
            for a in instance['affordances']:
                prerequisites[prefix+a['joint']]={prefix+k:v for k,v in a['requires'].items()}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)): continue
        name=model.joint(j).name; limits=model.jnt_range[j] if model.jnt_limited[j] else [-math.pi,math.pi]
        worst=0.; pair=None
        for value in np.linspace(*limits,21):
            mujoco.mj_resetData(model,data)
            for other,target in prerequisites.get(name,{}).items(): data.joint(other).qpos[0]=target
            data.qpos[model.jnt_qposadr[j]]=value
            mujoco.mj_forward(model,data)
            amount,contact=geometric_overlap(model,data) if geometric else (max([0.]+[-float(c.dist) for c in data.contact]),None)
            if amount>worst: worst,pair=amount,contact
        sweeps.append(dict(joint=name,max_penetration_m=worst,pair=pair,prerequisites=prerequisites.get(name,{}),
                           passed=worst<=POLICY['penetration_m'],method='21 kinematic samples; not dynamic reach proof'))
    dynamic=model.body_dofnum>0
    masses_ok=bool(np.all(model.body_mass[dynamic]>0) and np.all(model.body_inertia[dynamic]>0))
    physical_unverified=[]
    if manifest:
        physical_unverified=[k for k,v in manifest['instances'].items() if v['provenance']['kind']=='retrieved_source'
                             and v['provenance'].get('physical_profile')!=POLICY['version']]
    numerical=bool(finite and clock_ok and not any(warnings) and masses_ok)
    stability=bool(body_drift<=.005 and body_angle<=POLICY['hinge_drift_rad'] and scalar_pass and penetration<=.002)
    clearance=bool((closed is None or closed<=.002) and all(s['passed'] for s in sweeps))
    return dict(schema_version=1,passed=numerical and stability and clearance and not physical_unverified,
                checks=dict(numerical=numerical,stability=stability,collision_clearance=clearance,
                            physical_provenance=not physical_unverified),
                body_drift_m=body_drift,body_angle_rad=body_angle,joint_drift=joint_drift.tolist(),
                physics_penetration_m=penetration,closed_geometric_penetration_m=closed,closed_pair=closed_pair,
                warnings=warnings.tolist(),sweeps=sweeps,physical_unverified=physical_unverified,
                profile_hash=digest(dict(policy=fingerprint(),version='validation-v1',mujoco=mujoco.__version__)),
                limitations=['No calibrated physical accuracy claim','Finite collision sampling','Dynamic endpoint and robot manipulation checks are separate'])


def brief_checks(model,manifest,ir,*,robot_radius):
    """What the brief grades, read off the compiled model: scale and mass bands, robot access, pairwise articulation, density."""
    from shapely.geometry import Polygon,Point,box
    from shapely.ops import unary_union
    data=mujoco.MjData(model);mujoco.mj_forward(model,data);lo,hi=aabbs(model,data)
    colliders=(model.geom_contype!=0)|(model.geom_conaffinity!=0)
    metrics=[]
    for name,inst in manifest['instances'].items():
        category=inst['category'];prefix=name+'/'
        geoms=[i for i in range(model.ngeom) if colliders[i] and model.geom(i).name.startswith(prefix)]
        mass=float(sum(model.body_mass[b] for b in range(model.nbody) if model.body(b).name.startswith(prefix)))
        band=MASS_BANDS_KG.get(category)
        entry=dict(instance=name,category=category,mass_kg=mass,mass_band_kg=band,mass_ok=None if band is None else bool(band[0]<=mass<=band[1]))
        if category in TOP_BANDS_M and geoms:
            top=float(hi[geoms][:,2].max());entry.update(top_m=top,top_band_m=TOP_BANDS_M[category],top_ok=bool(TOP_BANDS_M[category][0]<=top<=TOP_BANDS_M[category][1]))
        if category=='door':
            width=inst['dimensions'][0];entry.update(clear_width_m=width,width_ok=bool(DOOR_CLEAR_WIDTH_M[0]<=width<=DOOR_CLEAR_WIDTH_M[1]))
        metrics.append(entry)
    openings=[dict(id=o['id'],clear_width_m=o['width'],ok=bool(DOOR_CLEAR_WIDTH_M[0]<=o['width']<=DOOR_CLEAR_WIDTH_M[1]))
              for o in ir['openings'] if o.get('kind','door')=='door']
    # Access: a disc of robot radius must connect the entrance to every fixture's front, through
    # the compiled static colliders in the 0.1-1.0 m band (walls, carcasses; not floors or wall cabinets).
    rooms=unary_union([Polygon(r['polygon']) for r in ir['rooms']])
    band=[i for i in np.flatnonzero(colliders) if lo[i,2]<1. and hi[i,2]>.1]
    free=rooms.buffer(-robot_radius)
    if band:free=free.difference(unary_union([box(lo[i,0],lo[i,1],hi[i,0],hi[i,1]) for i in band]).buffer(robot_radius))
    parts=list(free.geoms) if free.geom_type=='MultiPolygon' else ([] if free.is_empty else [free])
    def component(xy,tolerance=robot_radius+.15):
        distances=[(part.distance(Point(xy)),k) for k,part in enumerate(parts)]
        nearest=min(distances,default=None)
        return None if nearest is None or nearest[0]>tolerance else nearest[1]
    entrance=next((o for o in ir['openings'] if 'exterior' in o.get('rooms',[])),None)
    entry=component(entrance['position'][:2]) if entrance else None
    unreachable=[]
    for name,inst in manifest['instances'].items():
        if inst.get('support_parent') or inst['category']=='door':continue
        c,s=math.cos(inst['yaw']),math.sin(inst['yaw']);front=-(inst['dimensions'][1]/2+robot_radius+.05)
        approach=(inst['position'][0]-s*front,inst['position'][1]+c*front)
        k=component(approach)
        if k is None or k!=entry:unreachable.append(name)
    access=dict(robot_radius_m=robot_radius,entrance_reachable=entry is not None,unreachable=unreachable,components=len(parts),
                model='disc through compiled closed-state colliders; not manipulation reachability')
    # Pairwise articulation: neighbours whose swept volumes overlap are opened together.
    articulated=[(n,i) for n,i in manifest['instances'].items() if i['affordances']]
    pairwise=[]
    for a in range(len(articulated)):
        for b in range(a+1,len(articulated)):
            (na,ia),(nb,ib)=articulated[a],articulated[b]
            if not all(ia['bounds'][0][k]<ib['bounds'][1][k]+.002 and ib['bounds'][0][k]<ia['bounds'][1][k]+.002 for k in range(3)):continue
            mujoco.mj_resetData(model,data)
            for n,i in ((na,ia),(nb,ib)):
                for affordance in i['affordances']:
                    for other,target in affordance['requires'].items():data.joint(n+'/'+other).qpos[0]=target
                    data.joint(n+'/'+affordance['joint']).qpos[0]=affordance['open']
            mujoco.mj_forward(model,data);amount,pair=geometric_overlap(model,data)
            pairwise.append(dict(instances=[na,nb],max_penetration_m=amount,pair=pair,passed=bool(amount<=POLICY['penetration_m'])))
    density=dict(objects_per_m2=len(manifest['instances'])/ir['meta']['area_m2'],articulated=len(articulated),
                 supported=sum(bool(i.get('support_parent')) for i in manifest['instances'].values()),note='reported, not gated')
    return dict(metrics=metrics,openings=openings,access=access,pairwise_articulation=pairwise,density=density)


def validate_scene(root, *, require_articulated=True, robot_radius=None):
    root=Path(root);manifest=read_json(root/'manifest.json');model=load(root)
    result=probe(model,manifest)
    ir=read_json(root/'ir.json')
    articulated=sum(bool(v['affordances']) for v in manifest['instances'].values())
    result['articulated_objects']=articulated
    result['checks']['five_articulated_objects']=articulated>=5
    if require_articulated:result['passed'] &= articulated>=5
    result.update(brief_checks(model,manifest,ir,robot_radius=POLICY['robot_radius_m'] if robot_radius is None else robot_radius))
    checks=result['checks']
    checks['metric_bands']=all(m.get('mass_ok') is not False and m.get('top_ok') is not False and m.get('width_ok') is not False for m in result['metrics'])
    checks['door_widths']=all(o['ok'] for o in result['openings'])
    checks['robot_access']=result['access']['entrance_reachable'] and not result['access']['unreachable']
    checks['pairwise_articulation']=all(p['passed'] for p in result['pairwise_articulation'])
    result['passed']&=checks['metric_bands'] and checks['robot_access'] and checks['pairwise_articulation']
    # Opening widths gate the brief profile only: the sampled-architecture profile still scales
    # apertures with area and reports them until its door width becomes an engineering default.
    if require_articulated:result['passed']&=checks['door_widths']
    result['acceptance_profile']='brief_five_articulated' if require_articulated else 'scene_physics_no_articulated_quota'
    result['area_m2']=ir['meta']['area_m2']
    write_json(root/'validation.json',result)
    return result


def validate_asset(folder, promote=False):
    folder=Path(folder); asset=read_json(folder/'asset.json')
    model=mujoco.MjSpec.from_zip(str(folder/asset['model'])).compile()
    manifest={'instances':{'':asset}}
    # Asset-local joint names do not have an instance prefix.
    result=probe(model, manifest)
    if asset['provenance']['kind']=='retrieved_source' and asset['provenance'].get('physical_profile')!=POLICY['version']:
        result['checks']['physical_provenance']=False; result['passed']=False
        result['physical_unverified']=['Source-inferred density requires an approved physical profile']
    write_json(folder/'validation.json',result)
    if promote:
        from .contracts import PipelineError
        if not result['passed']:
            raise PipelineError('ASSET_UNVERIFIED','Cannot promote failing asset',result)
        from .provenance import classify
        asset['state']='verified'; asset['validation']=result['profile_hash']
        asset['source_classification']=classify(asset)
        write_json(folder/'asset.json',asset)
    return result
