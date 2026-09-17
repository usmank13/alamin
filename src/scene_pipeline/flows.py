"""Stock-robot execution. Only robot actuators receive controls during rollouts."""
import os
os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('LP_NUM_THREADS','1')
import collections
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.ndimage import binary_dilation,distance_transform_edt
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from PIL import Image
from shapely.geometry import Polygon,LineString,Point

from sim_harness.scene import _read_spec
from .contracts import PipelineError,read_json,write_json
from .sensors import attach_rig,sensor,noisy_ranges,NOISE,semantic_masks
from .dataset import Recorder,inspect
from .validation import aabbs


def belongs_to(model, geom, prefix):
    """Stock robot collision geoms can be unnamed; owning bodies are namespaced."""
    return model.body(int(model.geom_bodyid[geom])).name.startswith(prefix)


def mapping_approaches(ir,offset):
    """Use actual entrance geometry; legacy v1 uses its entrance-relative frame."""
    if ir['schema_version']==1:return [([offset,0,0],0.)]
    poses=[]
    for opening in ir['openings']:
        if opening['kind']!='door':continue
        aperture=Polygon(opening['polygon'])
        for room in ir['rooms']:
            polygon=Polygon(room['polygon']);points=room['polygon']
            for a,b in zip(points,points[1:]+points[:1]):
                edge=LineString([a,b])
                if aperture.buffer(1e-7).intersection(edge).length<=1e-5:continue
                tangent=(np.asarray(b)-a)/edge.length
                inward=np.array([-tangent[1],tangent[0]])*(1 if polygon.exterior.is_ccw else -1)
                midpoint=np.array(edge.interpolate(edge.project(aperture.centroid)).coords[0])
                xy=midpoint+offset*inward
                if polygon.covers(Point(xy)):
                    poses.append(([*map(float,xy),0.],math.atan2(inward[1],inward[0])))
    return poses


def robot_spec(root,flow):
    """Deterministic approach-pose search, rejecting occupied initial poses."""
    failures=[];ir=read_json(Path(root)/'ir.json')
    for offset in ([.9,1.,1.1,1.2] if flow=='interaction' else [.7,1.,1.3]):
        for placement in (mapping_approaches(ir,offset) if flow=='mapping' else [None]):
            try:
                return _robot_spec(root,flow,offset,placement)
            except PipelineError as exc:
                if exc.code!='ROBOT_PLACEMENT': raise
                failures.append({'offset_m':offset,'placement':placement,**exc.as_dict()})
    raise PipelineError('ROBOT_PLACEMENT','No clear initial robot pose among approach candidates',{'attempts':failures})


def _robot_spec(root,flow,offset,placement=None):
    root=Path(root);spec=_read_spec(root/'scene.mjz')
    # Older generated artifacts may disable this globally. Stock Menagerie
    # mechanisms require their native parent filtering; the geometric validator
    # is deliberately independent of these runtime filters.
    spec.option.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    for key in list(spec.keys):spec.delete(key)
    ir=read_json(root/'ir.json');manifest=read_json(root/'manifest.json')
    if flow=='mapping':
        path=Path('vendor/mujoco_menagerie/robot_soccer_kit/robot_soccer_kit.xml')
        prefix='base/';position,yaw=placement if placement is not None else ([offset,0,0],0.);target=None
        child=_read_spec(path)
        rig=attach_rig(child,'base','rig_',lidar=True)
    else:
        candidates=[o for o in ir['objects'] if o['category']=='drawer_unit']
        if not candidates: raise PipelineError('NO_INTERACTION_TARGET','No supported drawer interaction target')
        target=candidates[0];yaw=target['yaw']+math.pi/2
        R=Rotation.from_euler('z',target['yaw'])
        position=(np.array(target['position'])+R.apply([0,-offset,0])).tolist()
        path=Path('vendor/mujoco_menagerie/franka_emika_panda/panda.xml');prefix='arm/'
        child=_read_spec(path)
        child.body('hand').add_site(name='grasp',pos=[0,0,.1034],size=[.003,0,0],rgba=[0,0,0,0])
        rig=attach_rig(child,'link0','rig_',wrench_body='hand')
    source=child.compile();home=None
    if flow=='interaction':
        home=source.key('home').qpos.copy()
    for key in list(child.keys):child.delete(key)
    frame=spec.worldbody.add_frame(pos=position,quat=[math.cos(yaw/2),0,0,math.sin(yaw/2)])
    spec.attach(child,prefix=prefix,frame=frame)
    model=spec.compile();data=mujoco.MjData(model)
    if home is not None:
        for j in range(source.njnt):data.joint(prefix+source.joint(j).name).qpos[:]=home[source.jnt_qposadr[j]]
        for j in range(7):data.actuator(prefix+f'actuator{j+1}').ctrl[0]=home[j]
        data.actuator(prefix+'actuator8').ctrl[0]=255
    mujoco.mj_forward(model,data)
    # Preflight robot/environment penetration, without conflating self contact.
    max_overlap=0.;pair=None
    for c in data.contact:
        a=belongs_to(model,c.geom1,prefix);b=belongs_to(model,c.geom2,prefix)
        if a!=b and -float(c.dist)>max_overlap:
            max_overlap=-float(c.dist)
            pair=[model.body(int(model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
    if max_overlap>.002:raise PipelineError('ROBOT_PLACEMENT','Robot intersects environment',{'penetration_m':max_overlap,'bodies':pair})
    rig['placement']={'position':position,'yaw':yaw,'source':str(path),'prefix':prefix}
    return spec,model,data,manifest,rig,target


class Mapper:
    def __init__(self,ir,model,data,initial_xy=(.7,0.),initial_yaw=0.):
        points=np.array([p for room in ir['rooms'] for p in room['polygon']]);self.origin=points.min(axis=0)-.2
        self.res=.10;self.shape=tuple(np.ceil((points.max(axis=0)-self.origin+.2)/self.res).astype(int))
        self.logodds=np.zeros(self.shape);self.observed=np.zeros(self.shape,dtype=bool)
        self.initial_xy=np.asarray(initial_xy)
        self.pose=np.array([*initial_xy,initial_yaw]);self.path=[];self.last_plan=-1.;self.travel=0.
        self.scan_map={};self.scan_match_residual=None
        base=model.body('base/base').id;R=data.xmat[base].reshape(3,3)
        rows=[]
        for i in range(1,4):
            j=model.joint(f'base/wheel{i}_speed').id
            axis=R.T@data.xaxis[j];pos=R.T@(data.xanchor[j]-data.xpos[base]);direction=np.cross(axis,[0,0,1])[:2]
            # Native wheel contact envelope radius, engineering kinematic model.
            radius=.031
            rows.append([direction[0]/radius,direction[1]/radius,(pos[0]*direction[1]-pos[1]*direction[0])/radius])
        self.wheels=np.array(rows);self.inverse=np.linalg.inv(self.wheels)
        self.truth=np.zeros(self.shape,dtype=bool)
        self.legacy_truth=np.zeros(self.shape,dtype=bool)
        lo,hi=aabbs(model,data)
        xs=self.origin[0]+(np.arange(self.shape[0])+.5)*self.res
        ys=self.origin[1]+(np.arange(self.shape[1])+.5)*self.res
        for i in range(model.ngeom):
            if not (model.geom_contype[i] or model.geom_conaffinity[i]) or belongs_to(model,i,'base/'):continue
            if lo[i,2]<.23<hi[i,2]:
                # Occupancy at cell centers, not an extra row/column from inclusive
                # integer AABB endpoints. This convention is saved with the metric.
                self.legacy_truth |= ((xs>=lo[i,0])&(xs<=hi[i,0]))[:,None] & ((ys>=lo[i,1])&(ys<=hi[i,1]))[None,:]
                self.truth |= ((xs+self.res/2>=lo[i,0])&(xs-self.res/2<=hi[i,0]))[:,None] & ((ys+self.res/2>=lo[i,1])&(ys-self.res/2<=hi[i,1]))[None,:]

    def cell(self,xy):
        return tuple(np.clip(((np.asarray(xy)-self.origin)/self.res).astype(int),[0,0],np.array(self.shape)-1))

    def update_odometry(self,velocities,gyro,dt):
        v=self.inverse@velocities;theta=self.pose[2]
        step=np.array([math.cos(theta)*v[0]-math.sin(theta)*v[1],math.sin(theta)*v[0]+math.cos(theta)*v[1]])*dt
        self.pose[:2]+=step;self.pose[2]+=gyro*dt;self.travel+=float(np.linalg.norm(step))

    def observe(self,ranges,valid):
        valid=np.asarray(valid,dtype=bool)&np.isfinite(ranges)
        angles=np.linspace(-math.pi,math.pi,72,endpoint=False)
        local=np.column_stack([np.cos(angles[valid]),np.sin(angles[valid])])*ranges[valid,None]
        if len(self.scan_map)>30 and len(local)>10:
            tree=cKDTree(np.array(list(self.scan_map.values())))
            for _ in range(8):
                c,s=math.cos(self.pose[2]),math.sin(self.pose[2]);R=np.array([[c,-s],[s,c]])
                rotated=local@R.T;world=rotated+self.pose[:2];dist,index=tree.query(world,k=6)
                neighbors=tree.data[index];centers=neighbors.mean(axis=1)
                centered=neighbors-centers[:,None,:]
                covariance=np.einsum('nki,nkj->nij',centered,centered)/6
                values,vectors=np.linalg.eigh(covariance);normals=vectors[:,:,0]
                # Fit local surfaces, not sparse point correspondences. Tangential
                # sampling shifts must not be mistaken for robot yaw changes.
                keep=(dist[:,0]<.20)&(values[:,1]>.0001)&(values[:,0]<.15*values[:,1])
                if np.count_nonzero(keep)<10:break
                n=normals[keep];residual=np.einsum('ij,ij->i',n,world[keep]-centers[keep])
                angular=np.column_stack([-rotated[keep,1],rotated[keep,0]])
                J=np.column_stack([n,np.einsum('ij,ij->i',n,angular)])
                weights=np.minimum(1.,.03/np.maximum(abs(residual),1e-9))
                H=J.T@(weights[:,None]*J)
                if np.linalg.eigvalsh(H)[0]<1e-4:break
                delta=-np.linalg.solve(H+np.diag([.001,.001,.01]),J.T@(weights*residual))
                delta[:2]=np.clip(delta[:2],-.05,.05);delta[2]=np.clip(delta[2],-.02,.02)
                self.pose+=delta
                self.scan_match_residual=float(np.mean(abs(residual)))
                if np.linalg.norm(delta)<.0001:break
        c,s=math.cos(self.pose[2]),math.sin(self.pose[2]);R=np.array([[c,-s],[s,c]])
        for point in local@R.T+self.pose[:2]:
            key=tuple(np.round(point/.05).astype(int))
            # Fixed local references prevent a moving robot's pose errors from
            # dragging the entire registration map over time (positive feedback).
            self.scan_map.setdefault(key,point)
        free_cells=set();hit_cells=set()
        for r,hit,angle in zip(ranges,valid,np.linspace(-math.pi,math.pi,72,endpoint=False)):
            if not np.isfinite(r):continue
            direction=np.array([math.cos(angle+self.pose[2]),math.sin(angle+self.pose[2])])
            # Do not carve endpoint cells (or their uncertainty band) as free.
            # A half-cell diagonal plus 3 sigma protects noisy surface boundaries.
            margin=self.res/np.sqrt(2)+3*NOISE['range_sigma_m'] if hit else 0.
            samples=self.pose[:2]+np.arange(0,max(0,r-margin),self.res/3)[:,None]*direction
            cells=np.floor((samples-self.origin)/self.res).astype(int)
            cells=cells[np.all((cells>=0)&(cells<np.array(self.shape)),axis=1)]
            free_cells.update(map(tuple,cells))
            if hit:
                end=np.floor((self.pose[:2]+r*direction-self.origin)/self.res).astype(int)
                if np.all((end>=0)&(end<np.array(self.shape))):
                    hit_cells.add(tuple(end))
        # Fuse each cell once per scan; occupied evidence wins over crossing rays.
        for cell in free_cells-hit_cells:
            self.logodds[cell]-=.4;self.observed[cell]=True
        for cell in hit_cells:
            self.logodds[cell]+=1.2;self.observed[cell]=True
        np.clip(self.logodds,-8,8,out=self.logodds)

    def control(self,time,ranges):
        if np.count_nonzero(np.isfinite(ranges))<36:return np.zeros(3)
        start=self.cell(self.pose[:2])
        if time-self.last_plan>=1 or not self.path:
            blocked=binary_dilation(self.logodds>0,iterations=2)
            free=self.observed & ~blocked & (self.logodds<=0);free[start]=True
            queue=collections.deque([start]);parents={start:None};best=start;best_score=-1
            while queue:
                cell=queue.popleft();neighbors=[]
                for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    n=(cell[0]+dx,cell[1]+dy)
                    if 0<=n[0]<self.shape[0] and 0<=n[1]<self.shape[1]:neighbors.append(n)
                frontier=any(not self.observed[n] for n in neighbors)
                distance=np.linalg.norm(np.array(cell)-start)
                score=distance if frontier else distance*.2
                if score>best_score:best,best_score=cell,score
                for n in neighbors:
                    if free[n] and n not in parents:parents[n]=cell;queue.append(n)
            path=[]
            while best!=start:path.append(best);best=parents[best]
            self.path=path[::-1];self.last_plan=time
        while self.path and np.linalg.norm(self.origin+(np.array(self.path[0])+.5)*self.res-self.pose[:2])<.12:self.path.pop(0)
        if not self.path:return np.zeros(3)
        target=self.origin+(np.array(self.path[0])+.5)*self.res;delta=target-self.pose[:2]
        velocity=delta/max(np.linalg.norm(delta),.01)*.18
        theta=self.pose[2];local=np.array([math.cos(theta)*velocity[0]+math.sin(theta)*velocity[1],-math.sin(theta)*velocity[0]+math.cos(theta)*velocity[1],np.clip(-theta,-.5,.5)])
        # Local range safety overrides navigation without ground-truth geometry.
        angles=np.linspace(-math.pi,math.pi,72,endpoint=False);forward=np.cos(angles)*local[0]+np.sin(angles)*local[1]
        if np.any((forward>.08)&(ranges<.20)):local[:2]*=0
        return np.clip(self.wheels@local,-12,12)

    def report(self,output):
        occupied=self.logodds>0;mask=self.observed
        intersection=np.count_nonzero(occupied & self.truth & mask);union=np.count_nonzero((occupied|self.truth)&mask)
        iou=intersection/max(union,1)
        legacy_iou=np.count_nonzero(occupied & self.legacy_truth & mask)/max(np.count_nonzero((occupied|self.legacy_truth)&mask),1)
        free=~binary_dilation(self.truth,iterations=2)
        # Ground-truth reachable component is used for evaluation only.
        start=self.cell(self.initial_xy);reachable=np.zeros(self.shape,dtype=bool);queue=collections.deque([start]);reachable[start]=True
        while queue:
            x,y=queue.popleft()
            for a,b in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                if 0<=a<self.shape[0] and 0<=b<self.shape[1] and free[a,b] and not reachable[a,b]:reachable[a,b]=True;queue.append((a,b))
        coverage=np.count_nonzero(reachable&mask)/max(np.count_nonzero(reachable),1)
        distances=distance_transform_edt(~self.truth)*self.res
        error=float(distances[occupied&mask].mean()) if np.any(occupied&mask) else None
        image=np.full((*self.shape,3),128,dtype=np.uint8);image[mask]=230;image[occupied&mask]=20
        Image.fromarray(np.flip(image.transpose(1,0,2),axis=0)).resize((600,600)).save(Path(output)/'map.png')
        np.savez_compressed(Path(output)/'map.npz',logodds=self.logodds,observed=mask,truth=self.truth,legacy_truth=self.legacy_truth,reachable=reachable,origin=self.origin,resolution=self.res)
        return dict(metric_version='mapping-v2-cell-intersection',occupancy_iou=iou,legacy_cell_center_iou=legacy_iou,reachable_coverage=coverage,mean_occupied_distance_to_truth_m=error,odometry_travel_m=self.travel,
                    occupied_true_positive=int(intersection),occupied_false_positive=int(np.count_nonzero(occupied&~self.truth&mask)),
                    occupied_false_negative=int(np.count_nonzero(~occupied&self.truth&mask)),
                    scan_matching_residual_m=self.scan_match_residual,
                    passed=bool(np.any(mask) and np.isfinite(iou)),
                    acceptance='finite reported reconstruction metric with observations; no spec accuracy threshold',
                    evaluation='10 cm cell-intersection occupancy; observed-cell IoU; reachable free-space coverage; collider AABBs at 0.23m; not directly comparable to v1 center metric')


class DrawerController:
    def __init__(self,model,data,target,manifest):
        self.model=model;self.ik=mujoco.MjData(model);self.ik.qpos[:]=data.qpos
        self.ids=[model.joint(f'arm/joint{i}').id for i in range(1,8)]
        self.qadr=model.jnt_qposadr[self.ids];self.dofs=model.jnt_dofadr[self.ids]
        self.site=model.site('arm/grasp').id
        self.target=target;self.affordance=manifest['instances'][target['id']]['affordances'][0]
        self.joint=model.joint(target['id']+'/'+self.affordance['joint']).id
        self.range=model.jnt_range[self.joint].copy();self.min=0.;self.max=0.;self.contacts=0;self.final_q=0.;self.completed=False
        R=Rotation.from_euler('z',target['yaw']);self.direction=R.apply([0,-1,0]);self.R=R.as_matrix()@np.array([[-1,0,0],[0,0,1],[0,1,0]])
        self.closed=np.array(target['position'])+R.apply(self.affordance['point'])
        self.command=data.qpos[self.qadr].copy()

    def control(self,data,time):
        # Approach -> close fingers -> pull -> push -> release. No object actuator.
        if time<8: desired=self.closed+self.direction*.12; grip=255.
        elif time<14: desired=self.closed+self.direction*.12*(1-(time-8)/6);grip=255.
        elif time<19: desired=self.closed;grip=0.
        elif time<34: desired=self.closed+self.direction*self.range[1]*min(1,(time-19)/12);grip=0.
        elif time<49:
            progress=min(1,(time-34)/12)
            desired=self.closed+self.direction*(self.range[1]*(1-progress)-.01*progress);grip=0.
        else: desired=self.closed+self.direction*.12*min(1,(time-49)/5);grip=255.
        self.ik.qpos[:]=data.qpos
        self.ik.qpos[self.qadr]=self.command
        for _ in range(8):
            mujoco.mj_forward(self.model,self.ik)
            jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(self.model,self.ik,jp,jr,self.site)
            current=self.ik.site_xmat[self.site].reshape(3,3)
            error=np.r_[desired-self.ik.site_xpos[self.site],Rotation.from_matrix(self.R@current.T).as_rotvec()*.3]
            J=np.vstack([jp[:,self.dofs],jr[:,self.dofs]*.3])
            dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*.002,error)
            self.ik.qpos[self.qadr]=np.clip(self.ik.qpos[self.qadr]+np.clip(dq,-.12,.12),self.model.jnt_range[self.ids,0]+.01,self.model.jnt_range[self.ids,1]-.01)
        self.command=self.ik.qpos[self.qadr].copy()
        for i,q in enumerate(self.command):data.actuator(f'arm/actuator{i+1}').ctrl[0]=q
        data.actuator('arm/actuator8').ctrl[0]=grip
        q=float(data.qpos[self.model.jnt_qposadr[self.joint]]);self.min=min(self.min,q);self.max=max(self.max,q)
        self.final_q=q;self.completed=time>=54
        for c in data.contact:
            robot_a=belongs_to(self.model,c.geom1,'arm/');robot_b=belongs_to(self.model,c.geom2,'arm/')
            target_a=belongs_to(self.model,c.geom1,self.target['id']+'/');target_b=belongs_to(self.model,c.geom2,self.target['id']+'/')
            if (robot_a and target_b) or (robot_b and target_a):self.contacts+=1

    def report(self):
        error=abs(self.max-self.range[1]);closing_error=abs(self.final_q-self.range[0]);return dict(target=self.target['id'],joint=self.affordance['joint'],range=self.range.tolist(),
                observed_min=self.min,observed_max=self.max,upper_endpoint_error_m=error,contact_samples=self.contacts,
                closing_endpoint_error_m=closing_error,completed_sequence=self.completed,
                passed=bool(self.completed and error<=.005 and closing_error<=.005 and self.contacts>0),
                controller='Position/orientation IK -> native Panda actuators; no object commands; success must be measured')


def run(scene,flow,output,*,seconds=60,tier='full',seed=0):
    scene=Path(scene);output=Path(output)
    if output.exists():raise PipelineError('OUTPUT_EXISTS',str(output))
    if not 0<seconds<=60:raise PipelineError('FLOW_DURATION','Flow duration must be (0,60] simulated seconds')
    output.mkdir(parents=True);started=time.perf_counter()
    spec,model,data,manifest,rig,target=robot_spec(scene,flow)
    prefix=rig['placement']['prefix'];sensor_prefix=prefix+'rig_'
    initial=data.qpos.copy();initial_ctrl=data.ctrl.copy()
    spec.add_key(name='harness_initial',qpos=initial,ctrl=initial_ctrl);spec.to_zip(str(output/'rollout_scene.mjz'))
    write_json(output/'rig.json',rig)
    # Rendering must not change the robot's observations or trajectory.
    rng,camera_rng=np.random.default_rng(seed),np.random.default_rng(np.random.SeedSequence([seed,1]))
    recorder=Recorder(output/'data.h5',dict(schema_version=1,rig=rig,scene=manifest,flow=flow,tier=tier,seed=seed,
                                          semantics=dict(instance_ids={name:i+1 for i,name in enumerate(sorted(manifest['instances']))}),clock='simulation_seconds'))
    mapper=Mapper(read_json(scene/'ir.json'),model,data,rig['placement']['position'][:2],rig['placement']['yaw']) if flow=='mapping' else None
    controller=DrawerController(model,data,target,manifest) if flow=='interaction' else None
    renderer=mujoco.Renderer(model,height=240,width=320) if tier=='full' else None
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    ranges=np.full(72,8.);warnings=np.zeros(len(data.warning),dtype=int)
    frames=[]
    try:
        for tick in range(round(seconds/model.opt.timestep)+1):
            mujoco.mj_forward(model,data)
            t=float(data.time)
            if tick%5==0:
                gyro=sensor(data,sensor_prefix+'gyro');accel=sensor(data,sensor_prefix+'accelerometer')
                gyro_obs=gyro+rng.normal(0,NOISE['gyro_sigma_rad_s'],3);accel_obs=accel+rng.normal(0,NOISE['accelerometer_sigma_m_s2'],3)
                recorder.append('imu',t,gyro_truth=gyro,gyro=gyro_obs,accelerometer_truth=accel,accelerometer=accel_obs)
                if mapper:
                    velocities=np.array([data.joint(f'base/wheel{i}_speed').qvel[0] for i in range(1,4)])+rng.normal(0,NOISE['joint_sigma'],3)
                    if tick:mapper.update_odometry(velocities,float(gyro_obs[2]),.01)
                    controls=mapper.control(t,ranges)
                    for i,control in enumerate(controls):data.actuator(f'base/wheel{i+1}_speed').ctrl[0]=control
                    recorder.append('odometry',t,pose=mapper.pose.copy(),wheel_velocity=velocities)
                else:
                    controller.control(data,t)
                    force=sensor(data,sensor_prefix+'force');torque=sensor(data,sensor_prefix+'torque')
                    recorder.append('wrench',t,force_truth=force,force=force+rng.normal(0,NOISE['force_sigma_n'],3),torque_truth=torque,torque=torque+rng.normal(0,NOISE['torque_sigma_nm'],3))
                # ctrl is the command applied over the following integration interval.
                recorder.append('state',t,qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),body_position=data.xpos.copy(),body_quaternion=data.xquat.copy())
            if mapper and tick%25==0:
                truth,ranges,valid=noisy_ranges(data,sensor_prefix,rng);mapper.observe(ranges,valid)
                recorder.append('range',t,truth=truth,observation=ranges,valid=valid)
                recorder.append('localization',t,pose=mapper.pose.copy(),scan_residual_m=mapper.scan_match_residual if mapper.scan_match_residual is not None else np.nan)
            if renderer and tick%50==0:
                renderer.update_scene(data,prefix+'rig_rgbd',scene_option=opt);rgb=renderer.render().copy()
                renderer.enable_depth_rendering();renderer.update_scene(data,prefix+'rig_rgbd',scene_option=opt);depth=renderer.render().copy();renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering();renderer.update_scene(data,prefix+'rig_rgbd',scene_option=opt);seg=renderer.render().copy();renderer.disable_segmentation_rendering()
                instances,classes=semantic_masks(seg,model,manifest)
                recorder.append('camera',t,rgb_truth=rgb,rgb=np.clip(rgb.astype(float)+camera_rng.normal(0,2,rgb.shape),0,255).astype('uint8'),
                                depth_truth=depth,depth=np.maximum(0,depth+camera_rng.normal(0,.005,depth.shape)).astype('float32'),instance=instances,semantic=classes)
                frames.append(Image.fromarray(rgb))
            if tick==round(seconds/model.opt.timestep):break
            mujoco.mj_step(model,data);warnings=np.maximum(warnings,data.warning.number)
            if any(warnings) or not np.isfinite(data.qpos).all():raise PipelineError('PHYSICS_DIVERGED','Flow physics produced warnings/nonfinite state')
    finally:
        recorder.close()
        if renderer:renderer.close()
    result=mapper.report(output) if mapper else controller.report()
    if mapper:
        # Evaluation only, after all control has ended. No state truth is fed back.
        from .dataset import load_stream
        positions=load_stream(output/'data.h5','state')['body_position'][:,model.body('base/base').id,:2]
        actual_travel=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())
        result['actual_travel_m']=actual_travel
        result['passed']=bool(result['passed'] and np.isfinite(actual_travel))
        result['controller_inputs']=['noisy rangefinder readings','noisy wheel angular velocities','noisy IMU gyro',
                                     'known spawn pose','configured map bounds','robot wheel geometry']
        result['ground_truth_usage']='post-rollout scoring only; not localization, scan matching or navigation'
    result.update(flow=flow,simulated_seconds=float(data.time),wall_seconds=time.perf_counter()-started,warnings=warnings.tolist(),streams=inspect(output/'data.h5'))
    if frames:
        frames[0].save(output/'flow.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    write_json(output/'report.json',result)
    return result
