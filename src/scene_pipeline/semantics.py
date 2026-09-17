"""Stable semantic queries over a live composed model; model IDs are not identities."""
import numpy as np
from .validation import aabbs


def snapshot(model,data,manifest):
    low,high=aabbs(model,data);objects={}
    for name,entry in manifest['instances'].items():
        geoms=[i for i in range(model.ngeom) if model.geom(i).name.startswith(name+'/')]
        bodies=[i for i in range(model.nbody) if model.body(i).name.startswith(name+'/')]
        joints={}
        for j in range(model.njnt):
            if model.joint(j).name.startswith(name+'/'):
                q=int(model.jnt_qposadr[j]);v=int(model.jnt_dofadr[j]);kind=int(model.jnt_type[j])
                nq,nv=(7,6) if kind==0 else ((4,3) if kind==1 else (1,1))
                joints[model.joint(j).name]={'qpos':data.qpos[q:q+nq].tolist(),'qvel':data.qvel[v:v+nv].tolist()}
        if not geoms or not bodies:raise ValueError(f'Semantic instance absent from model: {name}')
        objects[name]=dict(class_id=entry['class_id'],category=entry['category'],source_classification=entry.get('source_classification'),position=data.xpos[bodies[0]].tolist(),
                           quaternion_wxyz=data.xquat[bodies[0]].tolist(),bbox=[low[geoms].min(axis=0).tolist(),high[geoms].max(axis=0).tolist()],joints=joints)
    return dict(time=float(data.time),objects=objects)
