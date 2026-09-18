"""Reusable G1 families and a source-aware native MJCF adapter."""
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim_harness.scene import _read_spec
from .contracts import PipelineError, digest, write_json, read_json
from .registry import POLICY, MATERIALS, fingerprint, lookup

ADAPTER_VERSION = 'native-mjcf-v9'  # v9: finish-named placeholder materials, rebound to scene PBR materials by the compiler


def vec(values):
    return ' '.join(f'{float(x):.12g}' for x in values)


def pair(body, name, size, pos, density=600., kind='box', material='panel'):
    common = dict(type=kind, size=vec(size), pos=vec(pos))
    ET.SubElement(body, 'geom', name=name+'_visual', **common, group='2', contype='0', conaffinity='0',
                  density='0', material=material)
    ET.SubElement(body, 'geom', name=name+'_collision', **common, group='3', rgba='.8 .3 .1 0', density=str(density))


def template(family, dimensions, finish='wood'):
    w, d, h = dimensions
    t, gap = POLICY['panel_thickness_m'], POLICY['gap_m']
    if min(dimensions) <= 0 or (family in ('cabinet', 'drawer') and min(dimensions) < 8*t):
        raise PipelineError('TEMPLATE_DOMAIN', 'Dimensions outside supported family domain')
    xml = ET.Element('mujoco', model=family)
    ET.SubElement(xml, 'compiler', angle='radian', inertiagrouprange='3 3')
    option = ET.SubElement(xml, 'option', timestep='.002', integrator='implicitfast')
    ET.SubElement(option, 'flag', filterparent='disable')
    if finish not in MATERIALS:
        raise PipelineError('TEMPLATE_DOMAIN', f'Unknown finish {finish}')
    asset = ET.SubElement(xml, 'asset')
    for name in sorted({finish, 'stainless'}):
        # Flat placeholder named by finish. The scene compiler rebinds every `finish_*` geom to the scene-level
        # PBR material of the same finish when one is realized, so textures are stored once per scene.
        m = MATERIALS[name]
        ET.SubElement(asset, 'material', name=f'finish_{name}', rgba=vec(m['rgba']), shininess=str(m['shininess']), reflectance=str(m['reflectance']))
    world = ET.SubElement(xml, 'worldbody')
    root = ET.SubElement(world, 'body', name='root')
    affordances, supports = [], []
    if family in ('jar','bottle','tray'):
        ET.SubElement(root,'freejoint',name='free')
        if family=='tray':
            shell=.004
            pair(root,'base',[w/2,d/2,shell/2],[0,0,shell/2],density=950)
            for i,x in enumerate((-1,1)):
                pair(root,f'side{i}',[shell/2,d/2,h/2],[x*(w-shell)/2,0,h/2],density=950)
            for i,y in enumerate((-1,1)):
                pair(root,f'end{i}',[(w-2*shell)/2,shell/2,h/2],[0,y*(d-shell)/2,h/2],density=950)
        else:
            bodyh=h*(.75 if family=='bottle' else .9)
            pair(root,'body',[w/2,bodyh/2],[0,0,bodyh/2],density=180,kind='cylinder')
            pair(root,'cap',[w*(.22 if family=='bottle' else .5),(h-bodyh)/2],[0,0,(h+bodyh)/2],density=950,kind='cylinder',material='metal')
    elif family == 'container':
        ET.SubElement(root, 'freejoint', name='free')
        pair(root, 'container', [w/2, d/2, h/2], [0, 0, h/2], density=120)
    elif family == 'box':
        # Static grey box at verified dimensions: the brief's fallback for unmodeled categories.
        pair(root, 'box', [w/2, d/2, h/2], [0, 0, h/2], POLICY['density_kg_m3']['panel'])
        supports.append(dict(center=[0, 0, h], size=[w-.08, d-.08]))
    elif family == 'door':
        moving = ET.SubElement(root, 'body', name='moving', pos=vec([-w/2, 0, 0]))
        ET.SubElement(moving, 'joint', name='door', type='hinge', axis='0 0 1', range=vec([0, math.pi/2]), damping='2', frictionloss='.2')
        pair(moving, 'panel', [w/2, d/2, h/2], [w/2, 0, h/2])
        pair(moving, 'handle', [.04, .018, .012], [w-.10, -.045, 1.], density=7800, material='metal')
        affordances = [dict(joint='door', role='door', closed=0., open=math.pi/2, requires={}, point=[w-.1, -.045, 1.], body='moving')]
    elif family in ('table', 'shelf'):
        density = POLICY['density_kg_m3']['panel']
        heights = [h-t/2] if family == 'table' else [t/2, h/3, 2*h/3, h-t/2]
        for i, z in enumerate(heights):
            pair(root, f'surface{i}', [w/2, d/2, t/2], [0, 0, z], density)
            # Lower shelf levels must clear the 5 cm corner uprights as well as
            # avoid overhang. A tabletop sits above its legs; shelf levels do not.
            inset=.12 if family=='shelf' else .08
            supports.append(dict(center=[0, 0, z+t/2], size=[w-inset, d-inset]))
        for i, (x, y) in enumerate([(x,y) for x in (-1,1) for y in (-1,1)]):
            pair(root, f'leg{i}', [.025, .025, (h-t)/2], [x*(w/2-.025), y*(d/2-.025), (h-t)/2], density)
    elif family in ('cabinet', 'drawer'):
        for i, x in enumerate((-1, 1)):
            pair(root, f'side{i}', [t/2, d/2, h/2], [x*(w-t)/2, 0, h/2])
        for i, z in enumerate((t/2, h-t/2)):
            pair(root, f'horizontal{i}', [(w-2*t)/2, d/2, t/2], [0, 0, z])
        pair(root, 'back', [(w-2*t)/2, t/2, (h-2*t)/2], [0, (d-t)/2, h/2])
        supports.append(dict(center=[0, 0, h], size=[w-.08, d-.08]))
        moving = ET.SubElement(root, 'body', name='moving')
        innerw, innerh = w-2*t-2*gap, h-2*t-2*gap
        if family == 'cabinet':
            moving.set('pos', vec([-w/2+t+gap, -d/2, 0]))
            ET.SubElement(moving, 'joint', name='door', type='hinge', axis='0 0 -1', range=vec([0, math.pi/2]), damping='2', frictionloss='.2')
            pair(moving, 'panel', [innerw/2, t/2, innerh/2], [innerw/2, t/2, h/2])
            pair(moving, 'handle', [.012, .016, .05], [innerw-.065, -.028, h*.6], density=7800, material='metal')
            affordances = [dict(joint='door', role='door', closed=0., open=math.pi/2, requires={}, body='moving', point=[innerw-.065,-.028,h*.6])]
        else:
            travel = d*.65
            ET.SubElement(moving, 'joint', name='drawer', type='slide', axis='0 -1 0', range=vec([0,travel]), damping='4', frictionloss='.5')
            pair(moving, 'tray', [innerw/2, (d-t-2*gap)/2, t/2], [0, -t/2, t+gap+t/2])
            pair(moving, 'front', [innerw/2, t/2, (innerh-t)/2], [0, -(d-t)/2, h/2+t/2])
            pair(moving, 'handle', [.06,.016,.012], [0,-d/2-.028,h*.65], density=7800, material='metal')
            affordances = [dict(joint='drawer', role='drawer', closed=0., open=travel, requires={}, body='moving', point=[0,-d/2-.028,h*.65])]
    else:
        raise PipelineError('TEMPLATE_DOMAIN', f'Unknown family {family}')
    for geom in xml.iter('geom'):
        if geom.get('material') == 'panel': geom.set('material', f'finish_{finish}')
        elif geom.get('material') == 'metal': geom.set('material', 'finish_stainless')
    return mujoco.MjSpec.from_string(ET.tostring(xml, encoding='unicode')), affordances, supports


def bounds(model, data, collision=False):
    points = []
    for i in range(model.ngeom):
        contact = bool(model.geom_contype[i] or model.geom_conaffinity[i])
        if contact != collision or model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not collision and model.geom_rgba[i,3] == 0:
            continue
        rotation = data.geom_xmat[i].reshape(3,3)
        if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(model.geom_dataid[i]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
            vertices=model.mesh_vert[start:start+count]@rotation.T+data.geom_xpos[i]
            points.extend([vertices.min(axis=0),vertices.max(axis=0)])
            continue
        c = data.geom_xpos[i]+rotation @ model.geom_aabb[i,:3]
        half = abs(rotation) @ model.geom_aabb[i,3:]
        points.extend([c-half,c+half])
    if not points:
        raise PipelineError('EMPTY_GEOMETRY', 'No visible/collision geometry')
    return np.min(points,axis=0), np.max(points,axis=0)


def scale_native(source, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise PipelineError('SCALE', 'Scale must be positive and finite')
    if any(len(getattr(source,k)) for k in ('keys','equalities','tendons','flexes','hfields','skins','frames')):
        raise PipelineError('UNSUPPORTED_SOURCE', 'Native adapter supports scalar-joint rigid fixtures only')
    spec = source.copy() if factor == 1 else mujoco.MjSpec.from_string(source.to_xml(), assets=source.assets)
    for b in spec.bodies:
        b.pos *= factor
        if b.explicitinertial:
            b.ipos *= factor
            b.mass *= factor**3
            b.inertia *= factor**5
            b.fullinertia *= factor**5
    for m in spec.meshes:
        m.scale *= factor
    for g in [*spec.geoms,*spec.sites]:
        g.pos *= factor
        g.size *= factor
        g.fromto *= factor
        if hasattr(g,'mass'):
            g.mass *= factor**3
            g.margin *= factor
            g.gap *= factor
    for j in spec.joints:
        if j.type not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):
            raise PipelineError('UNSUPPORTED_JOINT','Native adapter requires scalar joints')
        j.pos *= factor
        if j.type == mujoco.mjtJoint.mjJNT_SLIDE:
            j.range *= factor
            j.ref *= factor
            j.springref *= factor
            j.margin *= factor
    return spec


def instantiate(category, destination, *, vendor=None, scale=1., family=None, dimensions=None, dimension_basis=None,asset_ref=None,generated_request=None):
    if asset_ref is not None:
        from .asset_library import materialize
        folder,package=materialize(asset_ref,destination)
        if package['category']!=category or dimensions is not None or family is not None or scale!=1:
            raise PipelineError('ASSET_IDENTITY','Retrieved references require matching categories and unmodified native geometry')
        return folder,package
    from .resources import vendor_path
    vendor=vendor if vendor is not None else vendor_path('robocasa_native')
    try:
        if generated_request is not None:
            from .generated import config_from_request
            config = config_from_request(generated_request)
        else:config = lookup(category)
    except PipelineError:
        # Open-vocabulary category: resolved dimensions are mandatory; the family defaults to the grey box.
        if dimensions is None or dimension_basis is None:
            raise
        family = family or 'box'
        config = dict(route='G1', family=family, dimensions=list(dimensions), articulated=family in ('cabinet', 'drawer'), finish='paint',
                      open_vocabulary=True, **({'placement': 'freestanding'} if family in ('box', 'table') else {}))
    source_hashes = {}
    if config['route'] == 'G1':
        if scale != 1:
            raise PipelineError('TEMPLATE_DOMAIN', 'Use approved template dimensions; arbitrary template scaling not enabled')
        if dimensions is not None:
            if dimension_basis is None:
                raise PipelineError('UNBOUND_MEASUREMENT', 'Template dimensions require a resolved basis')
            config = {**config, 'dimensions': list(dimensions)}
        spec, affordances, supports = template(config['family'],config['dimensions'],config.get('finish','wood'))
        provenance = dict(kind='sourced_template' if config.get('open_vocabulary') else 'engineering_default', policy=POLICY['version'], calibrated=False)
        if dimension_basis is not None: provenance['dimension_basis'] = dimension_basis
        dimensions = config['dimensions']
    elif config['route'] == 'G6':
        if scale != 1 or dimensions is not None:
            raise PipelineError('TEMPLATE_DOMAIN', 'Generated decor is sized from its registry band; explicit scale or dimensions are not accepted')
        from .generated import package as generated_package
        spec, provenance = generated_package(category, config)
        affordances, supports = [], []
        source_hashes = provenance['hashes']
        model = spec.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        lo, hi = bounds(model, data); dimensions = (hi-lo).tolist()
    else:
        path = Path(vendor)/'fixtures'/config['source']/'model.xml'
        if not path.exists():
            raise PipelineError('MISSING_SOURCE', f'Download required fixture first: {path}')
        for file in sorted(path.parent.rglob('*')):
            if file.is_file():
                source_hashes[str(file.relative_to(path.parent))] = hashlib.sha256(file.read_bytes()).hexdigest()
        spec = scale_native(_read_spec(path),scale)
        spec.compiler.inertiagrouprange = [0,0]
        # Explicit hollow-appliance physical proxy. Never relabel this as measured
        # source material data; the immutable source remains available for comparison.
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.density=POLICY['retrieved_shell_proxy_kg_m3']
                # Source debug colliders are translucent red. Keep the physical
                # geometry, but do not render it over the native textured meshes.
                geom.rgba=[*geom.rgba[:3],0.]
            else:
                # Visual meshes must not carry mass: the inertia group range does not
                # survive the MJZ round trip, so the scene compiler would otherwise weigh
                # them at the MuJoCo default 1000 kg/m3 (a 3.7 t fridge).
                geom.density=0.
        # Joint/contact conventions are otherwise unchanged.
        model = spec.compile()
        affordances, supports = [], []
        for j in range(model.njnt):
            if not model.jnt_limited[j]:
                continue
            name = model.joint(j).name
            requires = {}
            if category=='fridge' and name=='fridge_drawer0_joint':
                requires={'fridge_door_joint': float(model.jnt_range[model.joint('fridge_door_joint').id,1])}
            if category=='dishwasher' and name.startswith('rack'):
                requires={'door_joint': float(model.jnt_range[model.joint('door_joint').id,1])}
            affordances.append(dict(joint=name, role='button' if 'Button' in name or 'button' in name else 'access',
                                    closed=0.,open=float(model.jnt_range[j,np.argmax(abs(model.jnt_range[j]))]), requires=requires,
                                    body=model.body(model.jnt_bodyid[j]).name, point=None))
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        lo,hi=bounds(model,data)
        dimensions=(hi-lo).tolist()
        provenance=dict(kind='retrieved_source', source=config['source'], hashes=source_hashes,
                        license='CC BY 4.0', attribution='RoboCasa Team / Lightwheel', scale=scale,
                        physical_policy='collision-only effective shell density from engineering registry',
                        visualization_policy='native visual meshes; collision debug overlays hidden',
                        physical_profile=POLICY['version'], effective_density_kg_m3=POLICY['retrieved_shell_proxy_kg_m3'], calibrated=False)
    model=spec.compile()
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    lo,hi=bounds(model,data)
    key=digest(dict(category=category,config=config,scale=scale,registry=fingerprint(),source=source_hashes,adapter=ADAPTER_VERSION))
    folder=Path(destination)/key
    package=dict(schema_version=1,key=key,category=category,route=config['route'], dimensions=(hi-lo).tolist(),nominal_dimensions=dimensions,
                 bounds=[lo.tolist(),hi.tolist()],model='asset.mjz',affordances=affordances,supports=supports,
                 layout={k:config[k] for k in ('placement','dynamic','mount_height_m','support_height_m') if k in config},
                 provenance=provenance,capabilities=dict(uniform_scale=config['route']=='G3',independent_dimensions=False,
                                                        semantic_state=True,preserve_handle_size=False),
                 state='candidate',validation=None,adapter_version=ADAPTER_VERSION)
    from .provenance import classify
    package['source_classification']=classify(package)
    if not folder.exists():
        folder.mkdir(parents=True)
        spec.add_text(name='provenance',data=str(provenance))
        spec.to_zip(str(folder/'asset.mjz'))
        write_json(folder/'asset.json',package)
    else:
        package=read_json(folder/'asset.json')
    return folder,package


def scale_for_constraints(native, constraints):
    """Intersect admissible uniform scales; never anisotropically distort an asset."""
    low,high=0.,float('inf')
    for axis,(minimum,maximum) in constraints.items():
        index={'width':0,'depth':1,'height':2}[axis]
        low=max(low,minimum/native[index]); high=min(high,maximum/native[index])
    if low<=0 or low>high or not np.isfinite(low):
        raise PipelineError('INCOMPATIBLE_PROPORTIONS','No uniform scale satisfies dimension constraints')
    return min(max(1.,low),high)
