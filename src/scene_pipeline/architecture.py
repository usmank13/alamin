"""Exact polygon extrusion shared by all SceneIR-v2 producers, not SVG-specific."""
import xml.etree.ElementTree as ET
import numpy as np
from shapely import constrained_delaunay_triangles, set_precision
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from .assets import vec
from .contracts import PipelineError


def extrude(asset,world,geometry,name,z0,z1,material):
    if geometry.is_empty: return
    triangles=constrained_delaunay_triangles(geometry)
    for i,tri in enumerate(triangles.geoms):
        if tri.area<1e-12: continue
        xy=list(tri.exterior.coords)[:3]
        vertices=[[x,y,z] for z in (z0,z1) for x,y in xy]
        mesh=f'{name}_{i}'
        ET.SubElement(asset,'mesh',name=mesh,vertex=vec(np.array(vertices).ravel()))
        ET.SubElement(world,'geom',name=mesh+'_visual',type='mesh',mesh=mesh,material=material,
                      group='2',contype='0',conaffinity='0',density='0')
        ET.SubElement(world,'geom',name=mesh+'_collision',type='mesh',mesh=mesh,
                      rgba='0 0 0 0',group='3',contype='1',conaffinity='1',density='0')


def compile_architecture(ir,asset,world):
    architecture=ir['architecture']
    doors=[Polygon(o['polygon']) for o in ir['openings'] if o['kind']=='door']
    windows=[Polygon(o['polygon']) for o in ir['openings'] if o['kind']=='window']
    floor=unary_union([Polygon(r['polygon']) for r in ir['rooms']]+doors)
    extrude(asset,world,floor,'floor_reference',-.1,0,'floor_material')
    # Reference wall polygons preserve thickness and oblique orientation. Openings
    # are boolean cuts, not decorative doorway props. Heights are explicit assumptions.
    walls=unary_union([Polygon(w['polygon']) for w in architecture['walls']])
    levels=sorted(set([0.,architecture['height'],architecture['door_height'],
                       architecture['window_sill'],architecture['window_top']]))
    for n,(z0,z1) in enumerate(zip(levels[:-1],levels[1:])):
        if z0>=architecture['height']:continue
        cuts=[]
        if z0<architecture['door_height']:cuts+=doors
        if z0>=architecture['window_sill'] and z1<=architecture['window_top']:cuts+=windows
        layer=walls.difference(unary_union(cuts))
        extrude(asset,world,layer,f'wall_reference_{n}',z0,min(z1,architecture['height']),'wall_material')


def compiled_footprints(model,data,prefix,z=None):
    """Recover footprints from compiled collision meshes, not from cached IR bounds."""
    polygons=[]
    for i in range(model.ngeom):
        name=model.geom(i).name
        if not name.startswith(prefix) or not name.endswith('_collision'):continue
        mesh=int(model.geom_dataid[i]); start=int(model.mesh_vertadr[mesh]); count=int(model.mesh_vertnum[mesh])
        points=model.mesh_vert[start:start+count]@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
        if z is not None and not points[:,2].min()-1e-7<=z<=points[:,2].max()+1e-7:continue
        from shapely.geometry import MultiPoint
        # MuJoCo stores mesh vertices as float32. Remove triangulation seams on a
        # declared 10-micrometre grid before boundary comparisons.
        polygons.append(set_precision(MultiPoint(points[:,:2]).convex_hull,1e-5))
    return unary_union(polygons)
