"""Reference producer and source-versus-compiled reconstruction experiment."""
from pathlib import Path
import html
import math

import mujoco
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import Polygon
from shapely import set_precision
from shapely.ops import unary_union

from .architecture import compiled_footprints
from .contracts import PipelineError, digest, validate_ir, write_json
from .reference_layout import metric_geometry


def reference_ir(reference, *, area_m2=None, metres_per_unit=None):
    g,scale=metric_geometry(reference,area_m2=area_m2,metres_per_unit=metres_per_unit)
    rooms=[dict(id=r['id'],polygon=r['polygon'],height=2.8,role=' '.join(r['labels'][1:])) for r in g['rooms']]
    openings=[]
    for o in g['openings']:
        polygon=Polygon(o['polygon']); center=polygon.centroid
        openings.append(dict(id=o['id'],kind='door' if 'Door' in o['labels'] else 'window',
                             polygon=o['polygon'],position=[center.x,center.y,0.],
                             width=math.sqrt(polygon.area),height=2.1 if 'Door' in o['labels'] else 1.3,rooms=[]))
    # Width/adjacency are measured from the opening geometry, not guessed semantic links.
    for o in openings:
        rectangle=Polygon(o['polygon']).minimum_rotated_rectangle
        corners=list(rectangle.exterior.coords)
        o['width']=max(math.dist(a,b) for a,b in zip(corners[:-1],corners[1:]))
        o['rooms']=[r['id'] for r in rooms if Polygon(r['polygon']).distance(Polygon(o['polygon']))<.02]
        if len(o['rooms'])==1:o['rooms'].append('exterior')
    return validate_ir(dict(schema_version=2,meta=dict(prompt='Reconstruct annotated source architecture',seed=0,
        area_m2=unary_union([Polygon(r['polygon']) for r in rooms]).area,units='metres',up='Z',
        origin=scale['origin'],pipeline_version='scene-grounding-v1'),rooms=rooms,openings=openings,
        objects=[],robots=[],architecture=dict(walls=[dict(id=w['id'],polygon=w['polygon']) for w in g['walls']],
        height=2.8,door_height=2.1,window_sill=.9,window_top=2.2),
        provenance=dict(generator='reference',reference_sha256=digest(reference),source=reference['source'],
        scale=scale,height_basis='explicit_engineering_defaults_not_recovered_from_2D',
        annotation_issues=reference['issues'],omitted_fixtures=[dict(id=f['id'],labels=f['labels'],reason='2D annotation only; no asset or height inferred') for f in g['fixtures']])) )


def compare_compiled(reference,model,data,*,area_m2=None,metres_per_unit=None):
    """Expected geometry comes from source; observed geometry comes from MuJoCo meshes."""
    g,scale=metric_geometry(reference,area_m2=area_m2,metres_per_unit=metres_per_unit)
    doors=[Polygon(o['polygon']) for o in g['openings'] if 'Door' in o['labels']]
    cuts=unary_union([Polygon(o['polygon']) for o in g['openings']])
    expected_floor=set_precision(unary_union([Polygon(r['polygon']) for r in g['rooms']]+doors),1e-5)
    expected_walls=set_precision(unary_union([Polygon(w['polygon']) for w in g['walls']]).difference(cuts),1e-5)
    observed_floor=compiled_footprints(model,data,'floor_reference')
    observed_walls=compiled_footprints(model,data,'wall_reference',z=1.2)
    def metrics(a,b):
        return dict(iou=a.intersection(b).area/a.union(b).area if a.union(b).area else 1.,
                    symmetric_difference_m2=a.symmetric_difference(b).area,
                    boundary_hausdorff_m=a.boundary.hausdorff_distance(b.boundary) if not a.is_empty and not b.is_empty else None)
    floor=metrics(expected_floor,observed_floor);walls=metrics(expected_walls,observed_walls)
    # Numerical reconstruction tolerance, not an empirical plausibility target.
    tolerance=1e-4
    passed=all(m['symmetric_difference_m2']<=tolerance*max(1,g.area) and m['boundary_hausdorff_m'] is not None and m['boundary_hausdorff_m']<=.001 for m,g in [(floor,expected_floor),(walls,expected_walls)])
    report=dict(schema_version=1,passed=passed,scope='architecture_reconstruction_only',floor=floor,walls_at_1_2m=walls,
                relative_area_tolerance=tolerance,boundary_tolerance_m=.001,mesh_seam_precision_m=1e-5,scale=scale,source=reference['source'],
                annotation_issues=reference['issues'],fixture_annotations_not_reconstructed=len(g['fixtures']),
                full_scene_validated=False,distribution_matching_proven=False)
    return report,(expected_floor,expected_walls,observed_floor,observed_walls),g


def draw_overlay(shapes,g,path):
    a,b,c,d=shapes;lowx,lowy,highx,highy=a.union(b).bounds
    image=Image.new('RGB',(1440,620),'#fafafa');draw=ImageDraw.Draw(image)
    scale=min(640/max(highx-lowx,.01),500/max(highy-lowy,.01))
    def paint(geometry,offset,fill):
        if geometry.is_empty:return
        polygons=[geometry] if geometry.geom_type=='Polygon' else geometry.geoms
        for poly in polygons:
            points=[(offset+30+(x-lowx)*scale,570-(y-lowy)*scale) for x,y in poly.exterior.coords]
            draw.polygon(points,fill=fill)
            for ring in poly.interiors:
                draw.polygon([(offset+30+(x-lowx)*scale,570-(y-lowy)*scale) for x,y in ring.coords],fill='#fafafa')
    for offset,floor,walls,title in [(0,a,b,'Source geometry (explicit area normalization)'),(720,c,d,'Compiled MuJoCo collision meshes / z=1.2 m')]:
        draw.text((offset+25,15),title,fill='black');paint(floor,offset,'#e3e8eb');paint(walls,offset,'#384858')
    for f in g['fixtures']:paint(Polygon(f['polygon']),0,'#f0a34a')
    draw.text((25,595),'Orange: source fixture annotations, NOT reconstructed assets. All heights are defaults.',fill='black')
    image.save(path)


def reconstruct(reference,output,*,area_m2=None,metres_per_unit=None,render=True):
    output=Path(output)
    if output.exists():raise PipelineError('OUTPUT_EXISTS','Refusing to overwrite reconstruction')
    ir=reference_ir(reference,area_m2=area_m2,metres_per_unit=metres_per_unit)
    write_json(output/'ir.json',ir);write_json(output/'reference.json',reference)
    from .compiler import compile_scene
    _,model,data=compile_scene(ir,output)
    report,shapes,g=compare_compiled(reference,model,data,area_m2=area_m2,metres_per_unit=metres_per_unit)
    write_json(output/'reconstruction.json',report)
    draw_overlay(shapes,g,output/'reconstruction.png')
    if render:
        from .render import preview
        preview(output,cutaway=False)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Floorplan reconstruction</title>'
        '<h1>Reference-to-simulation reconstruction</h1><p>Architecture only; not a robot-ready furnished scene. '
        'Source scale is not verified. Orange footprints are omitted fixtures.</p>'
        '<img width="100%" src="reconstruction.png"><img src="preview.png">'
        '<p><a href="reconstruction.json">Geometric comparison</a> | <a href="ir.json">Common SceneIR</a> | '
        '<a href="render/cycles.png">Cycles render (if requested)</a></p><pre>'+html.escape(str(report))+'</pre>')
    return report
