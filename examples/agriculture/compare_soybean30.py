"""Rebuild paired soybean diagnostics from completed captures; no new simulation."""
import argparse
import html
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image,ImageDraw

from scene_pipeline.contracts import read_json,write_json
from scene_pipeline.inspection import scene_page
from scene_pipeline.weeding_field import analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',type=Path,default=Path('outputs/soybean30_oracle'))
    parser.add_argument('--neural',type=Path,default=Path('outputs/soybean30_neural'))
    parser.add_argument('--output',type=Path,default=Path('outputs/soybean30_comparison'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    summary={};sections=[]
    for mode,root in [('oracle',args.oracle),('neural',args.neural)]:
        run=read_json(root/'report.json');field=analyze(root);scene_page(root)
        summary[mode]=dict(simulation_passed=run['passed'],simulated_seconds=run['simulated_seconds'],
            displacement_m=run['displacement_m'],goal_reached=run['rhizome']['goal_reached'],
            goal_distance_m=run['rhizome']['goal_distance_m'],oracle_predictions=run['rhizome']['oracle_detections'],
            neural_predictions=run['rhizome']['neural_detections'],
            **{key:len(field[key]) for key in ['unique_detected_weeds','unique_selected_weeds',
                'unique_contacted_weeds','unique_ordered_weed_contacts','unique_contacted_crops']},
            arms={name:{k:len(a[k]) for k in ['detected_weeds','selected_weeds','ordered_weed_contacts','contacted_crops']}
                  for name,a in field['arms'].items()},
            robot_urdf_sha256=read_json(root/'rig.json')['profile']['provenance']['urdf_sha256'])
        manifest=read_json(root/'manifest.json');bounds=np.asarray(manifest['terrain']['bounds_m'])
        plants=list(manifest['instances'].values())
        crops=[p for p in plants if p['category']=='crop']
        summary[mode]['field']=dict(crops=len(crops),weeds=sum(p['category']=='weed' for p in plants),
            crop_growth_range=[min(p['growth_stage'] for p in crops),max(p['growth_stage'] for p in crops)])
        picture=Image.new('RGB',(800,650),'#f4efe4');draw=ImageDraw.Draw(picture)
        def pixel(p):
            x,y=(np.asarray(p[:2])-bounds[0])/(bounds[1]-bounds[0]);return (30+float(x)*740,620-float(y)*570)
        for name,plant in manifest['instances'].items():
            x,y=pixel(plant['position_m']);crop=plant['category']=='crop'
            draw.ellipse((x-3,y-3,x+3,y+3),fill='#45844a' if crop else '#db931e')
            if name in field['unique_ordered_weed_contacts']:draw.ellipse((x-8,y-8,x+8,y+8),outline='#1d50de',width=3)
            if name in field['unique_contacted_crops']:draw.ellipse((x-8,y-8,x+8,y+8),outline='#cf1d34',width=3)
        with h5py.File(root/'data.h5') as data:
            draw.line([pixel(p) for p in data['state/chassis_position'][::10]],fill='#333333',width=3)
            for name in field['arms']:
                draw.line([pixel(p) for p in data['weeding_tool_'+name+'/position_world'][::25]],fill='#4c8bcc',width=1)
        draw.text((20,12),mode+': green=crops; orange=weeds; black=chassis; blue=tool paths / ordered weed contact; red=crop contact',fill='#111111')
        picture.save(args.output/(mode+'_map.png'))
        rel=html.escape(os.path.relpath(root.resolve(),args.output.resolve()))
        sections.append(f'<section><h2>{mode.title()}</h2><p><a href="{rel}/index.html">Run inspection</a> · '
            f'<a href="{rel}/field_weeding.html">Per-arm weeding diagnostics</a></p>'
            f'<video controls preload="metadata" poster="{rel}/replay.png" src="{rel}/replay.mp4"></video><img src="{mode}_map.png"></section>')
    assert summary['neural']['oracle_predictions']==0,'Neural comparison includes oracle observations'
    with h5py.File(args.oracle/'data.h5') as a,h5py.File(args.neural/'data.h5') as b:
        ma=json.loads(a.attrs['metadata_json']);mb=json.loads(b.attrs['metadata_json'])
        assert ma['source_sha256']==mb['source_sha256'],'Runs use different scenes'
        assert ma['rig']['spawn']==mb['rig']['spawn'] and ma['rig']['cameras']==mb['rig']['cameras'],'Different initial pose or camera rig'
        summary['source_scene_sha256']=ma['source_sha256']
    write_json(args.output/'comparison.json',summary)
    metrics=[('Simulation completed','simulation_passed'),('Distance travelled (m)','displacement_m'),
             ('Detected weeds (unique truth associations)','unique_detected_weeds'),('Selected weeds (unique)','unique_selected_weeds'),
             ('Ordered weed contacts (unique)','unique_ordered_weed_contacts'),('Any weed contacts (unique)','unique_contacted_weeds'),
             ('Crop proxy contacts (unique)','unique_contacted_crops')]
    rows=[]
    for title,key in metrics:
        values=[summary[m][key] for m in ['oracle','neural']]
        cells=[f'{v:.3f}' if isinstance(v,float) else str(v) for v in values]
        rows.append('<tr><th>'+title+'</th>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Soybean 30-inch comparison</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#253036}table{border-collapse:collapse}'
        'th,td{text-align:left;padding:10px 20px;border-bottom:1px solid #ddd}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}'
        'video,img{width:100%}button{padding:10px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'
        '<h1>30-inch soybean rows: oracle vs neural</h1><p>'
        f'{summary["oracle"]["field"]["crops"]} soybeans, {summary["oracle"]["field"]["weeds"]} scattered weeds; '
        f'crop growth values {summary["oracle"]["field"]["crop_growth_range"][0]:.2f}–{summary["oracle"]["field"]["crop_growth_range"][1]:.2f}. '
        'Inicio RAPTOR_30 layout. '
        'Same field, initial pose and requested route. Actual trajectories can diverge as arm forces differ.</p>'
        '<p>Oracle: visible-plant ground-truth roots/classes and terrain row profiles. Neural: packaged multicrop RGB/depth predictions; no oracle fallback.</p>'
        '<table><tr><th>Metric</th><th>Oracle</th><th>Neural</th></tr>'+''.join(rows)+'</table>'
        '<p>Contacts use swept tool boxes and fixed near-stem capsule proxies. No plant removal or biological damage is modeled. '
        'The path covers part of the field; associations within 6 cm can be ambiguous in dense crops.</p>'
        '<p><a href="comparison.json">Comparison data</a> · <button onclick="document.querySelectorAll(\'video\').forEach(v=>{v.currentTime=0;v.play()})">Play both from start</button></p>'
        '<div class="pair">'+''.join(sections)+'</div>')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
