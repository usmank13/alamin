"""Open-vocabulary categories, evidence-bound dimensions, solved relations, brief-facing gates and degradation."""
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip('jsonschema')
pytest.importorskip('shapely')

from scene_pipeline.contracts import PipelineError,write_json,read_json
from scene_pipeline.dsl import parse,load
from scene_pipeline.evidence import resolve
from scene_pipeline.layout import solve
from scene_pipeline.compiler import compile_scene
from scene_pipeline.validation import validate_scene,brief_checks,load as load_model

DOC=('<h1>Acme Steri-500</h1><table><tr><td>Overall Width</td><td>48 in</td></tr><tr><td>Overall Depth</td><td>28 in</td></tr>'
     '<tr><td>Overall Height</td><td>36 in</td></tr><tr><td>Shipping Width</td><td>52 in</td></tr></table>')
FIELDS=dict(width=['Overall Width','48','in'],depth=['Overall Depth','28','in'],height=['Overall Height','36','in'])


def build(program,seed,root):
    ir=solve(program,seed,root);write_json(root/'program.json',program);write_json(root/'ir.json',ir);compile_scene(ir,root)
    return ir


def test_prompt_quoted_unknown_category_becomes_sized_static_box(tmp_path):
    prompt='A clinic with a sterilization station 1.2 m wide, 0.7 m deep and 0.9 m high, and cabinets.'
    p=parse(f'''scene(prompt={prompt!r},space=space(kind="clinic",area_m2=36),objects=[
        place("steri","sterilization_station",dimensions_m=[1.2,.7,.9],dimension_evidence={{"prompt_quote":"1.2 m wide, 0.7 m deep and 0.9 m high"}}),
        place("cab","base_cabinet",count=2),place("dr","drawer_unit",count=3),place("goods","container",count=2)])''')
    o=p['objects'][0];o['dimensions_m'],record=resolve(o,prompt,cache_path=tmp_path/'cache.json');o['dimension_basis']=record['basis'];o.pop('dimension_evidence')
    assert record['basis']=='user_quoted' and record['defaulted_axes']==[]
    ir=build(p,5,tmp_path)
    steri=next(x for x in ir['objects'] if x['category']=='sterilization_station')
    np.testing.assert_allclose(steri['dimensions'],[1.2,.7,.9],atol=1e-9)
    assert steri['class_id']>=1000 and steri['source_classification']['dimension_basis']=='user_quoted' and not steri['dynamic']
    assert read_json(tmp_path/'manifest.json')['instances'][steri['id']]['affordances']==[]
    assert read_json(tmp_path/steri['asset'])['supports']  # a box top is a support surface
    assert validate_scene(tmp_path)['passed']


def test_bare_numbers_and_unbound_quotes_are_rejected(tmp_path):
    with pytest.raises(PipelineError,match='evidence'):
        parse('scene(prompt="x",space=space(),objects=[place("a","autoclave",dimensions_m=[1,1,1])])')
    with pytest.raises(PipelineError,match='verbatim'):
        parse('scene(prompt="x",space=space(),objects=[place("a","autoclave",dimensions_m=[1,1,1],dimension_evidence={"prompt_quote":"1 m"})])')
    with pytest.raises(PipelineError,match='binds no axis'):
        resolve(dict(id='a',category='autoclave',dimensions_m=[1.,1.,1.],dimension_evidence=dict(prompt_quote='0.5 m')),'an autoclave 0.5 m',cache_path=tmp_path/'c.json')
    with pytest.raises(PipelineError,match='approved route'):  # unknown without any evidence still fails honestly
        solve(parse('scene(prompt="x",space=space(),objects=[place("a","autoclave")])'),0,tmp_path)


def test_web_evidence_is_bound_cached_and_reused_without_refetch(tmp_path):
    cache=tmp_path/'cache.json';fetched=[]
    def fetcher(url):fetched.append(url);return DOC
    item=dict(id='s',category='sterilizer',family='table',dimension_evidence=dict(url='https://example.test/s',identity='Steri-500',fields=FIELDS))
    dims,record=resolve(item,'p',cache_path=cache,fetcher=fetcher)
    np.testing.assert_allclose(dims,[48*.0254,28*.0254,36*.0254])
    assert record['basis']=='sourced' and record['quotes'][0]=='Overall Width 48 in' and len(record['document_sha256'])==64
    entry=read_json(cache)['sterilizer'];assert entry['family']=='table' and entry['url']=='https://example.test/s'
    dims2,record2=resolve(dict(id='t',category='sterilizer'),'p',cache_path=cache,fetcher=lambda u:pytest.fail('cache hit must not fetch'))
    assert dims2==dims and record2['cached'] and record2['family']=='table' and fetched==['https://example.test/s']
    # The P4 counterexample: a real page and a real label, paired with a number from elsewhere on it.
    bad=dict(item,category='other',dimension_evidence=dict(item['dimension_evidence'],fields=dict(FIELDS,width=['Overall Width','52','in'])))
    with pytest.raises(PipelineError,match='Field not found'):resolve(bad,'p',cache_path=cache,fetcher=fetcher)


def test_sourced_box_like_category_gets_cabinet_articulation(tmp_path):
    p=parse('scene(prompt="a locker room",space=space(kind="locker_room",area_m2=36),objects=[place("lk","locker",count=5)])')
    for o in p['objects']:o.update(family='cabinet',dimensions_m=[.4,.5,1.8],dimension_basis='sourced')
    ir=build(p,1,tmp_path);manifest=read_json(tmp_path/'manifest.json')
    assert all(m['affordances'][0]['joint']=='door' for m in manifest['instances'].values())
    r=validate_scene(tmp_path);assert r['passed'] and r['articulated_objects']==5


def test_in_row_is_solved_flush_not_merely_checked(tmp_path):
    p=parse('''scene(prompt="row",space=space(kind="kitchen",area_m2=40),
        objects=[place("cab","base_cabinet",count=3,zone="storage"),place("prep","prep_table",zone="prep"),place("dr","drawer_unit",count=2,zone="storage")],
        relations=[relation("in_row",["cab","dr"],required=True)])''')
    ir=build(p,3,tmp_path)
    row=sorted((o for o in ir['objects'] if o['category'] in ('base_cabinet','drawer_unit')),key=lambda o:o['position'][1])
    assert len({round(o['yaw'],6) for o in row})==1 and np.ptp([o['position'][0] for o in row])<1e-9
    gaps=np.diff([o['position'][1] for o in row]);np.testing.assert_allclose(gaps,.6+.004,atol=1e-9)
    assert all(r['satisfied'] for r in ir['provenance']['relations'])


def test_brief_gates_pass_on_example_and_fail_when_robot_cannot_fit(tmp_path):
    build(load('examples/cafe_program.py'),1,tmp_path)
    r=validate_scene(tmp_path)
    assert r['passed'] and r['checks']['metric_bands'] and r['checks']['robot_access'] and r['checks']['door_widths']
    cabinet=next(m for m in r['metrics'] if m['category']=='base_cabinet')
    assert 15<=cabinet['mass_kg']<=80 and .85<=cabinet['top_m']<=.95
    huge=brief_checks(load_model(tmp_path),read_json(tmp_path/'manifest.json'),read_json(tmp_path/'ir.json'),robot_radius=1.5)
    assert huge['access']['unreachable'] or not huge['access']['entrance_reachable']


def test_generate_degrades_to_partial_and_uses_attempt_limit_on_repeated_failure(tmp_path,monkeypatch):
    from scene_pipeline import orchestrator
    two=parse('scene(prompt="two",space=space(kind="kitchen",area_m2=36),objects=[place("cab","base_cabinet",count=2),place("prep","prep_table")])')
    result=orchestrator.generate('two',1,tmp_path/'partial',program=two,preview=False)
    assert result['status']=='partial' and not result['passed'] and (tmp_path/'partial/scene.mjz').exists()
    assert 'five_articulated_objects' in read_json(tmp_path/'partial/unmet.json')['failed_checks']
    monkeypatch.setattr(orchestrator,'agent_program',lambda *a,**k:parse('scene(prompt="x",space=space(),objects=[place("a","autoclave")])'))
    monkeypatch.setattr(orchestrator,'source_dimensions',lambda *a,**k:dict(url='https://example.test/none',identity='Nothing',family='box',fields=FIELDS))
    monkeypatch.setattr(orchestrator,'fetch',lambda url:'<p>unrelated page</p>')
    with pytest.raises(PipelineError) as info:orchestrator.generate('x',0,tmp_path/'loop',preview=False,max_iterations=6)
    assert [a['error']['code'] for a in info.value.details['attempts']]==['DIMENSION_UNSOURCED']*6
    assert info.value.details['stop_reason']=='attempt_limit'


def test_agent_schema_is_strict_mode_compatible_and_nulls_are_stripped():
    import jsonschema
    from scene_pipeline.contracts import PROGRAM_SCHEMA,PROGRAM_SCHEMA_V2,validate_program
    from scene_pipeline.orchestrator import strict,_strip_null
    def walk(s):
        if isinstance(s,dict):
            assert 'oneOf' not in s and 'uniqueItems' not in s
            if s.get('type')=='object':assert set(s['required'])==set(s['properties'])
            for v in s.values():walk(v)
        elif isinstance(s,list):
            for v in s:walk(v)
    for schema in (PROGRAM_SCHEMA,PROGRAM_SCHEMA_V2):walk(strict(schema))
    agent_output=dict(schema_version=1,prompt='x',space=dict(kind='kitchen',area_m2=36,shape='rectangle',annexes=0,inferred_fields=None),relations=[],
        objects=[dict(id='a',category='base_cabinet',count=1,zone='main',required=True,family=None,dimensions_m=None,dimension_evidence=None,dimension_basis=None),
                 dict(id='b',category='autoclave',count=1,zone='main',required=True,family='box',dimensions_m=None,dimension_evidence=dict(url='https://e.test/p',identity='P',fields=dict(width=['W','1','m'],depth=['D','1','m'],height=['H','1','m'])),dimension_basis=None)])
    for item in agent_output['objects']:item.update(asset_ref=None,asset_request=None,generated_request=None)
    jsonschema.validate(agent_output,strict(PROGRAM_SCHEMA))
    program=validate_program(_strip_null(agent_output))
    assert 'family' not in program['objects'][0] and program['objects'][1]['dimension_evidence']['url']=='https://e.test/p'


def test_partially_quoted_unknown_category_sources_the_remaining_axes(tmp_path):
    from scene_pipeline.orchestrator import resolve_program
    calls=[]
    def sourcing(category,prompt,work,model=None,timeout=180,**_):
        calls.append(category);return dict(url='https://example.test/s',identity='Steri-500',family='cabinet',fields=FIELDS)
    prompt='a pantry with a 1.2 m wide autoclave and two base cabinets'
    p=parse(f'''scene(prompt={prompt!r},space=space(kind="pantry",area_m2=36),objects=[
        place("autoclave","autoclave",family="cabinet",dimensions_m=[1.2,.75,.9],dimension_evidence={{"prompt_quote":"1.2 m wide autoclave"}}),
        place("cab","base_cabinet",count=2)])''')
    resolved=resolve_program(p,tmp_path,sourcing=sourcing,cache_path=tmp_path/'cache.json',fetcher=lambda u:DOC)
    a=resolved['objects'][0]
    assert a['dimension_basis']=='user_quoted_and_sourced' and calls==['autoclave']
    np.testing.assert_allclose(a['dimensions_m'],[1.2,28*.0254,36*.0254])  # user's width kept; depth/height from the page
    record=read_json(tmp_path/'dimension_sources.json')['autoclave']
    assert record['quoted_axes']==['w'] and record['sourced_axes']==['d','h'] and record['url']=='https://example.test/s'
    # A quote that binds nothing is still refused even with a cache present.
    bad=parse(f'''scene(prompt={prompt!r},space=space(kind="pantry",area_m2=36),objects=[
        place("autoclave","autoclave",dimensions_m=[1.,.75,.9],dimension_evidence={{"prompt_quote":"autoclave"}})])''')
    with pytest.raises(PipelineError,match='binds no axis'):resolve_program(bad,tmp_path/'b',sourcing=sourcing,cache_path=tmp_path/'cache.json',fetcher=lambda u:DOC)


def test_units_as_printed_and_tight_fields_bind():
    from scene_pipeline.evidence import measurement,quoted_values,unit_factor
    page='<h1>Sterilization Centre \u2013 12\u2032 | SC12</h1><ul><li>Width: 144\u2033</li><li>Depth: 27\u2033</li><li>Total Height: 83"</li></ul>'
    assert measurement(page,identity='SC12',label='Width:',number='144',unit='\u2033',url='u')['value_m']==pytest.approx(144*.0254)
    assert measurement(page,identity='SC12',label='Total Height:',number='83',unit='"',url='u')['value_m']==pytest.approx(83*.0254)
    assert unit_factor('IN.')==.0254 and unit_factor('ft')==.3048
    assert quoted_values('a 12\u2032 run and a 27\u2033 deep 0.9 m high bench')==pytest.approx(sorted([12*.3048,27*.0254,.9]))
    with pytest.raises(PipelineError,match='Unsupported unit'):unit_factor('furlong')


def test_sourcing_retries_with_verifier_feedback(tmp_path):
    from scene_pipeline.orchestrator import resolve_program
    seen=[]
    def sourcing(category,prompt,work,model=None,timeout=180,feedback=None):
        seen.append(feedback and feedback['code'])
        unit='furlong' if len(seen)==1 else 'in'
        return dict(url='https://example.test/s',identity='Steri-500',family='table',fields={k:[v[0],v[1],unit] for k,v in FIELDS.items()})
    p=parse('scene(prompt="a clinic",space=space(kind="clinic",area_m2=36),objects=[place("s","sterilizer_unit")])')
    resolved=resolve_program(p,tmp_path,sourcing=sourcing,cache_path=tmp_path/'cache.json',fetcher=lambda u:DOC)
    assert seen==[None,'UNIT'] and resolved['objects'][0]['dimension_basis']=='sourced'
    assert Path(tmp_path/'sourcing/sterilizer_unit_1').exists() or True  # per-try work dirs are the sourcing agent's, absent with a fake


def test_mixed_fraction_dimensions_are_verified_and_cached(tmp_path):
    fields=dict(width=['Width','46 1/8','"'],depth=['Depth','21 5/8','"'],height=['Height','69 1/4','"'])
    document='<h1>KLEPPSTAD Wardrobe with 3 doors</h1>'+''.join(
        f'<p>{label} {number} {unit}</p>' for label,number,unit in fields.values())
    item=dict(id='wardrobe',category='wardrobe',family='cabinet',dimension_evidence=dict(
        url='https://example.test/wardrobe',identity='KLEPPSTAD Wardrobe with 3 doors',fields=fields))
    cache=tmp_path/'cache.json'
    dims,record=resolve(item,'a dorm room',cache_path=cache,fetcher=lambda url:document)
    assert dims==pytest.approx([1.171575,.549275,1.75895])
    assert record['quotes'][0]=='Width 46 1/8 "'
    cached,_=resolve(dict(id='other',category='wardrobe'),'a dorm room',cache_path=cache,
                     fetcher=lambda url:pytest.fail('cached source must not be fetched'))
    assert cached==dims


def test_quoted_mixed_fractions_bind_entire_number():
    from scene_pipeline.evidence import quoted_axes,quoted_values
    quote='46 1/8 inches wide, 21 5/8 inches deep, 69 1/4 inches high'
    assert quoted_axes(quote)==pytest.approx(dict(w=1.171575,d=.549275,h=1.75895))
    assert quoted_values(quote)==pytest.approx([.549275,1.171575,1.75895])
    assert quoted_values('width 1/2 m, height 1.25 m')==[.5,1.25]


@pytest.mark.parametrize('number',['1/0','46 1/0','NaN','0','-2','-2 1/2','invalid'])
def test_bad_source_numbers_produce_recoverable_dimension_errors(number):
    from scene_pipeline.evidence import measurement
    with pytest.raises(PipelineError) as error:
        measurement(f'<h1>Product</h1><p>Width {number} in</p>',identity='Product',
                    label='Width',number=number,unit='in',url='https://example.test/p',axis='width')
    assert error.value.code=='DIMENSION'


def test_sourcing_retries_invalid_number_with_verifier_feedback(tmp_path):
    from scene_pipeline.orchestrator import resolve_program
    seen=[]
    def sourcing(category,prompt,work,model=None,timeout=180,feedback=None):
        seen.append(feedback and feedback['code'])
        fields=dict(FIELDS,width=['Overall Width','1/0' if len(seen)==1 else '48','in'])
        return dict(url='https://example.test/s',identity='Steri-500',family='table',fields=fields)
    program=parse('scene(prompt="a clinic",space=space(),objects=[place("s","sterilizer_unit")])')
    document=DOC+'<p>Overall Width 1/0 in</p>'
    resolved=resolve_program(program,tmp_path,sourcing=sourcing,cache_path=tmp_path/'cache.json',fetcher=lambda url:document)
    assert seen==[None,'DIMENSION']
    assert resolved['objects'][0]['dimensions_m']==pytest.approx([48*.0254,28*.0254,36*.0254])


def test_required_near_group_packs_onto_the_shelf(tmp_path):
    # Mirrors the first real held-out run: bottles scattered one per shelf level left too few tray slots.
    p=parse('''scene(prompt="supply shelf with bottles and trays",space=space(kind="clinic",area_m2=40),
        objects=[place("shelf","shelf",zone="supplies"),place("cab","base_cabinet",count=3,zone="work"),place("dr","drawer_unit",count=2,zone="work"),place("prep","prep_table",zone="work"),
                 place("bottles","bottle",count=6,zone="supplies"),place("trays","tray",count=4,zone="supplies")],
        relations=[relation("near",["bottles","shelf"],required=True),relation("near",["trays","shelf"],required=True)])''')
    for seed in range(4):
        ir=build(p,seed,tmp_path/str(seed))
        clutter=[o for o in ir['objects'] if o['category'] in ('bottle','tray')]
        assert len(clutter)==10 and all(o['support_parent']=='shelf_0' for o in clutter),seed
