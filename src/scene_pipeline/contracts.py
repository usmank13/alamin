"""Strict, portable JSON contracts and content identities."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import jsonschema

VERSION = 1


class PipelineError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}

    def as_dict(self):
        return dict(code=self.code, message=str(self), details=self.details)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def obj(properties, required=None):
    return dict(type='object', properties=properties, required=list(properties) if required is None else required,
                additionalProperties=False)


ID = dict(type='string', pattern='^[a-z][a-z0-9_]*$')
NUM = dict(type='number')
VEC3 = dict(type='array', items=NUM, minItems=3, maxItems=3)
# Open-vocabulary objects: the agent names a template family and either quotes the user's
# stated size or points at a page the harness fetches and verifies. Bare numbers are rejected.
FAMILY = dict(type='string', enum=['box', 'table', 'shelf', 'cabinet', 'drawer'])
SIZE = dict(type='array', items={'type': 'number', 'exclusiveMinimum': 0}, minItems=3, maxItems=3)
FIELD = dict(type='array', items={'type': 'string', 'minLength': 1}, minItems=3, maxItems=3)  # [label, number, unit]
EVIDENCE = {'oneOf': [obj({'prompt_quote': {'type': 'string', 'minLength': 1}}),
                      obj({'url': {'type': 'string', 'pattern': '^https?://'}, 'identity': {'type': 'string', 'minLength': 1},
                           'fields': obj({'width': FIELD, 'depth': FIELD, 'height': FIELD})})]}
ASSET_REQUEST=obj({'query':{'type':'string','minLength':1,'maxLength':300},
    'candidate_id':{'type':'string','minLength':1,'maxLength':400},
    'mode':{'type':'string','enum':['static','articulated','dynamic']},
    'placement':{'type':'string','enum':['freestanding','support','wall']}},required=['query'])
GENERATED_REQUEST=obj({'prompt':{'type':'string','minLength':1,'maxLength':900},
    'size_m':{'type':'number','minimum':.02,'maximum':2.},
    'placement':{'type':'string','enum':['support','freestanding']},
    # Legacy authored requests remain loadable; new imports always add the proxy.
    'physical_use':{'type':'string','enum':['static_collision','visual_only']}})
PROGRAM_SCHEMA = obj({
    'schema_version': {'type':'integer','const': VERSION},
    'prompt': {'type': 'string', 'minLength': 1},
    'space': obj({'kind': {'type':'string','minLength':1,'maxLength':100},
                  'area_m2': {'type': 'number', 'minimum': 30, 'maximum': 120},
                  'shape': {'type':'string','enum': ['rectangle', 'l_shape']},
                  'annexes': {'type': 'integer', 'minimum': 0, 'maximum': 2},
                  'inferred_fields': {'type':'array','uniqueItems':True,'items':{'type':'string','enum':['kind','area_m2','shape','annexes']}}},
                 required=['kind','area_m2','shape','annexes']),
    'objects': {'type': 'array', 'minItems': 1, 'maxItems': 100, 'items': obj({
        'id': ID, 'category': ID, 'count': {'type': 'integer', 'minimum': 1, 'maximum': 50},
        'zone': ID, 'required': {'type': 'boolean'},
        'family': FAMILY, 'dimensions_m': SIZE, 'dimension_evidence': EVIDENCE,
        'asset_ref':{'type':'string','pattern':'^[0-9a-f]{64}$'},'asset_request':ASSET_REQUEST,'generated_request':GENERATED_REQUEST,
        'dimension_basis': {'type': 'string', 'enum': ['user_quoted', 'sourced', 'user_quoted_and_sourced']}},
        required=['id', 'category', 'count', 'zone', 'required'])},
    'relations': {'type': 'array', 'items': obj({'kind': {'type':'string','enum': ['against_wall', 'near', 'in_row', 'under']},
                                               'objects': {'type': 'array', 'items': ID, 'minItems': 1},
                                               'required': {'type': 'boolean'}})},
})

PROGRAM_SCHEMA_V2=deepcopy(PROGRAM_SCHEMA)
PROGRAM_SCHEMA_V2['properties']['schema_version']={'type':'integer','const':2}
PROGRAM_SCHEMA_V2['properties']['objects']['minItems']=0
PROGRAM_SCHEMA_V2['properties']['space']['properties']['shape']['enum']=['sampled','rectangle','l_shape','concave']
PROGRAM_SCHEMA_V2['properties']['space']['properties']['annexes']={'type':'integer','const':0}
ORIGIN={'type':'string','enum':['explicit','inferred']}
PROGRAM_SCHEMA_V2['properties']['architecture']=obj({
    'openings':{'type':'array','maxItems':20,'items':obj({
        'id':ID,'kind':{'type':'string','enum':['door','window']},
        'count':{'type':'integer','minimum':1,'maximum':20},'role':{'type':'string'},
        'required':{'type':'boolean'},'origin':ORIGIN})},
    'relationships':{'type':'array','items':obj({
        'kind':{'type':'string','enum':['same_wall','opposite_wall']},
        'openings':{'type':'array','items':ID,'minItems':2,'maxItems':2},
        'required':{'type':'boolean'},'origin':ORIGIN})},
    'unsupported_requirements':{'type':'array','items':{'type':'string','minLength':1}}},required=['openings','relationships'])
PROGRAM_SCHEMA_V2['required'].append('architecture')


def validate_program(program):
    try:
        jsonschema.validate(program, PROGRAM_SCHEMA_V2 if program.get('schema_version')==2 else PROGRAM_SCHEMA)
        canonical(program)  # Disallow NaN/Infinity (JSON Schema number alone permits these).
    except (jsonschema.ValidationError, ValueError) as exc:
        raise PipelineError('PROGRAM_SCHEMA', str(exc)) from exc
    ids = [x['id'] for x in program['objects']]
    if len(set(ids)) != len(ids):
        raise PipelineError('DUPLICATE_ID', 'Scene program IDs must be unique')
    for relation in program['relations']:
        if not set(relation['objects']) <= set(ids):
            raise PipelineError('UNKNOWN_REFERENCE', 'Relation references an absent object')
    for o in program['objects']:
        if any(k in o for k in ('asset_ref','asset_request','generated_request')):
            if sum(k in o for k in ('asset_ref','asset_request','generated_request'))>1 or any(k in o for k in ('family','dimensions_m','dimension_evidence','dimension_basis')):
                raise PipelineError('ASSET_CONTRACT','Use an asset request/reference OR a sized template; native asset dimensions cannot be silently overridden',o['id'])
        evidence, basis = o.get('dimension_evidence'), o.get('dimension_basis')
        if evidence is not None and basis is not None:
            raise PipelineError('UNBOUND_MEASUREMENT', 'Unresolved evidence and a resolved basis cannot coexist', o['id'])
        if 'dimensions_m' in o and evidence is None and basis is None:
            raise PipelineError('UNBOUND_MEASUREMENT', 'dimensions_m requires dimension_evidence', o['id'])
        quote = (evidence or {}).get('prompt_quote')
        if (basis is not None or quote is not None) and 'dimensions_m' not in o:
            raise PipelineError('UNBOUND_MEASUREMENT', 'Quoted or resolved dimensions must state dimensions_m', o['id'])
        if quote is not None and quote not in program['prompt']:
            raise PipelineError('UNBOUND_MEASUREMENT', 'prompt_quote must occur verbatim in the prompt', o['id'])
    if program['schema_version']==2:
        requests=program['architecture']['openings'];ids=[r['id'] for r in requests]
        if len(set(ids))!=len(ids):raise PipelineError('DUPLICATE_ID','Duplicate opening request')
        for r in requests+program['architecture']['relationships']:
            if r['origin']=='explicit' and not r['required']:
                raise PipelineError('INTENT_AUTHORITY','Explicit architecture requirements cannot be optional')
        for r in program['architecture']['relationships']:
            if len(set(r['openings']))!=2 or not set(r['openings'])<=set(ids):
                raise PipelineError('UNKNOWN_REFERENCE','Invalid opening relationship references')
    return program


def validate_ir(ir):
    required = {'schema_version', 'meta', 'rooms', 'openings', 'objects', 'robots', 'provenance'}
    if ir.get('schema_version') == 2:
        required.add('architecture')
    if set(ir) != required or ir['schema_version'] not in (1,2):
        raise PipelineError('IR_SCHEMA', 'Unsupported or incomplete SceneIR')
    if ir['meta']['units'] != 'metres' or ir['meta']['up'] != 'Z':
        raise PipelineError('FRAME', 'SceneIR must be metres and Z-up')
    ids = [o['id'] for o in ir['objects']]
    if len(ids) != len(set(ids)):
        raise PipelineError('DUPLICATE_ID', 'Instance IDs must be unique')
    for item in ir['objects']:
        if len(item['position']) != 3 or len(item['dimensions']) != 3 or min(item['dimensions']) <= 0:
            raise PipelineError('IR_DIMENSIONS', 'Invalid instance dimensions or position')
        p = Path(item['asset'])
        if p.is_absolute() or '..' in p.parts:
            raise PipelineError('ASSET_PATH', 'IR asset references must be package-relative')
    if ir['schema_version']==2:
        from shapely.geometry import Polygon
        architecture=ir['architecture']
        keys={'walls','height','door_height','window_sill','window_top'}
        if set(architecture)!=keys or not 0<architecture['window_sill']<architecture['window_top']<architecture['height'] or not 0<architecture['door_height']<architecture['height']:
            raise PipelineError('ARCHITECTURE_SCHEMA','Invalid architecture heights or fields')
        for section in (ir['rooms'],ir['openings'],architecture['walls']):
            if len({x['id'] for x in section})!=len(section):
                raise PipelineError('DUPLICATE_ID','Duplicate architectural IDs')
            for record in section:
                polygon=Polygon(record['polygon'])
                if not polygon.is_valid or polygon.area<=0:
                    raise PipelineError('ARCHITECTURE_POLYGON','Architecture requires valid polygons')
        for opening in ir['openings']:
            if opening['kind'] not in ('door','window'):
                raise PipelineError('ARCHITECTURE_OPENING','Unsupported opening kind')
    canonical(ir)
    return ir
