"""Frozen program-declared semantics, separate from metric realizations/evidence.

This compatibility contract does NOT certify interpretation of natural language.
It prevents a repair loop from weakening its first accepted required inventory.
"""
from copy import deepcopy

from .contracts import PipelineError,digest,validate_program


def freeze(program, *, authority):
    validate_program(program)
    if authority not in ('supplied_program','agent_program_not_independently_verified'):
        raise PipelineError('INTENT_AUTHORITY','Unknown intent authority')
    return dict(schema_version=1,kind='scene_intent',authority=authority,architecture=deepcopy(program.get('architecture')),
                prompt=program['prompt'],space=deepcopy(program['space']),
                requirements=deepcopy(program['objects']),relations=deepcopy(program['relations']),
                source_program_sha256=digest(program),
                limitations=['Semantic extraction not independently verified','Zone labels do not yet define zone geometry'])


def check_revision(intent,program):
    validate_program(program)
    agent=intent['authority']=='agent_program_not_independently_verified'
    inferred=intent['space'].get('inferred_fields',[]) if agent else []
    fixed_space=[k for k in intent['space'] if k!='inferred_fields' and k not in inferred]
    if program['prompt']!=intent['prompt'] or any(program['space'].get(k)!=intent['space'][k] for k in fixed_space):
        raise PipelineError('INTENT_DRIFT','Repair changed the prompt or space requirement')
    by_id={r['id']:r for r in program['objects']}
    for original in intent['requirements']:
        if not original['required']:continue
        revised=by_id.get(original['id'],{})
        # Required inventory is semantic, not a commitment to the first retrieved
        # package or inferred template. Supplied programs retain exact contracts.
        keys=['id','category','count','zone','required'] if agent else list(original)
        if (original.get('dimension_evidence') or {}).get('prompt_quote') or original.get('dimension_basis','').startswith('user_quoted'):
            keys+=['dimensions_m','dimension_evidence','dimension_basis']
        if (not agent and revised!=original) or any(revised.get(k)!=original.get(k) for k in keys):
            raise PipelineError('INTENT_DRIFT','Repair changed a frozen required object request',original)
    for relation in intent['relations']:
        if relation['required'] and relation not in program['relations']:
            raise PipelineError('INTENT_DRIFT','Repair dropped a frozen required relation',relation)
    if intent.get('architecture') is not None:
        for requirement in intent['architecture'].get('unsupported_requirements',[]):
            if requirement not in program.get('architecture',{}).get('unsupported_requirements',[]):
                raise PipelineError('INTENT_DRIFT','Repair dropped a declared unsupported requirement')
        for key in ('openings','relationships'):
            for r in intent['architecture'][key]:
                if r['origin']=='explicit' and r not in program.get('architecture',{}).get(key,[]):
                    raise PipelineError('INTENT_DRIFT','Repair changed an explicit architecture requirement',r)


def check_inventory(intent,ir):
    errors=[]
    for request in intent['requirements']:
        if not request['required']:continue
        matches=[o for o in ir['objects'] if o['id'].rsplit('_',1)[0]==request['id']]
        if len(matches)!=request['count'] or any(o['category']!=request['category'] or o['zone']!=request['zone'] for o in matches):
            errors.append(dict(code='REQUIRED_INVENTORY',request=request['id']))
    return errors
