"""Python-shaped declarative DSL, interpreted as data without exec/eval/imports."""
import ast
import json
from pathlib import Path

from .contracts import PipelineError,validate_program


def space(kind='kitchen',area_m2=60,shape='rectangle',annexes=0,inferred_fields=None):
    return dict(kind=kind,area_m2=area_m2,shape=shape,annexes=annexes,
                **({'inferred_fields':inferred_fields} if inferred_fields is not None else {}))


def place(id,category,count=1,zone='main',required=True,family=None,dimensions_m=None,dimension_evidence=None,asset_ref=None,asset_request=None,generated_request=None):
    extra={k:v for k,v in dict(family=family,dimensions_m=dimensions_m,dimension_evidence=dimension_evidence,asset_ref=asset_ref,asset_request=asset_request,generated_request=generated_request).items() if v is not None}
    return dict(id=id,category=category,count=count,zone=zone,required=required,**extra)


def relation(kind,objects,required=False):
    return dict(kind=kind,objects=objects,required=required)


def scene(prompt,space,objects,relations=None):
    return validate_program(dict(schema_version=1,prompt=prompt,space=space,objects=objects,relations=relations or []))


def parse(text):
    functions={'scene':scene,'space':space,'place':place,'relation':relation}
    def value(node,depth=0):
        if depth>30: raise PipelineError('DSL_DEPTH','Program nesting exceeds limit')
        if isinstance(node,ast.Constant) and isinstance(node.value,(str,int,float,bool,type(None))): return node.value
        if isinstance(node,(ast.List,ast.Tuple)): return [value(x,depth+1) for x in node.elts]
        if isinstance(node,ast.Dict) and all(isinstance(k,ast.Constant) and isinstance(k.value,str) for k in node.keys):
            return {k.value:value(v,depth+1) for k,v in zip(node.keys,node.values)}
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in functions:
            if any(k.arg is None for k in node.keywords): raise PipelineError('DSL_SYNTAX','Keyword expansion prohibited')
            return functions[node.func.id](*[value(x,depth+1) for x in node.args],**{k.arg:value(k.value,depth+1) for k in node.keywords})
        raise PipelineError('DSL_SYNTAX','Only literal scene/space/place/relation expressions are supported')
    if len(text)>100_000: raise PipelineError('DSL_SIZE','Program exceeds size limit')
    try:
        return validate_program(value(ast.parse(text,mode='eval').body))
    except (SyntaxError,TypeError) as exc:
        raise PipelineError('DSL_SYNTAX',str(exc)) from exc


def load(path):
    path=Path(path)
    return validate_program(json.loads(path.read_text())) if path.suffix=='.json' else parse(path.read_text())
