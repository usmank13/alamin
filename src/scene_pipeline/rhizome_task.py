"""Inspect Rhizome's task and node configuration contract from a checkout.

This does not start nodes or replace NodeConfig. Missing hardware configuration
is reported explicitly; the runtime must supply it before constructing nodes.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tomllib

import yaml


class Loader(yaml.SafeLoader):
    pass


# Match Rhizome's YAML parser: on/off are ordinary keys, not booleans.
Loader.yaml_implicit_resolvers={
    k:[(tag,rx) for tag,rx in v if tag!='tag:yaml.org,2002:bool']
    for k,v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
Loader.add_implicit_resolver('tag:yaml.org,2002:bool',
    re.compile(r'^(?:true|True|TRUE|false|False|FALSE)$'),list('tTfF'))


def merge_patch(target, patch):
    if not isinstance(patch,dict):return deepcopy(patch)
    target=deepcopy(target) if isinstance(target,dict) else {}
    for key,value in patch.items():
        if value is None:target.pop(key,None)
        else:target[key]=merge_patch(target.get(key),value)
    return target


class Task:
    def __init__(self, root, name, *, config_dir=None):
        self.root=Path(root).resolve()
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name):
            raise ValueError('Task must name src/tasks/<task>.yml')
        self.name=name
        self.config_dir=Path(config_dir).resolve() if config_dir else None
        self.sources={}
        self.config=self.read(self.root/'src/tasks'/f'{name}.yml') or {}
        if not isinstance(self.config,dict):raise ValueError('Task must be a mapping')
        self.nodes={}
        for path in sorted((self.root/'src').rglob('*.node.*')):
            if path.suffix not in ('.yml','.yaml','.json','.toml'):continue
            node=path.name.split('.node.')[0]
            if node in self.nodes:raise ValueError(f'Ambiguous node configuration: {node}')
            self.nodes[node]=path

    def read(self, path):
        data=path.read_bytes()
        self.sources[str(path)]=hashlib.sha256(data).hexdigest()
        if path.suffix=='.json':return json.loads(data)
        if path.suffix=='.toml':return tomllib.loads(data.decode())
        documents=list(yaml.load_all(data,Loader=Loader))
        if len(documents)==2:return dict(meta=documents[0],config=documents[1])
        if len(documents)!=1:raise ValueError(f'Expected one or two YAML documents: {path}')
        return documents[0]

    def services(self):
        """Device membership follows Rhizome tools/generate_task.py."""
        baseline={}
        for name,path in self.nodes.items():
            for device in self.read(path)['meta'].get('target_devices',[]):
                baseline.setdefault(device,set()).add(name)
        explicit=self.config.get('services') or {}
        extra=self.config.get('extra_services') or {}
        disabled=set(self.config.get('disabled_services') or [])
        devices=baseline.keys()
        unknown_devices=(explicit.keys() | extra.keys())-devices
        if unknown_devices:raise ValueError(f'Task names devices without node targets: {sorted(unknown_devices)}')
        result={device:sorted(set(explicit.get(device,[])) if explicit else
            (baseline.get(device,set())-disabled) | set(extra.get(device,[]))) for device in sorted(devices)}
        unknown={name for names in result.values() for name in names}-self.nodes.keys()
        if unknown:raise ValueError(f'Task names services without node configs: {sorted(unknown)}')
        return result

    def include_path(self, name):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Include must be relative to the configuration directory: {name}')
        if self.config_dir and (self.config_dir/relative).is_file():return self.config_dir/relative
        if relative.parts[0]=='nodes':
            return self.nodes.get(relative.stem)
        path=self.root/'src'/relative
        return path if path.is_file() else None

    def node(self, name):
        raw=self.read(self.nodes[name])
        meta=raw['meta']
        config=deepcopy(raw['config'] or {})
        missing=[]
        for include in meta.get('includes',[]):
            path=self.include_path(include)
            if path is None:
                missing.append(include)
                continue
            parts=Path(include).with_suffix('').parts
            target=config
            patch=self.config
            for part in parts[:-1]:
                if not isinstance(target.get(part),dict):target[part]={}
                target=target[part]
                patch=patch.get(part,{}) if isinstance(patch,dict) else {}
            key=parts[-1]
            target[key]=merge_patch(target.get(key),self.read(path))
            if isinstance(patch,dict) and key in patch:
                target[key]=merge_patch(target[key],patch[key])
        overrides=self.config.get('nodes') or {}
        if name in overrides:config=merge_patch(config,overrides[name])
        return dict(meta=meta,config=config,missing_includes=missing)

    def snapshot(self):
        services=self.services()
        nodes={name:self.node(name) for name in sorted({n for names in services.values() for n in names})}
        return dict(task=self.name,services=services,nodes=nodes,source_sha256=self.sources.copy(),
                    configuration_complete=all(not n['missing_includes'] for n in nodes.values()))
