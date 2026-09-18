"""Searchable pinned model catalogs. No per-category scene rules or downloaded code."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote

from .contracts import PipelineError,digest,read_json,write_json
from .evidence import fetch_bytes

SOURCES = [
    dict(id='house',repo='aws-robotics/aws-robomaker-small-house-world',revision='dd44be6205b2e6577bf641300dacc1699cb2ca81',prefix='models/',license_file='LICENSE',license='MIT-0'),
    dict(id='warehouse',repo='aws-robotics/aws-robomaker-small-warehouse-world',revision='ee0af733315e78432408c3cd98d378ecee5f767c',prefix='models/',license_file='LICENSE',license='MIT-0'),
    dict(id='hospital',repo='aws-robotics/aws-robomaker-hospital-world',revision='7161eb8448f5cba6a469da7a79ae5d660b0b7f58',prefix='models/',license_file='LICENSE',license='MIT-0'),
    dict(id='gazebo',repo='osrf/gazebo_models',revision='8163eb4b5e7e21985c6591d1c0bfb56468c0093f',prefix='',license_file='LICENSE',license='CC-BY-3.0 (repository default; see model notices)'),
]
MAX_FILE=32*1024*1024
MAX_MODEL=160*1024*1024
ALLOWED={'.sdf','.config','.dae','.stl','.obj','.mtl','.png','.jpg','.jpeg','.bmp','.tga','.material','.txt','.md'}


def store_root(path=None):
    return Path(path or os.environ.get('SCENE_PIPELINE_ASSET_STORE',Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'scene-pipeline'/'assets'))


def safe_path(path):
    p=PurePosixPath(path)
    if not path or p.is_absolute() or '..' in p.parts or '\\' in path or any(ord(c)<32 for c in path):
        raise PipelineError('ASSET_PATH','Unsafe source-relative path',path)
    return p


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.index-',dir=path.parent)
    with os.fdopen(fd,'w') as f:json.dump(value,f,allow_nan=False)
    os.replace(name,path)


def index_sources(store=None,sources=None,*,fetcher=fetch_bytes):
    root=store_root(store);entries=[]
    for source in sources or SOURCES:
        if not re.fullmatch('[a-z][a-z0-9_-]*',source['id']) or not re.fullmatch(r'[\w.-]+/[\w.-]+',source['repo']) or not re.fullmatch('[0-9a-f]{40}',source['revision']):
            raise PipelineError('ASSET_SOURCE','Sources require an identifier, GitHub owner/repository and full pinned commit')
        safe_path(source['license_file'])
        if source['prefix']:safe_path(source['prefix'])
        url=f"https://api.github.com/repos/{source['repo']}/git/trees/{source['revision']}?recursive=1"
        tree=json.loads(fetcher(url,limit=16*1024*1024,content_types=None))
        if tree.get('truncated'):raise PipelineError('ASSET_INDEX','Truncated GitHub tree cannot establish a complete catalog')
        files={e['path']:e for e in tree['tree'] if e['type']=='blob'}
        if source['license_file'] not in files:raise PipelineError('ASSET_LICENSE','Configured license file absent')
        for path in sorted(files):
            if not path.startswith(source['prefix']) or not path.endswith('/model.config'):continue
            folder=path.rsplit('/',1)[0]
            members=[]
            for name,item in files.items():
                if name.startswith(folder+'/'):
                    safe_path(name)
                    if item.get('mode')!='100644' and item.get('mode')!='100755':
                        raise PipelineError('ASSET_PATH','Symlinks are not supported in model packages',name)
                    members.append(dict(path=name,sha=item['sha'],size=item.get('size',0)))
            if not any(m['path'].endswith('.sdf') for m in members):continue
            identity=f"{source['id']}:{folder}"
            label=re.sub(r'([a-z])([A-Z])',r'\1 \2',folder.rsplit('/',1)[-1]).replace('_',' ')
            entries.append(dict(candidate_id=identity,label=label,source=source,folder=folder,files=members,
                                license_blob={k:files[source['license_file']][k] for k in ('path','sha','size')},
                                source_url=f"https://github.com/{source['repo']}/tree/{source['revision']}/{quote(folder)}"))
    result=dict(schema_version=1,entries=entries,sha256=digest(entries))
    atomic_json(root/'catalog.json',result)
    return dict(passed=True,candidates=len(entries),catalog=str(root/'catalog.json'),sha256=result['sha256'])


def catalog(store=None):
    path=store_root(store)/'catalog.json'
    if not path.exists():index_sources(store)
    value=read_json(path)
    if value.get('sha256')!=digest(value['entries']):raise PipelineError('ASSET_INDEX','Catalog hash mismatch')
    return value['entries']


def search(query,store=None,limit=12):
    tokens=set(re.findall('[a-z0-9]+',query.lower()))
    results=[]
    for entry in catalog(store):
        words=set(re.findall('[a-z0-9]+',entry['label'].lower()))
        hits=tokens&words
        if not hits:continue
        score=len(hits)/max(1,len(tokens))
        results.append({k:entry[k] for k in ('candidate_id','label','source_url')} | dict(score=score,
            license=entry['source']['license'],download_bytes=sum(f['size'] for f in entry['files']),
            semantic_match='lexical_candidate_not_verified',capabilities='inspect_after_import'))
    return sorted(results,key=lambda e:(-e['score'],e['candidate_id']))[:limit]


def download(candidate_id,store=None,*,fetcher=fetch_bytes):
    root=store_root(store)
    entry=next((e for e in catalog(root) if e['candidate_id']==candidate_id),None)
    if entry is None:raise PipelineError('ASSET_NOT_FOUND','Candidate is not in the pinned catalog',candidate_id)
    identity=digest(entry);target=root/'sources'/identity
    if target.exists():
        record=read_json(target/'source.json')
        for name,expected in record['sha256'].items():
            safe_path(name)
            if hashlib.sha256((target/name).read_bytes()).hexdigest()!=expected:raise PipelineError('ASSET_HASH','Cached source changed',name)
        return target,record
    members=[f for f in entry['files'] if Path(f['path']).suffix.lower() in ALLOWED or re.search(r'(license|notice|copying|copyright)',Path(f['path']).name,re.I)]
    members.append(entry['license_blob'])
    if len(members)>300 or sum(f['size'] for f in members)>MAX_MODEL or any(f['size']>MAX_FILE for f in members):
        raise PipelineError('ASSET_BUDGET','Model exceeds bounded download budget')
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='download-',dir=target.parent))
    source=entry['source'];hashes={}
    def get(item):
        relative='REPOSITORY_LICENSE' if item['path']==source['license_file'] else str(PurePosixPath(item['path']).relative_to(entry['folder']))
        safe_path(relative)
        data=fetcher(f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/{quote(item['path'])}",limit=min(MAX_FILE,item['size']+1),content_types=None)
        # Verify Git blob identity, not merely a checksum supplied alongside downloaded bytes.
        if len(data)!=item['size'] or hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=item['sha']:
            raise PipelineError('ASSET_HASH','Downloaded file does not match pinned Git blob',item['path'])
        path=staging/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return relative,hashlib.sha256(data).hexdigest()
    try:
        # Bounded parallel downloads; each URL uses public-address and redirect checks.
        with ThreadPoolExecutor(max_workers=4) as pool:hashes=dict(pool.map(get,members))
        record=dict(candidate=entry,sha256=hashes,units='SDF metres; source-model scale, not product measurement')
        write_json(staging/'source.json',record)
        try:staging.rename(target)
        except FileExistsError:return download(candidate_id,root,fetcher=fetcher)
    except Exception as exc:
        write_json(staging/'failure.json',dict(error=str(exc)));raise
    return target,record
