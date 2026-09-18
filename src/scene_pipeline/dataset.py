"""Append-only HDF5 streams and reproducible variant collection."""
from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np

from .contracts import PipelineError,write_json,read_json
from .registry import CATALOG
from . import fal


class Recorder:
    def __init__(self,path,metadata):
        self.file=h5py.File(path,'x')
        self.file.attrs['schema_version']=1
        self.file.attrs['metadata_json']=json.dumps(metadata,allow_nan=False)

    def append(self,stream,time,**fields):
        group=self.file.require_group(stream)
        for key,value in {'time':np.float64(time),**fields}.items():
            value=np.asarray(value)
            if key not in group:
                group.create_dataset(key,shape=(0,*value.shape),maxshape=(None,*value.shape),dtype=value.dtype,
                                     chunks=True,compression='lzf' if value.ndim else None)
            ds=group[key]
            ds.resize(len(ds)+1,axis=0);ds[-1]=value

    def close(self):
        self.file.close()


def inspect(path):
    with h5py.File(path,'r') as f:
        if f.attrs['schema_version']!=1: raise PipelineError('DATASET_VERSION','Unsupported dataset')
        result={}
        for name,g in f.items():
            times=g['time'][:]
            if len(times)>1 and not np.all(np.diff(times)>0): raise PipelineError('CLOCK','Non-monotonic stream time')
            if any(len(ds)!=len(times) for ds in g.values()): raise PipelineError('STREAM_LENGTH','Misaligned fields')
            if not np.allclose(times/.002,np.round(times/.002),atol=1e-6): raise PipelineError('CLOCK','Samples off physics clock')
            result[name]={'samples':len(times),'fields':list(g.keys())}
        return result


def load_stream(path,stream,start=None,stop=None):
    """Read a bounded sample slice; timestamps align fields within each stream.

    Different-rate streams must be joined by time, not by array index.
    """
    with h5py.File(path,'r') as f:
        if f.attrs['schema_version']!=1: raise PipelineError('DATASET_VERSION','Unsupported dataset')
        if stream not in f: raise PipelineError('UNKNOWN_STREAM',stream)
        return {name:ds[slice(start,stop)] for name,ds in f[stream].items()}


CLUTTER=('container','jar','bottle','tray')+tuple(k for k,v in CATALOG.items() if v['route']=='G6')


def randomize_program(program,rng):
    """Clutter-count axis. Placement re-samples through the layout seed; appearance lives in IR meta."""
    program=deepcopy(program);counts={}
    for o in program['objects']:
        if o['category'] in CLUTTER:
            o['count']=max(1,int(round(o.get('count',1)*rng.uniform(.7,1.3))));counts[o['id']]=o['count']
    return program,counts


def collect_variants(scene,output,variants=10,seconds=60,*,start_seed=100):
    from .orchestrator import generate
    from .flows import run
    from .compiler import compile_scene
    from .validation import validate_scene
    output=Path(output)
    if output.exists(): raise PipelineError('OUTPUT_EXISTS',str(output))
    if not 1<=variants<=100: raise PipelineError('VARIANT_COUNT','Expected 1..100 variants')
    output.mkdir(parents=True)
    scene=Path(scene)
    program=read_json(scene/('input_program.json' if (scene/'input_program.json').exists() else 'program.json'));results=[];fal_records=[]
    # References replay from the portable scene, never re-search a mutable remote catalog.
    resolved=read_json(scene/'program.json');by_id={o['id']:o for o in resolved['objects']}
    asset_store=output/'asset_library'
    from .asset_library import verify
    import shutil
    for item in program['objects']:
        ref=by_id[item['id']].get('asset_ref')
        if not ref:continue
        source=scene/'assets'/ref;verify(source,ref)
        destination=asset_store/'packages'/ref
        if not destination.exists():shutil.copytree(source,destination)
        item.pop('asset_request',None);item.pop('generated_request',None);item['asset_ref']=ref
    config=read_json(scene/'generation.json') if (scene/'generation.json').exists() else dict(layout_backend='heuristic')
    materials=config.get('materials','flat');clutter=config.get('clutter','off');budget=config.get('max_fal_usd')
    config={k:v for k,v in config.items() if k in ('layout_backend','allow_prior_backoff','robot_radius','access_margin','architecture_only')}
    priors=read_json(scene/'priors.json') if (scene/'priors.json').exists() else None
    if config.get('layout_backend') in ('architecture','empirical') and priors is None:
        raise PipelineError('REPLAY_EVIDENCE','Generation requires the original prior bundle')
    cache=output/'evidence_cache.json'
    write_json(cache,read_json(scene/'evidence_cache.json') if (scene/'evidence_cache.json').exists() else {})
    for i in range(variants):
        target=output/f'variant_{i:03d}';seed=start_seed+i;rng=np.random.default_rng(seed)
        try:
            variant,counts=randomize_program(program,rng)
            generation=generate(program['prompt'],seed,target,program=variant,preview=False,priors=priors,cache_path=cache,asset_store=asset_store,
                                materials=materials,clutter=clutter,max_fal_usd=budget,**config)
            if not generation['passed']:raise PipelineError('VARIANT_VALIDATION','Variant generation is partial, not accepted')
            # Appearance randomization is encoded into IR, not patched in generated MJCF.
            ir=read_json(target/'ir.json')
            appearance={**ir['meta'].get('appearance',{}),'light_multiplier':float(rng.uniform(.8,1.2)),
                        'tint':rng.uniform(.8,1.,3).tolist(),'texture_repeat':float(rng.uniform(5,12))}
            if materials=='fal':
                # Texture-set axis: three PATINA seeds per finish bound the spend; every seed is cached and recorded per variant.
                appearance['materials']=dict(source='fal',seed=int(rng.integers(3)))
                fal_records+=fal.prefetch(fal.material_jobs(appearance['materials']['seed']),budget_usd=budget,spent_usd=sum(r['cost_usd'] for r in fal_records))
                write_json(output/'fal_calls.json',fal_records)
            ir['meta']['appearance']=appearance
            factors=dict(layout_seed=seed,clutter_counts=counts,**appearance)
            write_json(target/'ir.json',ir);compile_scene(ir,target)
            validation=validate_scene(target,require_articulated=config.get('layout_backend')!='architecture',robot_radius=config.get('robot_radius'))
            if not validation['passed']:
                raise PipelineError('VARIANT_VALIDATION','Appearance variant failed validation',validation)
            tier='full' if i<3 else 'state'
            report=run(target,'mapping',target/'mapping',seconds=seconds,tier=tier,seed=i)
            results.append(dict(variant=i,tier=tier,factors=factors,report=report,streams=inspect(target/'mapping'/'data.h5')))
        except PipelineError as exc:
            results.append(dict(variant=i,layout_seed=seed,error=exc.as_dict()))
    summary=dict(schema_version=1,start_seed=start_seed,variants=results,passed=all(x.get('report',{}).get('passed',False) for x in results),
                 tier_rationale='Three RGB-D runs; remaining runs avoid rendering overhead while retaining state/range/IMU observations and truth.',
                 randomization='Per variant: layout seed (object placement), clutter counts x0.7-1.3 for containers/jars/bottles/trays and decor, light multiplier 0.8-1.2, material tint, texture repeat, and with --materials fal the PBR texture set seed (0-2); factors recorded per variant.')
    write_json(output/'dataset.json',summary)
    (output/'DATA_CARD.md').write_text('# Generated mapping dataset\n\n'+summary['tier_rationale']+'\n\n'+summary['randomization']+' Sequential collection; each variant also carries replay.mp4 when ffmpeg is present.\n\nSynthetic engineering sensor noise; not hardware calibrated. Stock small omnibase; no claim of full-size service-robot dynamics. Fixed category and finish vocabulary; kitchen-biased; PBR sets and decor meshes, when enabled, are fal generations at declared engineering tile sizes and size bands. Failed runs remain explicitly reported. See dataset.json and HDF5 metadata for seeds, timestamps, provenance and acceptance results.\n')
    return summary
