"""Append-only HDF5 streams and reproducible variant collection."""
import json
from pathlib import Path

import h5py
import numpy as np

from .contracts import PipelineError,write_json,read_json


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


def collect_variants(scene,output,variants=10,seconds=60):
    from .orchestrator import generate
    from .flows import run
    from .compiler import compile_scene
    from .validation import validate_scene
    output=Path(output)
    if output.exists(): raise PipelineError('OUTPUT_EXISTS',str(output))
    if not 1<=variants<=100: raise PipelineError('VARIANT_COUNT','Expected 1..100 variants')
    output.mkdir(parents=True)
    program=read_json(Path(scene)/'program.json');results=[]
    for i in range(variants):
        target=output/f'variant_{i:03d}'
        try:
            generate(program['prompt'],i+100,target,program=program,preview=False)
            # Appearance randomization is encoded into IR, not patched in generated MJCF.
            ir=read_json(target/'ir.json');rng=np.random.default_rng(i+100)
            ir['meta']['appearance']={'light_multiplier':float(rng.uniform(.8,1.2)),
                                      'tint':rng.uniform(.8,1.,3).tolist(),'texture_repeat':float(rng.uniform(5,12))}
            write_json(target/'ir.json',ir);compile_scene(ir,target)
            validation=validate_scene(target)
            if not validation['passed']:
                raise PipelineError('VARIANT_VALIDATION','Appearance variant failed validation',validation)
            tier='full' if i<3 else 'state'
            report=run(target,'mapping',target/'mapping',seconds=seconds,tier=tier,seed=i)
            results.append(dict(variant=i,tier=tier,report=report,streams=inspect(target/'mapping'/'data.h5')))
        except PipelineError as exc:
            results.append(dict(variant=i,error=exc.as_dict()))
    summary=dict(schema_version=1,variants=results,passed=all(x.get('report',{}).get('passed',False) for x in results),
                 tier_rationale='Three RGB-D runs; remaining runs avoid rendering overhead while retaining state/range/IMU observations and truth.')
    write_json(output/'dataset.json',summary)
    (output/'DATA_CARD.md').write_text('# Generated kitchen mapping dataset\n\n'+summary['tier_rationale']+'\n\nSynthetic engineering sensor noise; not hardware calibrated. Stock small omnibase; no claim of full-size service-robot dynamics. Fixed category/material vocabulary; kitchen-biased. Failed runs remain explicitly reported. See dataset.json and HDF5 metadata for seeds, timestamps, provenance and acceptance results.\n')
    return summary
