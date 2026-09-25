"""Repackage a legacy grid model with current geometry, preserving its core.

Run with ml-aigen-tools' inference Python. The source package is never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools-root',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=832)
    parser.add_argument('--height',type=int,default=480)
    args=parser.parse_args()
    root=args.tools_root.resolve();source=args.package.resolve();out=args.output.resolve()
    if out.exists():parser.error('Choose a fresh output directory')
    sys.path.insert(0,str(root))
    import torch
    import yaml
    from aigen.models.octonet.data.class_mapper import ClassMapper
    from aigen.models.octonet.variants.grid.package import build_postprocessor,write_config
    from aigen.models.octonet.training.combined_model import OctoNetGridPostModule
    config=yaml.safe_load((source/'config.yml').read_text())
    if config.get('variant')!='grid':parser.error('Expected a grid package')
    torch.set_num_threads(2)
    model=torch.jit.load(str(source/'model.ts'),map_location='cpu').eval()
    mapper=ClassMapper.default()
    out.mkdir(parents=True)
    torch.jit.save(model.core,str(out/'core.ts'))
    post,grid=build_postprocessor(config,mapper,args.width,args.height)
    torch.jit.save(torch.jit.script(OctoNetGridPostModule(post)),str(out/'post.ts'))
    write_config(config,grid,mapper,str(out/'config.yml'),
                 crop_conditioning=config['inference']['crop_conditioning'],
                 core_input_order=config['inference']['core_input_order'])
    loaded=torch.jit.load(str(out/'core.ts')).state_dict()
    original=model.core.state_dict()
    if set(loaded)!=set(original) or any(not torch.equal(v,loaded[k]) for k,v in original.items()):
        raise ValueError('Repackaged core weights differ from original')
    if (source/'reference.npz').exists():shutil.copyfile(source/'reference.npz',out/'reference.npz')
    manifest=dict(original_package=str(source),core_weights_identical=True,
        tools_root=str(root),tools_revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        segmentation_classes=mapper.seg_classes,
        change='Unchanged trained core; current ml-aigen-tools geometry postprocessor and class thresholds',
        source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('model.ts','config.yml')},
        output_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('core.ts','post.ts','config.yml')})
    (out/'upgrade.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
