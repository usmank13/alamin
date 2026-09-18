"""Replaceable layout producers; compiler/export consumers depend only on SceneIR."""
from pathlib import Path
import numpy as np

from .contracts import PipelineError,validate_ir,write_json

BACKENDS=('heuristic','empirical','architecture')


def generate_layout(program,seed,root,*,backend='heuristic',priors=None,allow_prior_backoff=False,candidates=12,robot_radius=None,access_margin=.05):
    from .layout import solve
    root=Path(root)
    if backend not in BACKENDS:raise PipelineError('GENERATOR_UNKNOWN',f'Unknown backend {backend}')
    if backend=='architecture':
        from .architecture_sampler import sample_architecture
        ir=sample_architecture(program,seed,root,bundle=priors,allow_backoff=allow_prior_backoff,
                               robot_radius=robot_radius,access_margin=access_margin)
        return solve(program,seed,root,architecture=ir) if program['objects'] else ir
    if program['schema_version']!=1:raise PipelineError('GENERATOR_CONTRACT','Legacy backends require SceneProgram v1')
    if backend=='heuristic':
        # Try a few deterministic placements before spending another agent call.
        # Keep the first success; do not weaken geometry or relation checks.
        trials=[]
        for offset in range(3):
            from .runtime import CURRENT
            if CURRENT.get():CURRENT.get().remaining()
            try:ir=solve(program,seed+offset,root)
            except PipelineError as exc:
                if exc.code not in ('LAYOUT_UNSAT','RELATION_UNSAT'):raise
                trials.append(dict(seed=seed+offset,error=exc.as_dict()))
                write_json(root/'layout_candidates.json',trials)
                if offset==2:raise
                continue
            trials.append(dict(seed=seed+offset,passed=True))
            write_json(root/'layout_candidates.json',trials)
            ir['provenance']['layout_generator']=dict(name='heuristic',grounding='engineering_defaults_not_empirical',
                                                     requested_seed=seed,selected_seed=seed+offset,candidate_count=len(trials))
            return validate_ir(ir)
    if priors is None:raise PipelineError('PRIOR_REQUIRED','Empirical backend requires an explicit source-backed bundle')
    if program['space']['annexes'] or program['space']['shape']!='rectangle':
        raise PipelineError('PRIOR_DOMAIN','First empirical backend supports a single rectangular room')
    if not 1<=candidates<=64:raise PipelineError('CANDIDATE_BUDGET','Candidate budget must be 1..64')
    from .layout_priors import condition,distance,scene_features,evaluate
    prior=condition(priors,program['space']['kind'],allow_backoff=allow_prior_backoff)
    rng=np.random.default_rng(seed);best=None;trials=[]
    for i in range(candidates):
        # Sample aspect from training only. Joint-feature scoring chooses placement;
        # no source room coordinates or held-out layouts are copied.
        row=prior['train'][int(rng.integers(len(prior['train'])))];aspect=row['values']['aspect']
        try:
            candidate=solve(program,seed+i,root,aspect=aspect)
            score=distance(scene_features(candidate,root),prior)
            trials.append(dict(candidate=i,score=score,aspect=aspect,source_group=row['source_group'],room_id=row['room_id']))
            if best is None or score<best[0]:best=(score,candidate,trials[-1])
        except PipelineError as exc:trials.append(dict(candidate=i,error=exc.as_dict()))
    if best is None:raise PipelineError('LAYOUT_UNSAT','All empirical proposals failed',dict(trials=trials))
    ir=best[1]
    ir['meta']['seed']=seed
    ir['provenance']['layout_generator']=dict(name='empirical',algorithm='empirical_aspect_and_best_of_seeded_placements_v1',
        prior_sha256=priors['sha256'],conditioning=prior['backoff'],selected=best[2],candidate_count=candidates,
        limitations=['No empirical zone/group placement yet','No calibrated distribution acceptance gate',
                     'Area, heights, openings and asset sizes are not measured by this corpus'])
    write_json(root/'layout_candidates.json',trials)
    write_json(root/'layout_distribution.json',evaluate(ir,root,priors,allow_backoff=allow_prior_backoff))
    return validate_ir(ir)
