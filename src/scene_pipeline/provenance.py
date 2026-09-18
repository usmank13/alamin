"""Deterministic source classification, not appearance-based guessing or certification."""
from collections import Counter
from .contracts import PipelineError,write_json


def classify(asset):
    source=asset.get('provenance',{});route=asset.get('route')
    if route=='G1' and source.get('kind') in ('engineering_default','sourced_template'):
        origin,provider='procedural','scene_pipeline.templates'
        basis=source.get('dimension_basis') or 'engineering_design_parameters'
    elif route=='G3' and source.get('kind')=='retrieved_source':
        origin,provider,basis='retrieved',source.get('provider','RoboCasa / Lightwheel'),source.get('dimension_basis','native_asset_scale_not_measured_product_dimensions')
    elif route in ('G6','G8') and source.get('kind')=='generated_source':
        origin,provider,basis='generated',source.get('provider','unspecified'),'unverified_generated_geometry'
    else:
        raise PipelineError('PROVENANCE_UNKNOWN','Cannot classify contradictory or unsupported asset provenance')
    return dict(schema_version=1,origin=origin,provider=provider,route=route,
                source_id=source.get('source',asset.get('category')),asset_key=asset.get('key'),
                dimension_basis=basis,physical_basis='not_assigned_visual_only' if origin=='generated' else source.get('physical_policy','engineering_density_proxy'),
                allowed_use='visual_only' if origin=='generated' else 'simulation_candidate',
                contact_rich_certified=False,validation_state=asset.get('state','candidate'),
                license=source.get('license','unspecified'),calibrated=source.get('calibrated',False))


def inventory(manifest,path):
    objects={key:value['source_classification'] for key,value in manifest['instances'].items()}
    result=dict(schema_version=1,objects=objects,counts=dict(Counter(v['origin'] for v in objects.values())),
                architecture=dict(origin='procedural',provider='scene_pipeline.floorplan',dimension_basis='program_area_and_engineering_defaults'),
                materials=manifest.get('materials'),
                note='Origin is not evidence of contact suitability; validation and capability are separate. fal jobs, when used, are listed under materials and per-instance provenance.')
    layout=manifest.get('layout_provenance',{})
    if layout.get('generator')=='reference':
        result['architecture']=dict(origin='retrieved_2d_procedurally_extruded',provider=layout['source']['dataset'],
                                    source=layout['source'],scale=layout['scale'],height_basis=layout['height_basis'])
    elif layout.get('layout_generator'):
        result['architecture']['layout_grounding']=layout['layout_generator']
    elif layout.get('architecture_sampling'):
        result['architecture']=dict(origin='empirical_joint_geometry_procedurally_realized',
                                    evidence=layout['architecture_sampling'],metric_scale_verified=False,
                                    semantic_function='not_certified_by_geometric_validation')
    write_json(path,result)
    return result
