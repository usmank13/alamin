"""Cached fal meshes -> static, visual-only MuJoCo asset packages (route G6). Never calls the network."""
import hashlib
import io
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from . import fal
from .contracts import PipelineError, read_json


def config_from_request(request):
    from .contracts import GENERATED_REQUEST
    import jsonschema
    jsonschema.validate(request, GENERATED_REQUEST)
    return dict(route='G6',prompt=request['prompt'],size_m=[request['size_m']]*2,
                size_basis='agent_estimate_visual_only_not_measured',placement=request['placement'],
                dynamic=False,support_height_m=[.3,1.6])


def package(category, config):
    """MjSpec plus provenance for a decor category from the fal cache: Z-up, band-midpoint size, base on z=0, no contact geoms."""
    model, payload = fal.decor_request(category, config)
    entry = fal.cached(model, payload)
    if entry is None:
        raise PipelineError('GENERATED_ASSET_UNAVAILABLE', f'No cached fal mesh for {category}; run generate with --clutter fal and FAL_KEY set',
                            dict(category=category, model=model))
    import trimesh
    from trimesh.exchange.obj import export_obj
    glb = next(iter(sorted(entry.glob('*.glb'))), None)
    if glb is None:
        raise PipelineError('GENERATED_ASSET_UNAVAILABLE', f'Cache entry for {category} holds no GLB', dict(folder=str(entry)))
    parts = trimesh.load(str(glb), force='scene').dump()
    if not parts:
        raise PipelineError('GENERATED_MESH_EMPTY', f'{category}: GLB has no geometry', dict(folder=str(entry)))
    mesh = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    # trimesh preserves glTF scene coordinates (Y-up). Rotate the fully transformed
    # scene into MuJoCo Z-up explicitly; a same-library round trip hid this before.
    mesh.apply_transform([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
    # Uniform scale to the declared band midpoint on the largest extent.
    scale = float(np.mean(config['size_m']))/float(max(mesh.extents))
    mesh.apply_scale(scale)
    lo, hi = mesh.bounds; mesh.apply_translation([-(lo[0]+hi[0])/2, -(lo[1]+hi[1])/2, -lo[2]])
    textured = getattr(mesh.visual, 'uv', None) is not None and len(mesh.visual.uv) == len(mesh.vertices)
    obj, files = export_obj(mesh, include_texture=True, return_texture=True) if textured else (export_obj(mesh, include_texture=False), {})
    images = [v for k, v in files.items() if k.lower().endswith(('.png', '.jpg', '.jpeg'))]
    assets = {'mesh.obj': obj.encode()}
    xml = ET.Element('mujoco', model=category)
    ET.SubElement(xml, 'compiler', angle='radian')
    asset = ET.SubElement(xml, 'asset'); ET.SubElement(asset, 'mesh', name='mesh', file='mesh.obj')
    if images:
        buffer = io.BytesIO(); Image.open(io.BytesIO(images[0])).convert('RGB').save(buffer, 'PNG'); assets['albedo.png'] = buffer.getvalue()
        ET.SubElement(asset, 'texture', name='albedo', type='2d', file='albedo.png')
        # ponytail: albedo only; glTF metallicRoughness/normal maps would need ORM/normal layers here and in the worker.
        ET.SubElement(asset, 'material', name='surface', texture='albedo', shininess='.3', reflectance='.05')
    else:
        ET.SubElement(asset, 'material', name='surface', rgba='.7 .7 .7 1', shininess='.3', reflectance='.05')
    body = ET.SubElement(ET.SubElement(xml, 'worldbody'), 'body', name='root')
    ET.SubElement(body, 'geom', name='surface_visual', type='mesh', mesh='mesh', material='surface', group='2', contype='0', conaffinity='0', density='0')
    spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding='unicode'), assets=assets)
    meta = read_json(entry/'meta.json')
    hashes = {**meta.get('files', {}), **{k: hashlib.sha256(v).hexdigest() for k, v in assets.items()}}
    provenance = dict(kind='generated_source', provider='fal', model=model, request_id=meta.get('request_id'), source=meta.get('request_id') or fal.key(model, payload),
                      prompt=payload['prompt'], retrieved_at=meta.get('completed_at'), license='fal.ai and Tencent Hunyuan3D terms; not independently verified',
                      hashes=hashes, scale=scale, size_basis=config.get('size_basis','registry_decor_size_band_midpoint'), size_band_m=list(config['size_m']),
                      orientation='gltf_y_up_to_mujoco_z_up_explicit_rx_90', textured=bool(images), physical_use='visual_only_no_contact_geometry', calibrated=False)
    return spec, provenance
