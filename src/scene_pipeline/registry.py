"""Small explicit engineering registry, not a learned dimension distribution.

Defaults below are declared design parameters for parametric proxies, not product
measurements or legal compliance claims. Policy version participates in cache keys.
"""
from .contracts import PipelineError, digest

POLICY = {
    'version': 'kitchen-engineering-v1',
    'status': 'engineering_defaults_not_hardware_calibrated',
    'density_kg_m3': {'panel': 600., 'steel': 7800., 'plastic': 950.},
    'panel_thickness_m': .018, 'gap_m': .004,
    'aisle_m': 1.2, 'door_clear_width_m': .95, 'room_height_m': 2.8,
    'wall_thickness_m': .12, 'aspect': 1.35,
    'timestep_s': .002, 'settle_s': 5., 'penetration_m': .002,
    'slide_drift_m': .005, 'hinge_drift_rad': .008726646259971648,
    'joint_damping': 2., 'joint_frictionloss': .2,
    'retrieved_shell_proxy_kg_m3': 180.,
    'robot_radius_m': .25, 'robot_radius_note': 'Default access disc for a compact mobile base (Stretch-class footprint); override per robot.',
    'retrieved_shell_note': 'Effective bulk density for hollow appliance collider volumes, not material density or measured appliance mass. Explicit engineering proxy; original source remains archived.',
}

# Geometric family dimensions are design defaults; no fabricated manufacturer citations.
CATALOG = {
    'base_cabinet': dict(route='G1', dimensions=[.6, .6, .9], family='cabinet', articulated=True,finish='paint'),
    'drawer_unit': dict(route='G1', dimensions=[.6, .6, .9], family='drawer', articulated=True,finish='paint'),
    'wall_cabinet': dict(route='G1', dimensions=[.6, .35, .7], family='cabinet', articulated=True,finish='paint',mount_height_m=1.55),
    'prep_table': dict(route='G1', dimensions=[1.2, .7, .9], family='table', articulated=False,finish='stainless',placement='freestanding'),
    'counter': dict(route='G1', dimensions=[1.2, .6, .9], family='table', articulated=False,finish='stainless'),
    'shelf': dict(route='G1', dimensions=[1., .45, 1.8], family='shelf', articulated=False,finish='stainless'),
    'door': dict(route='G1', dimensions=[.95, .04, 2.1], family='door', articulated=True),
    'container': dict(route='G1', dimensions=[.09, .09, .13], family='container', articulated=False,placement='support',dynamic=True,finish='cream'),
    'jar': dict(route='G1',dimensions=[.09,.09,.14],family='jar',articulated=False,placement='support',dynamic=True,finish='amber'),
    'bottle': dict(route='G1',dimensions=[.07,.07,.22],family='bottle',articulated=False,placement='support',dynamic=True,finish='blue'),
    'tray': dict(route='G1',dimensions=[.32,.22,.055],family='tray',articulated=False,placement='support',dynamic=True,finish='stainless'),
    'microwave': dict(route='G3', source='microwaves/Microwave075', articulated=True,placement='support',support_height_m=[.7,1.2]),
    'dishwasher': dict(route='G3', source='dishwashers/Dishwasher051', articulated=True),
    'fridge': dict(route='G3', source='fridges/Refrigerator055', articulated=True),
}

# Brief-facing plausibility bands, checked on the compiled model. Wide by design: these catch
# wrong-by-an-order-of-magnitude scale and mass, not manufacturing tolerances.
TOP_BANDS_M={'counter':(.85,.95),'prep_table':(.85,.95),'base_cabinet':(.85,.95),'drawer_unit':(.85,.95)}
DOOR_CLEAR_WIDTH_M=(.813,1.2)
MASS_BANDS_KG={'base_cabinet':(15,80),'drawer_unit':(15,80),'wall_cabinet':(8,50),'prep_table':(10,150),'counter':(10,150),
               'shelf':(10,150),'door':(10,80),'container':(.05,3),'jar':(.05,2),'bottle':(.05,2),'tray':(.1,3),
               'fridge':(40,250),'microwave':(8,40),'dishwasher':(25,90)}

# Persistent taxonomy: append entries; never derive IDs from one scene's inventory.
CLASS_IDS = {'base_cabinet':1,'drawer_unit':2,'wall_cabinet':3,'prep_table':4,
             'counter':5,'shelf':6,'door':7,'container':8,'microwave':9,'dishwasher':10,'fridge':11,'jar':12,'bottle':13,'tray':14}


def class_id(category):
    """Fixed taxonomy first; open-vocabulary categories get a stable hashed id in 1000..9999."""
    # ponytail: hash ids can collide across two novel categories in one scene; move to an
    # append-only taxonomy file if that ever shows up in a manifest.
    return CLASS_IDS.get(category) or 1000+int(digest(category)[:8],16)%9000


def search(category=None):
    return {k: {**v, 'provenance_kind': 'engineering_default' if v['route']=='G1' else 'retrieved_source',
                'policy': POLICY['version']} for k, v in CATALOG.items() if category is None or category in k}


def lookup(category):
    if category not in CATALOG:
        raise PipelineError('UNKNOWN_CATEGORY', f'No approved route for {category}')
    return CATALOG[category].copy()


def fingerprint():
    return digest({'policy': POLICY, 'catalog': CATALOG,'class_ids':CLASS_IDS})
