"""Sensor-only integration regressions; no scene ground truth supplied."""
import numpy as np
import pytest
pytest.importorskip('h5py')
pytest.importorskip('scipy')
pytest.importorskip('jsonschema')
from scene_pipeline.flows import Mapper
from scene_pipeline import sensors


def mapper():
    result=Mapper.__new__(Mapper)
    result.res=.1;result.origin=np.zeros(2);result.shape=(20,20)
    result.logodds=np.zeros(result.shape);result.observed=np.zeros(result.shape,dtype=bool)
    result.pose=np.array([.501,.501,0.]);result.scan_map={};result.scan_match_residual=None;result.arrived=False;result.speed=.18;result.path=[];result.last_plan=-1.;result.wheels=np.eye(3)
    return result


def test_missing_scan_does_not_clear_obstacles():
    m=mapper();m.logodds[8,5]=4.
    m.observe(np.full(72,np.nan),np.zeros(72,dtype=bool))
    assert m.logodds[8,5]==4 and not m.observed.any()
    np.testing.assert_array_equal(m.control(5.,np.full(72,np.nan)),np.zeros(3))


def test_mapping_accuracy_is_reported_not_an_invented_gate(tmp_path):
    m=mapper();m.truth=np.zeros(m.shape,dtype=bool);m.truth[10,10]=True
    m.legacy_truth=m.truth.copy();m.initial_xy=m.pose[:2].copy();m.travel=0.
    m.observed[:]=True
    report=m.report(tmp_path)
    assert report['occupancy_iou']==0. and report['passed']
    assert 'no spec accuracy threshold' in report['acceptance']
    m.observed[:]=False
    assert not m.report(tmp_path)['passed']


def test_hit_cell_is_not_also_cleared():
    m=mapper();ranges=np.full(72,np.nan);valid=np.zeros(72,dtype=bool)
    ranges[36]=.099;valid[36]=True
    m.observe(ranges,valid)
    assert m.logodds[5,5]==pytest.approx(1.2)


def test_registration_references_do_not_follow_pose_error():
    m=mapper();point=np.array([.7,.5]);key=tuple(np.round(point/.05))
    m.scan_map[key]=point.copy()
    ranges=np.full(72,np.nan);ranges[36]=.2;valid=np.isfinite(ranges)
    m.observe(ranges,valid)
    np.testing.assert_array_equal(m.scan_map[key],point)


def test_no_return_still_observes_free_space():
    m=mapper();ranges=np.full(72,np.nan);ranges[36]=1.
    m.observe(ranges,np.zeros(72,dtype=bool))
    assert m.logodds[10,5]<0 and m.observed[10,5]
    assert not (m.logodds>0).any()


def test_dropout_is_not_a_max_range_return(monkeypatch):
    monkeypatch.setattr(sensors,'sensor',lambda *args:np.array([1.]))
    class DroppingRng:
        def normal(self,mean,sigma,n):return np.zeros(n)
        def random(self,n):return np.zeros(n)
    truth,observation,valid=sensors.noisy_ranges(None,'',DroppingRng())
    assert (truth==1).all() and np.isnan(observation).all() and not valid.any()


def test_surface_matching_corrects_translation_and_yaw():
    m=mapper();m.pose=np.array([1.04,.97,.02])
    # An independently supplied synthetic scan map of a rectangular room.
    points=[]
    for x in np.arange(0,4.01,.04):points.extend(([x,0],[x,3]))
    for y in np.arange(0,3.01,.04):points.extend(([0,y],[4,y]))
    m.scan_map={tuple(p):np.array(p,dtype=float) for p in points}
    ranges=[]
    for a in np.linspace(-np.pi,np.pi,72,endpoint=False):
        dx,dy=np.cos(a),np.sin(a);candidates=[]
        if abs(dx)>1e-9:candidates.append((3 if dx>0 else -1)/dx)
        if abs(dy)>1e-9:candidates.append((2 if dy>0 else -1)/dy)
        ranges.append(min(candidates))
    m.observe(np.array(ranges),np.ones(72,dtype=bool))
    np.testing.assert_allclose(m.pose,[1,1,0],atol=.003)


def test_goal_resolution_through_semantic_manifest():
    from scene_pipeline.flows import resolve_goal
    from scene_pipeline.contracts import PipelineError
    box=[[0,0,0],[1,1,2]]
    manifest={'instances':{'cold_storage_0':dict(id='cold_storage_0',category='fridge',position=[5,0,.9],bounds=box),
                           'cold_storage_1':dict(id='cold_storage_1',category='fridge',position=[1,0,.9],bounds=box),
                           'drawers_0':dict(id='drawers_0',category='drawer_unit',position=[2,2,0],bounds=box)}}
    goal=resolve_goal('Go to the walk-in fridge',manifest,(0,0))
    assert goal['id']=='cold_storage_1' and goal['alternatives']==['cold_storage_0']
    assert resolve_goal('cold_storage_0',manifest,(0,0))['id']=='cold_storage_0'
    assert resolve_goal('open the drawer',manifest,(0,0))['category']=='drawer_unit'
    with pytest.raises(PipelineError) as failure:resolve_goal('go to the sink',manifest,(0,0))
    assert failure.value.code=='UNKNOWN_GOAL' and 'fridge' in failure.value.details['categories']


def test_goal_directed_plan_stops_at_free_cell_beside_goal():
    from scene_pipeline.flows import clearance,REACH
    m=mapper();m.observed[:]=True;m.logodds[:]=-1;m.logodds[14:17,8:11]=4
    lo,hi=np.array([1.4,.8]),np.array([1.7,1.1])
    ranges=np.full(72,2.);local=m.control(0.,ranges,(lo,hi))
    assert local[0]>0 and m.path and not m.arrived
    end=m.path[-1];centre=m.origin+(np.array(end)+.5)*m.res
    assert m.logodds[end]<=0 and clearance(centre,lo,hi)<=REACH
    from scipy.ndimage import binary_dilation
    assert not binary_dilation(m.logodds>0,iterations=2)[end]  # outside the two-cell obstacle dilation
    assert m.drive(local,ranges).shape==(3,)
    blocked=ranges.copy();blocked[36]=.1  # ray 36 points along +x, the commanded direction
    assert not m.drive([.3,0,0],blocked).any()
    m.pose=np.array([1.2,.95,0.])
    np.testing.assert_array_equal(m.control(1.,ranges,(lo,hi)),np.zeros(3));assert m.arrived
    survey=mapper();survey.observed[:]=True;survey.logodds[:]=-1
    assert survey.control(0.,ranges).shape==(3,) and survey.path  # mapping mode still explores
