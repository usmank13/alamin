from collections import Counter

import pytest

from scene_pipeline.rhizome_e2e_trace import _analyze_arm


def test_applied_strike_requires_active_plan_source_and_exact_motor_reference():
    data = dict(detections={10: (12, Counter(WEED=1))}, fields={10: 11},
                selected={19: (10, 16, [1, 1]), 99: (10, 17, [1])},
                activated={19: (10, 18)}, commands={20: (.3, .4), 25: (.3, .4)},
                ticks=[(20, 19, False, 1, (.3, .4)), (25, 99, False, 1, (.3, .4))])
    rows = []
    for ts, count, cmd, measured in [(5, 0, None, .0), (10, 1, None, .0),
                                      (21, 1, 20, .1), (30, 1, 25, .2)]:
        rows.append(dict(frame_ns=ts, time_s=ts/1e9, end_time_s=(ts+1)/1e9, camera_capture_ns=10 if count else None,
                         camera_frames={'a': count}, applied_commands={'a': dict(pitch=.3, yaw=.4)},
                         end_measured={'a': dict(pitch=measured+.01,yaw=.0)},
                         command_ns={'a': cmd}, commands={'a': dict(pitch=.3, yaw=.4)},
                         measured={'a': dict(pitch=measured, yaw=.0)}))
    result = _analyze_arm(data, 'a', rows, 5, 30)
    assert result['matched_activations'] == 1
    assert result['commanded_strikes'] == result['applied_strikes'] == 1
    assert result['unmatched_plan_ticks'] == 1
    assert result['applied_chain_example']['moved_toward_reference']
    rows[2]['applied_commands']['a']['pitch'] = .2
    clipped = _analyze_arm(data, 'a', rows, 5, 30)
    assert clipped['commanded_strikes'] == 1 and clipped['applied_strikes'] == 0
    assert clipped['matched_native_request_steps'] == 1 and clipped['modified_reference_steps'] == 1
    assert clipped['matched_applied_steps'] == 0
    assert clipped['applied_chain_example']['reference_rad']['pitch'] == .2
    legacy = [{k: v for k, v in row.items() if k not in ('applied_commands', 'end_measured')}
              for row in rows]
    unverified = _analyze_arm(data, 'a', legacy, 5, 30)
    assert unverified['matched_native_request_steps'] == 1 and unverified['applied_strikes'] is None
    assert unverified['applied_chain_example'] is None
    rows[2]['commands']['a']['pitch'] = .9
    with pytest.raises(ValueError, match='references differ'):
        _analyze_arm(data, 'a', rows, 5, 30)
