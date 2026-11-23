"""Drive the native OdometryNode fixture once per 10 ms simulation sample."""
import json

import numpy as np
from scipy.spatial.transform import Rotation


def vector(values, frame):
    return dict(zip(('x', 'y', 'z'), values), frame=frame)


class Odometry:
    def __init__(self, runtime, config, description):
        from pyzome import nav

        self.p = p = runtime.p
        if not hasattr(p, 'set_sim_time') or not hasattr(nav, 'set_test_description'):
            raise ValueError('Build the Rhizome host with scene_pipeline.rhizome_build --sim-odometry')
        self.set_clock = p.set_sim_time
        self.set_clock(p.seconds(1))
        meta, params = runtime.documents('src/nav/odometry.node.yml')
        overrides = config['params']
        if unknown := overrides.keys() - params.keys():
            raise ValueError(f'Unknown odometry parameters: {sorted(unknown)}')
        if meta['trigger']['period'] != .01:
            raise ValueError('Odometry trigger must match the 100 Hz sensor clock')
        # The wheel sample is sent last and triggers exactly one fixed-dt update.
        meta['trigger'].update(type='ON_MSG', topics=['/motors/velocity'])
        params.update(overrides)
        (runtime.conf/'nodes/odometry.json').write_text(json.dumps(dict(meta=meta, config=params)))
        nav.set_test_description(description)
        chassis = json.loads(description.description)['chassis']
        self.antenna = np.array([chassis['rtk_rover'][axis] for axis in 'xyz'])
        self.odom = p.OdometryMsg()
        self.frame = p.CoordinateFrame()
        self.offset = p.PointMsg()
        self.received_motors = p.DriveMotorMsg()
        self.fixture, self.node = runtime.create_node('odometry', nav.Odometry, {
            '/odom': self.odom, '/frames/odom': self.frame, '/odom/path_offset': self.offset,
            '/motors/velocity': self.received_motors})
        self.map_frame = p.make_frame(p.Pose(frame=p.Frame.MAP), p.Pose(frame=p.Frame.ODOM))
        self.last_fix = None

    def tick(self, t, sample):
        p = self.p
        ns = round((t + 1) * 1e9)
        self.set_clock(p.nanoseconds(ns))
        inputs = {
            '/imu/raw': p.ImuMsg(dict(accel=vector(sample['accel'], 'robot'),
                                      gyro=vector(sample['gyro'], 'robot'))),
            '/imu/attitude': p.OrientationMsg(dict(zip(('qw', 'qx', 'qy', 'qz'), sample['attitude']),
                                                   frame='robot')),
            '/frames/map': self.map_frame,
        }
        if fix := sample.get('rtk_kinematics'):
            w, x, y, z = fix['rotation']
            lever = Rotation.from_quat([x, y, z, w]).apply(self.antenna)
            position = np.asarray(fix['position']) + lever
            velocity = np.asarray(fix['velocity']) + np.cross(fix['omega'], lever)
            enu = p.Point(x=position[0], y=position[1], z=0., frame=p.Frame.MAP)
            gps = p.geo.map_to_gps(enu, p.Point(x=40., y=-88., z=0., frame=p.Frame.GPS))
            height = position[2] + fix['dz']
            inputs['/rtk/data'] = p.Rtk(dict(position=vector([gps.x, gps.y, height], 'gps'),
                velocity=vector(velocity, 'map'), position_status=fix['status'],
                v_acc=fix['v_acc'], h_acc=.014, time=t, num_sats=0, pdop=0., heading_acc=0.))
            self.last_fix = dict(time=t, height_m=height, status=fix['status'], v_acc=fix['v_acc'])
        for topic, message in inputs.items():
            self.fixture.publish(topic, message)
        self.fixture.publish('/motors/velocity', p.DriveMotorMsg(sample['wheels']))
        self.fixture.tick()
        for message in (self.odom, self.frame, self.offset, self.received_motors):
            if message.header.timestamp.nanoseconds() != ns:
                raise RuntimeError('Odometry fixture and native publications must use simulation time')
