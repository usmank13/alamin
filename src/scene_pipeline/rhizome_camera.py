"""MuJoCo RGB-D frames in Rhizome's existing navigation SHM wire format."""
from pathlib import Path
import struct

import numpy as np

from .navigation import raw_depth


def combo_frame(rgb, depth_m, timestamp_ns):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB image')
    if np.shape(depth_m) != rgb.shape[:2] or rgb.shape[0] * rgb.shape[1] < 4:
        raise ValueError('Depth dimensions must match RGB and contain metadata space')
    rgba = np.full((*rgb.shape[:2], 4), 255, dtype=np.uint8)
    rgba[:, :, :3] = rgb
    depth = bytearray(raw_depth(depth_m).astype('<u2').tobytes())
    struct.pack_into('<Q', depth, 0, timestamp_ns)
    return rgba.tobytes() + depth


class ShmCamera:
    def __init__(self, path, width=640, height=480):
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        self.Gst = Gst
        Gst.init(None)
        if not Gst.ElementFactory.find('shmsink'):
            raise RuntimeError('GStreamer shmsink missing; install gstreamer1.0-plugins-bad')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Rhizome's combo reader treats this as an opaque, oversized GRAY8 frame.
        self.pipeline = Gst.parse_launch(
            f'appsrc name=source is-live=true format=time do-timestamp=true '
            f'caps="video/x-raw,format=GRAY8,width={width},height={height*2},framerate=15/1" '
            f'! shmsink socket-path={path} shm-size={width*height*6*4} '
            'wait-for-connection=false sync=false')
        self.source = self.pipeline.get_by_name('source')
        self.pipeline.set_state(Gst.State.PLAYING)

    def push(self, payload):
        buffer = self.Gst.Buffer.new_allocate(None, len(payload), None)
        buffer.fill(0, payload)
        if self.source.emit('push-buffer', buffer) != self.Gst.FlowReturn.OK:
            raise RuntimeError('SHM camera rejected frame')
        message = self.pipeline.get_bus().pop_filtered(self.Gst.MessageType.ERROR)
        if message:
            raise RuntimeError(str(message.parse_error()))

    def close(self):
        self.pipeline.set_state(self.Gst.State.NULL)
