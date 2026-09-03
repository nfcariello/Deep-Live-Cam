import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from modules.virtual_camera import VirtualCamera


class _FakeCamera:
    def __init__(self, width, height, fps, **_kwargs):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = "Fake Virtual Camera"
        self.sent = []
        self.closed = False

    def send(self, frame):
        self.sent.append(frame)

    def close(self):
        self.closed = True


def _fake_pyvirtualcam(camera_cls):
    return types.SimpleNamespace(Camera=camera_cls)


class VirtualCameraTests(unittest.TestCase):
    def test_start_reports_when_pyvirtualcam_missing(self):
        vc = VirtualCamera()
        # sys.modules[name] = None makes `import name` raise ImportError.
        with patch.dict(sys.modules, {"pyvirtualcam": None}):
            ok, msg = vc.start(640, 480, 30)
        self.assertFalse(ok)
        self.assertFalse(vc.is_running)
        self.assertIn("pyvirtualcam", msg)

    def test_start_reports_when_device_open_fails(self):
        class _Raises:
            def __init__(self, *a, **k):
                raise RuntimeError("no backend device")

        vc = VirtualCamera()
        with patch.dict(sys.modules, {"pyvirtualcam": _fake_pyvirtualcam(_Raises)}):
            ok, msg = vc.start(640, 480, 30)
        self.assertFalse(ok)
        self.assertFalse(vc.is_running)
        self.assertIn("no backend device", msg)

    def test_start_send_resizes_and_close(self):
        created = {}

        def _factory(width, height, fps, **kw):
            cam = _FakeCamera(width, height, fps, **kw)
            created["cam"] = cam
            return cam

        vc = VirtualCamera()
        with patch.dict(sys.modules, {"pyvirtualcam": _fake_pyvirtualcam(_factory)}):
            ok, _msg = vc.start(64, 48, 30)
            self.assertTrue(ok)
            self.assertTrue(vc.is_running)

            # A frame of a different size must be resized to the camera size.
            vc.send(np.zeros((100, 200, 3), dtype=np.uint8))
            cam = created["cam"]
            self.assertEqual(len(cam.sent), 1)
            self.assertEqual(cam.sent[0].shape, (48, 64, 3))

            vc.close()
            self.assertTrue(cam.closed)
            self.assertFalse(vc.is_running)

    def test_send_is_noop_when_not_running(self):
        vc = VirtualCamera()
        # Must not raise even though nothing is started.
        vc.send(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertFalse(vc.is_running)


if __name__ == "__main__":
    unittest.main()
