import unittest

try:
    from modules.ui import _preferred_camera_position
    _UI_IMPORTABLE = True
except Exception:  # PySide6 / cv2 not available in this environment
    _UI_IMPORTABLE = False


@unittest.skipUnless(_UI_IMPORTABLE, "modules.ui (PySide6/cv2) not importable")
class PreferredCameraPositionTests(unittest.TestCase):
    def test_prefers_builtin_over_iphone_continuity(self):
        # macOS enumerates the iPhone Continuity Camera at index 0; its slow
        # warm-up shows as a black preview, so it must not be the default.
        names = ["Nicholas Cariello’s iPhone Camera", "MacBook Pro Camera"]
        self.assertEqual(_preferred_camera_position(names), 1)

    def test_keeps_first_when_it_is_a_real_webcam(self):
        names = ["MacBook Pro Camera", "Some USB Webcam"]
        self.assertEqual(_preferred_camera_position(names), 0)

    def test_skips_ipad_and_desk_view(self):
        names = ["iPad Camera", "Desk View Camera", "Logitech BRIO"]
        self.assertEqual(_preferred_camera_position(names), 2)

    def test_falls_back_to_first_when_all_are_continuity(self):
        names = ["My iPhone Camera", "My iPad Camera"]
        self.assertEqual(_preferred_camera_position(names), 0)

    def test_single_camera(self):
        self.assertEqual(_preferred_camera_position(["MacBook Pro Camera"]), 0)


if __name__ == "__main__":
    unittest.main()
