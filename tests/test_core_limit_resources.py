import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def _patched_core_import_stubs():
    """Import modules.core with its heavy optional deps stubbed out."""
    stubs = {
        "cv2": types.SimpleNamespace(
            IMREAD_COLOR=1,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (True, None),
        ),
        "numpy": types.SimpleNamespace(uint8=object, fromfile=lambda *_a, **_k: b""),
        "onnxruntime": types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"]
        ),
        "tensorflow": types.SimpleNamespace(),
        "modules.metadata": types.SimpleNamespace(name="Deep-Live-Cam", version="test"),
        "modules.ui": types.SimpleNamespace(
            check_and_ignore_nsfw=lambda *_args, **_kwargs: False,
            update_status=lambda *_args, **_kwargs: None,
            init=lambda *_args, **_kwargs: types.SimpleNamespace(mainloop=lambda: None),
        ),
        "modules.processors.frame.core": types.SimpleNamespace(
            get_frame_processors_modules=lambda _names: [],
            process_video_in_memory=lambda *_args, **_kwargs: False,
        ),
        "modules.utilities": types.SimpleNamespace(
            has_image_extension=lambda _path: False,
            is_image=lambda _path: False,
            is_video=lambda _path: True,
            detect_fps=lambda _path: 24.0,
            create_video=lambda *_a, **_k: True,
            extract_frames=lambda *_a, **_k: None,
            get_temp_frame_paths=lambda *_a, **_k: [],
            restore_audio=lambda *_a, **_k: None,
            create_temp=lambda *_a, **_k: None,
            move_temp=lambda *_a, **_k: None,
            clean_temp=lambda *_a, **_k: None,
            normalize_output_path=lambda _source, _target, output: output,
        ),
    }
    with patch.dict(sys.modules, stubs, clear=False):
        sys.modules.pop("modules.core", None)
        yield importlib.import_module("modules.core")
        sys.modules.pop("modules.core", None)


class LimitResourcesTests(unittest.TestCase):
    def test_setrlimit_einval_does_not_crash_startup(self):
        """macOS rejects setrlimit(RLIMIT_DATA) with EINVAL, which CPython
        surfaces as ValueError. The best-effort memory cap must swallow it
        instead of taking down startup."""
        with _patched_core_import_stubs() as core:
            core.HAS_TENSORFLOW = False
            core.modules.globals.max_memory = 4

            def _raise_einval(*_args, **_kwargs):
                raise ValueError("current limit exceeds maximum limit")

            with patch("resource.setrlimit", side_effect=_raise_einval):
                # Must not raise.
                core.limit_resources()

    def test_setrlimit_receives_clamped_soft_limit(self):
        """When the platform accepts it, the soft limit is applied without
        raising the hard limit above the existing one."""
        with _patched_core_import_stubs() as core:
            core.HAS_TENSORFLOW = False
            core.modules.globals.max_memory = 4

            captured = {}

            def _capture(which, limits):
                captured["which"] = which
                captured["limits"] = limits

            import resource

            with patch("resource.setrlimit", side_effect=_capture), patch(
                "resource.getrlimit",
                return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY),
            ):
                core.limit_resources()

            self.assertEqual(captured["which"], resource.RLIMIT_DATA)
            soft, hard = captured["limits"]
            self.assertEqual(soft, 4 * 1024 ** 3)
            self.assertEqual(hard, resource.RLIM_INFINITY)


if __name__ == "__main__":
    unittest.main()
