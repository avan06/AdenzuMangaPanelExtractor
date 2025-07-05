class Model:
    def __init__(self):
        self.model = None
        self.imported = False

    def load(self):
        if self.model is None:
            self.__load()

    def __load(self):
        if not self.imported:
            self.imported = True
            import torch
            import torch.serialization
            from torch.serialization import safe_globals
            import pathlib
            import sys
            import os
            from myutils.respath import resource_path
            from yolov5.models.yolo import DetectionModel

            # Add DetectionModel to safe globals—must be inside context that covers load
            torch.serialization.add_safe_globals([DetectionModel])
            
            # Context manager to ensure allowlisted globals during load
            self._safe_ctx = safe_globals([DetectionModel])
            
        # Redirect sys.stderr to a file or a valid stream
        if sys.stderr is None:
            sys.stderr = open(os.devnull, 'w')
            
        # Check if the current operating system is Windows
        is_windows = (sys.platform == "win32")

        # Use context during actual load
        with getattr(self, '_safe_ctx', dummy_context()):
            if is_windows:
                # If on Windows, apply the patch temporarily
                temp = pathlib.PosixPath
                pathlib.PosixPath = pathlib.WindowsPath
                try:
                    # Load the model with the patch applied
                    self.model = torch.hub.load(
                        'ultralytics/yolov5', 'custom',
                        path=resource_path('ai-models/2024-11-00/best.pt'),
                        force_reload=True
                    )
                finally:
                    # CRITICAL: Always restore the original class, even if loading fails
                    pathlib.PosixPath = temp
            else:
                # If on Linux, macOS, or other systems, load the model directly
                self.model = torch.hub.load(
                    'ultralytics/yolov5', 'custom',
                    path=resource_path('ai-models/2024-11-00/best.pt'),
                    force_reload=True
                )

    def __call__(self, *args, **kwds):
        if self.model is None:
            self.__load()
        return self.model(*args, **kwds)

# A no-op context in case safe_globals isn't set
from contextlib import contextmanager
@contextmanager
def dummy_context():
    yield

model = Model()
