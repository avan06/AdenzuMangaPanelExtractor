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
            import pathlib
            import sys
            import os
            from myutils.respath import resource_path

        # Redirect sys.stderr to a file or a valid stream
        if sys.stderr is None:
            sys.stderr = open(os.devnull, 'w')

        # Check if the current operating system is Windows
        is_windows = (sys.platform == "win32")

        if is_windows:
            # If on Windows, apply the patch temporarily
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            try:
                # Load the model with the patch applied
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=resource_path('ai-models/2024-11-00/best.pt'))
            finally:
                # CRITICAL: Always restore the original class, even if loading fails
                pathlib.PosixPath = temp
        else:
            # If on Linux, macOS, or other systems, load the model directly
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=resource_path('ai-models/2024-11-00/best.pt'))
        
    
    def __call__(self, *args, **kwds):
        if self.model is None:
            self.__load()
        return self.model(*args, **kwds)

model = Model()
