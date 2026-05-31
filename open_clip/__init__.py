from .src.open_clip import *  # re-export local OpenCLIP API

try:
	from .src.open_clip.version import __version__
except Exception:
	__version__ = "0.0.0-local"
