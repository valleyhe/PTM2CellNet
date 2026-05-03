import os
import sys

os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
