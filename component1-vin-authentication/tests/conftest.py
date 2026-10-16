import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The service uses paths relative to the component folder.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
