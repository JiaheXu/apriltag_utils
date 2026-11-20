import sys
from pathlib import Path

# Make direct `pytest` and ament/colcon runs work without an editable pip install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
