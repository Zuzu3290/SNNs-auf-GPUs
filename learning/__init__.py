import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton, event_data_workflow, learning, frameworks

from .inference import SNNTester
from .training import SNNTrainer

__all__ = ['SNNTester', 'SNNTrainer']