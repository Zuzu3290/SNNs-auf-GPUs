import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root → skeleton, event_data_workflow
sys.path.insert(0, str(Path(__file__).parent.parent))          # src/ → learning, compiler

from .frameworks.snn_torch import SNN_TORCH
from .frameworks.snn_norse import SNN_NORSE
from .frameworks.snn_spikingjelly import SNN_SJ
from .inference import SNNTester
from .training import SNNTrainer

__all__ = ['SNN_TORCH', 'SNN_NORSE', 'SNN_SJ', 'SNNTester', 'SNNTrainer']