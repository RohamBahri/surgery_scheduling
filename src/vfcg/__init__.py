"""Exact VFCG training stack.

Heavy solver modules are intentionally imported from their concrete files
(``solver.py``, ``master.py``, ``oracle.py``) to keep this package entry point
lightweight.
"""

from src.core.config import VFCGConfig
from src.vfcg.types import CertificationResult, OracleResult, VFCGIteration, VFCGResult

__all__ = [
    "CertificationResult",
    "OracleResult",
    "VFCGConfig",
    "VFCGIteration",
    "VFCGResult",
]
