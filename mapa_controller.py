"""
mapa_controller.py
==================
MAPA: Magnitude-Adaptive Probabilistic Approximation gate.

The controller decides, per multiplication, whether to route operands
through the EXACT path or the LOBSTER approximate path.
"""

import numpy as np
from lobster_multiplier import LobsterMultiplier, magnitude_score


class MapaController:
    """
    Runtime gate that selects exact vs. LOBSTER multiply per operation.

    Decision rule
    -------------
    M = magnitude_score(a, b)        # in [0, 1]
    P_approx = 1 - M

    Draw uniform random r ~ U[0,1]:
        if r < P_approx  → use LOBSTER
        else             → use exact

    High-magnitude pairs (M near 1) almost always go exact.
    Low-magnitude pairs (M near 0) almost always go approximate.
    """

    def __init__(self, lfsr_seed: int = 0xACE1):
        self._lobster = LobsterMultiplier(lfsr_seed=lfsr_seed)
        self._rng = np.random.default_rng(lfsr_seed)

        # Statistics
        self.total_ops = 0
        self.approx_ops = 0
        self.exact_ops = 0

    def multiply(self, a: int, b: int) -> int:
        """
        Route one multiply through MAPA gate.
        Inputs: 8-bit signed integers in [-128, 127].
        Returns: 16-bit signed approximate product.
        """
        a = int(np.clip(a, -128, 127))
        b = int(np.clip(b, -128, 127))

        M = magnitude_score(a, b)
        P_approx = 1.0 - M

        r = self._rng.uniform(0.0, 1.0)
        self.total_ops += 1

        if r < P_approx:
            self.approx_ops += 1
            return self._lobster.multiply(a, b)
        else:
            self.exact_ops += 1
            return int(a) * int(b)           # exact Python integer multiply

    @property
    def approximation_rate(self) -> float:
        """Fraction of operations that were approximated."""
        if self.total_ops == 0:
            return 0.0
        return self.approx_ops / self.total_ops

    def reset_stats(self):
        self.total_ops = 0
        self.approx_ops = 0
        self.exact_ops = 0
        self._lobster.reset_error_stats()