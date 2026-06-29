"""
mapa_mac.py
===========
MAPA-LOBSTER MAC (Multiply-Accumulate) unit and Conv2D layer simulation.

This is the functional model that validates hardware behaviour before
the Verilog design is complete.
"""

import numpy as np
from mapa_controller import MapaController


# ─────────────────────────────────────────────
#  Single MAC unit
# ─────────────────────────────────────────────

class MapaMacUnit:
    """
    One Multiply-Accumulate unit using the MAPA-LOBSTER gate.

    mac(inputs, weights) = sum_i( MAPA_multiply(inputs[i], weights[i]) )
    """

    def __init__(self, lfsr_seed: int = 0xACE1):
        self.controller = MapaController(lfsr_seed=lfsr_seed)

    def mac(self, inputs: np.ndarray, weights: np.ndarray) -> int:
        """
        Compute dot product using MAPA-gated multiplications.

        Parameters
        ----------
        inputs  : 1D array of int8
        weights : 1D array of int8, same length

        Returns
        -------
        int : accumulated sum (no overflow clipping — matches hardware 32-bit acc)
        """
        assert inputs.shape == weights.shape, "Shapes must match"
        acc = 0
        for a, w in zip(inputs.flatten(), weights.flatten()):
            acc += self.controller.multiply(int(a), int(w))
        return acc


# ─────────────────────────────────────────────
#  Conv2D layer using MAPA MACs
# ─────────────────────────────────────────────

class MapaConv2D:
    """
    Simplified Conv2D layer with MAPA-LOBSTER approximate MACs.
    Supports single-batch inference for validation.

    Parameters
    ----------
    weights : np.ndarray  shape (out_ch, in_ch, kH, kW)  int8
    bias    : np.ndarray  shape (out_ch,)                 int32
    stride  : int
    padding : int
    """

    def __init__(self,
                 weights: np.ndarray,
                 bias: np.ndarray,
                 stride: int = 1,
                 padding: int = 0,
                 lfsr_seed: int = 0xACE1):
        self.weights = weights.astype(np.int8)
        self.bias = bias.astype(np.int32)
        self.stride = stride
        self.padding = padding
        self.out_ch, self.in_ch, self.kH, self.kW = weights.shape
        self._mac = MapaMacUnit(lfsr_seed=lfsr_seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (in_ch, H, W) int8
        Returns (out_ch, H_out, W_out) int32
        """
        in_ch, H, W = x.shape
        if self.padding > 0:
            x = np.pad(x, ((0,0),(self.padding,self.padding),(self.padding,self.padding)))
        H_out = (H + 2*self.padding - self.kH) // self.stride + 1
        W_out = (W + 2*self.padding - self.kW) // self.stride + 1

        out = np.zeros((self.out_ch, H_out, W_out), dtype=np.int32)
        for oc in range(self.out_ch):
            for oh in range(H_out):
                for ow in range(W_out):
                    h0 = oh * self.stride
                    w0 = ow * self.stride
                    patch = x[:, h0:h0+self.kH, w0:w0+self.kW]   # (in_ch, kH, kW)
                    out[oc, oh, ow] = (
                        self._mac.mac(patch, self.weights[oc]) + self.bias[oc]
                    )
        return out


# ─────────────────────────────────────────────
#  Exact reference layer (for comparison)
# ─────────────────────────────────────────────

class ExactConv2D:
    """Same as MapaConv2D but uses exact integer arithmetic — reference baseline."""

    def __init__(self, weights, bias, stride=1, padding=0):
        self.weights = weights.astype(np.int8)
        self.bias    = bias.astype(np.int32)
        self.stride, self.padding = stride, padding
        self.out_ch, self.in_ch, self.kH, self.kW = weights.shape

    def forward(self, x):
        in_ch, H, W = x.shape
        if self.padding > 0:
            x = np.pad(x, ((0,0),(self.padding,self.padding),(self.padding,self.padding)))
        H_out = (H + 2*self.padding - self.kH) // self.stride + 1
        W_out = (W + 2*self.padding - self.kW) // self.stride + 1
        out = np.zeros((self.out_ch, H_out, W_out), dtype=np.int32)
        for oc in range(self.out_ch):
            for oh in range(H_out):
                for ow in range(W_out):
                    h0, w0 = oh*self.stride, ow*self.stride
                    patch = x[:, h0:h0+self.kH, w0:w0+self.kW]
                    out[oc,oh,ow] = int(np.sum(patch.astype(np.int32) *
                                               self.weights[oc].astype(np.int32))) + self.bias[oc]
        return out