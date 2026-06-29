"""
benchmark.py
============
Validation and benchmarking script for MAPA-LOBSTER.

What this script does
---------------------
1. Unit-tests: verifies LOBSTER is zero-mean (E[error]=0) over 10k samples
2. Conv layer test: compares MAPA vs exact output on random int8 data
3. CNN simulation: runs a simple LeNet-style CNN on synthetic MNIST-like
    data and reports output divergence and approximation rate.
    (Replace synthetic data with real MNIST/CIFAR-10 when PyTorch is available.)

Usage
-----
    python benchmark.py

Expected output (approximate)
------------------------------
    [UNIT TEST] LOBSTER mean error over 10000 samples: ~0.00  ✓
    [CONV TEST]  Max output divergence: <5%
    [CNN TEST]   Approximation rate: ~65-75%
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from lobster_multiplier import LobsterMultiplier, magnitude_score, lob_position
from mapa_controller import MapaController
from mapa_mac import MapaMacUnit, MapaConv2D, ExactConv2D


# ════════════════════════════════════════════
#  1. UNIT TEST: Zero-mean error property
# ════════════════════════════════════════════

def test_zero_mean_error(n_samples=10_000, seed=42):
    print("\n" + "="*55)
    print("  UNIT TEST: LOBSTER zero-mean error property")
    print("="*55)

    rng = np.random.default_rng(seed)
    mult = LobsterMultiplier(lfsr_seed=0xACE1)

    errors = []
    for _ in range(n_samples):
        a = int(rng.integers(-128, 128))
        b = int(rng.integers(-128, 128))
        approx = mult.multiply(a, b)
        exact  = a * b
        errors.append(approx - exact)

    mean_err = np.mean(errors)
    std_err  = np.std(errors)
    print(f"  Samples          : {n_samples}")
    print(f"  Mean error       : {mean_err:.4f}  (target ≈ 0.00)")
    print(f"  Std  error       : {std_err:.2f}")
    passed = abs(mean_err) < 1.0
    print(f"  Zero-mean test   : {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ════════════════════════════════════════════
#  2. LOB / magnitude score sanity check
# ════════════════════════════════════════════

def test_lob_and_magnitude():
    print("\n" + "="*55)
    print("  UNIT TEST: LOB position and magnitude score")
    print("="*55)

    cases = [
        (0,   0),
        (1,   0),
        (2,   1),
        (4,   2),
        (127, 6),
        (128, 7),
        (255, 7),
    ]
    all_ok = True
    for val, expected_lob in cases:
        got = lob_position(val)
        ok = (got == expected_lob)
        if not ok:
            all_ok = False
        print(f"  lob_position({val:3d}) = {got}  {'✓' if ok else f'✗ expected {expected_lob}'}")

    # Magnitude score boundary tests
    M_low  = magnitude_score(1, 1)      # very small operands → M near 0
    M_high = magnitude_score(127, 127)  # large operands → M near 1
    print(f"\n  magnitude_score(1,   1)   = {M_low:.4f}  (should be near 0)")
    print(f"  magnitude_score(127,127)  = {M_high:.4f}  (should be near 1)")
    return all_ok and (M_low < 0.2) and (M_high > 0.8)


# ════════════════════════════════════════════
#  3. Conv layer output divergence test
# ════════════════════════════════════════════

def test_conv_divergence(seed=7):
    print("\n" + "="*55)
    print("  CONV TEST: MAPA vs Exact output divergence")
    print("="*55)

    rng = np.random.default_rng(seed)

    # Small 4x4 conv, 1 input channel, 4 output channels, 3x3 kernel
    out_ch, in_ch, kH, kW = 4, 1, 3, 3
    H, W = 8, 8

    weights = rng.integers(-64, 64, size=(out_ch, in_ch, kH, kW), dtype=np.int8)
    bias    = rng.integers(-100, 100, size=(out_ch,), dtype=np.int32)
    x       = rng.integers(-128, 127, size=(in_ch, H, W), dtype=np.int8)

    exact_layer = ExactConv2D(weights, bias)
    mapa_layer  = MapaConv2D(weights, bias, lfsr_seed=0xACE1)

    out_exact = exact_layer.forward(x).astype(np.float32)
    out_mapa  = mapa_layer.forward(x).astype(np.float32)

    # Relative error where exact output is non-zero
    nonzero = np.abs(out_exact) > 1
    rel_err = np.abs(out_mapa[nonzero] - out_exact[nonzero]) / np.abs(out_exact[nonzero])

    mean_rel = float(np.mean(rel_err)) * 100
    max_rel  = float(np.max(rel_err))  * 100
    approx_rate = mapa_layer._mac.controller.approximation_rate * 100

    print(f"  Approx rate      : {approx_rate:.1f}%")
    print(f"  Mean rel error   : {mean_rel:.2f}%")
    print(f"  Max  rel error   : {max_rel:.2f}%")
    passed = mean_rel < 10.0
    print(f"  Divergence test  : {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ════════════════════════════════════════════
#  4. Mini CNN simulation (synthetic data)
# ════════════════════════════════════════════

def simulate_mini_cnn(seed=99):
    """
    Simulate a tiny LeNet-style conv block:
      Conv1(1→4, 3x3) → ReLU → Conv2(4→8, 3x3) → Global Average Pool

    Input: 28x28 int8 (like MNIST after quantisation to INT8)
    """
    print("\n" + "="*55)
    print("  CNN SIM: Mini LeNet block (synthetic MNIST-like)")
    print("="*55)

    rng = np.random.default_rng(seed)
    N_SAMPLES = 20

    w1 = rng.integers(-64, 64, (4, 1, 3, 3),  dtype=np.int8)
    b1 = rng.integers(-50, 50, (4,),           dtype=np.int32)
    w2 = rng.integers(-64, 64, (8, 4, 3, 3),  dtype=np.int8)
    b2 = rng.integers(-50, 50, (8,),           dtype=np.int32)

    def relu(x):
        return np.maximum(0, x)

    def gap(x):   # global average pool
        return x.mean(axis=(1, 2))

    exact_outs, mapa_outs = [], []
    approx_rates = []

    for _ in range(N_SAMPLES):
        img = rng.integers(-128, 127, (1, 28, 28), dtype=np.int8)

        # ── Exact path ──
        e1 = relu(ExactConv2D(w1, b1).forward(img))
        e1_q = np.clip(e1 // 64, -128, 127).astype(np.int8)
        e2 = ExactConv2D(w2, b2).forward(e1_q)
        exact_outs.append(gap(e2))

        # ── MAPA path ──
        mc1 = MapaConv2D(w1, b1, lfsr_seed=0xACE1)
        a1  = relu(mc1.forward(img))
        a1_q = np.clip(a1 // 64, -128, 127).astype(np.int8)
        mc2  = MapaConv2D(w2, b2, lfsr_seed=0xBEEF)
        a2   = mc2.forward(a1_q)
        mapa_outs.append(gap(a2))
        approx_rates.append(
            (mc1._mac.controller.approximation_rate +
             mc2._mac.controller.approximation_rate) / 2
        )

    exact_arr = np.array(exact_outs, dtype=np.float32)
    mapa_arr  = np.array(mapa_outs,  dtype=np.float32)

    nz = np.abs(exact_arr) > 1
    rel_err = np.abs(mapa_arr[nz] - exact_arr[nz]) / (np.abs(exact_arr[nz]) + 1e-6)

    avg_approx = np.mean(approx_rates) * 100
    mean_err   = np.mean(rel_err) * 100
    print(f"  Samples          : {N_SAMPLES}")
    print(f"  Avg approx rate  : {avg_approx:.1f}%")
    print(f"  Mean output err  : {mean_err:.2f}%")
    print(f"  Note: With real PyTorch + MNIST, run full training loop.")
    print(f"        Replace ExactConv2D/MapaConv2D with torch.nn.Conv2d + custom autograd.")
    return mean_err < 15.0


# ════════════════════════════════════════════
#  5. Hardware test vector generation
#     (for Verilog testbench comparison)
# ════════════════════════════════════════════

def generate_testvectors(n=256, seed=5, outfile="testvectors.txt"):
    """
    Generate (A, B, exact_product, lobster_approx) test vectors.
    Hardware team loads these into the Verilog testbench to compare
    RTL output against software reference.
    """
    print("\n" + "="*55)
    print("  GENERATING hardware test vectors →", outfile)
    print("="*55)

    rng  = np.random.default_rng(seed)
    mult = LobsterMultiplier(lfsr_seed=0xACE1)

    lines = ["# A(dec)  B(dec)  A(hex)  B(hex)  exact   lobster_approx"]
    for _ in range(n):
        a = int(rng.integers(-128, 128))
        b = int(rng.integers(-128, 128))
        exact  = a * b
        approx = mult.multiply(a, b)
        # Convert signed to unsigned 8-bit hex for Verilog
        ah = a & 0xFF
        bh = b & 0xFF
        lines.append(f"{a:5d}  {b:5d}  {ah:02X}  {bh:02X}  {exact:7d}  {approx:7d}")

    outpath = os.path.join(os.path.dirname(__file__), outfile)
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written {n} vectors to {outpath}")


# ════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════

if __name__ == "__main__":
    results = {}
    results["zero_mean"]     = test_zero_mean_error()
    results["lob_score"]     = test_lob_and_magnitude()
    results["conv_diverge"]  = test_conv_divergence()
    results["cnn_sim"]       = simulate_mini_cnn()
    generate_testvectors()

    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    for name, passed in results.items():
        print(f"  {name:<20s}  {'PASS ✓' if passed else 'FAIL ✗'}")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)