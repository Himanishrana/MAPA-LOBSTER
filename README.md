<div align="center">

# MAPA-LOBSTER

### Magnitude-Adaptive Probabilistic Approximation with Leading-One-Bit Segmented Truncation & Error Redistribution

**A Research-Oriented Adaptive Approximate Computing Framework for FPGA-Based AI Accelerators**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Verilog](https://img.shields.io/badge/Verilog-HDL-orange)
![Vivado](https://img.shields.io/badge/Xilinx-Vivado-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-purple)

</div>

---

## Overview

Modern AI accelerators execute billions of Multiply-Accumulate (MAC) operations during neural network inference. Although exact arithmetic guarantees numerical accuracy, it significantly increases power consumption, switching activity, and hardware resource utilization—making it inefficient for low-power edge AI devices.

**MAPA-LOBSTER** introduces an adaptive approximate computing framework that dynamically balances computational accuracy and energy efficiency. Instead of approximating every multiplication, the proposed architecture evaluates the significance of each operation at runtime and selectively applies approximation only when the expected impact on inference accuracy is low.

The framework consists of two complementary components:

- **MAPA (Magnitude-Adaptive Probabilistic Approximation)** – A runtime controller that estimates operand significance using Leading-One-Bit (LOB) analysis and probabilistically selects between exact and approximate multiplication.

- **LOBSTER (Leading-One-Bit Segmented Truncation & Error Redistribution)** – A lightweight approximate multiplier that performs adaptive truncation and stochastic error redistribution to minimize accumulated approximation bias.

Together, these components form a hardware-friendly adaptive MAC architecture suitable for FPGA-based CNN accelerators.

---

# Key Features

- Runtime adaptive approximation
- Operand significance-aware computation
- Leading-One-Bit (LOB) magnitude estimation
- Probabilistic approximation control
- Adaptive truncation strategy
- Stochastic error redistribution
- Near zero-mean approximation error
- FPGA-oriented Verilog RTL implementation
- Python reference implementation
- Vivado synthesis and RTL verification
- CNN functional validation
- Interactive visualization dashboard

---

# How MAPA Works

For every multiplication,

1. Operand magnitudes are estimated using Leading-One-Bit encoding.
2. A normalized significance score is computed.
3. The significance score determines the approximation probability.
4. An LFSR-based stochastic controller decides whether the multiplication should follow the exact or approximate datapath.

The significance score is defined as

\[
M(A,B)=\frac{LOB(A)+LOB(B)}{12}
\]

Approximation probability:

\[
P_{approx}=1-M(A,B)
\]

Low-significance operations are approximated more frequently, while high-significance operations are protected using exact computation.

---

# How LOBSTER Works

The approximate multiplier performs three major steps:

### Adaptive Truncation

The truncation depth is determined dynamically according to operand significance.

\[
k=LOB(min(|A|,|B|))
\]

---

### Parity Extraction

The discarded bits are compressed into a parity signal.

\[
P=t_0 \oplus t_1 \oplus \cdots \oplus t_k
\]

---

### Error Redistribution

Instead of permanently discarding truncation error, a lightweight stochastic correction is applied.

The objective is to produce

\[
E(error)\approx0
\]

thereby reducing accumulated MAC bias during CNN inference.

---

# Software Validation

The Python implementation validates

- Zero-Mean Error Property
- MAPA Runtime Controller
- CNN Functional Correctness
- Approximation Statistics
- Hardware Test Vector Generation

---

# Hardware Implementation

The proposed architecture has been implemented in Verilog HDL and verified using Xilinx Vivado.

The hardware implementation includes

- MAPA Controller
- LOBSTER Multiplier
- Exact Multiplier
- MAC Datapath
- RTL Simulation
- Functional Verification
- FPGA Synthesis

*(Insert Vivado RTL Screenshot here)*

---

# Experimental Results

The framework is evaluated using software simulation and RTL hardware implementation.

Example metrics include:

| Metric | Result |
|----------|---------|
| Approximation Rate | ~50% |
| Mean Relative Error | ~0.21% |
| Functional Verification | PASS |
| Zero-Mean Validation | PASS |
| FPGA Synthesis | Successful |

*(Insert Pareto Curve here)*

*(Insert Benchmark Graphs here)*

---

# Interactive Dashboard

The repository also includes an interactive dashboard for visualizing runtime approximation behavior.

Dashboard features include:

- Approximation rate
- Magnitude score distribution
- Runtime path selection
- Error statistics
- Energy estimation
- Hardware synthesis metrics
- CNN benchmark statistics

*(Insert Dashboard Screenshot here)*

---

# Applications

- FPGA AI Accelerators
- Edge AI
- TinyML
- Approximate Computing
- CNN Inference
- Low-Power Embedded Systems
- Hardware-Aware Machine Learning

---

# Future Work

- ASIC implementation
- Systolic array integration
- CIFAR-10 / ImageNet evaluation
- INT4 / BF16 support
- Transformer acceleration
- Dynamic probability scheduling
- Reinforcement learning-based approximation policies

---

# Research Contributions

The major contributions of this work include

- Runtime significance-aware approximation
- Operand-dependent approximation probability
- Adaptive truncation strategy
- Stochastic error redistribution
- FPGA-compatible approximate MAC architecture
- Hardware-software co-validation framework

---

# Installation

```bash
git clone https://github.com/<your-username>/MAPA-LOBSTER.git

cd MAPA-LOBSTER

pip install -r requirements.txt
```

Run benchmark

```bash
python benchmark/benchmark.py
```

---

# Citation

If this work contributes to your research, please cite the repository.

```bibtex
@software{rana2026mapalobster,
  author = {Himanish Rana},
  title = {MAPA-LOBSTER: Magnitude-Adaptive Probabilistic Approximation with Leading-One-Bit Segmented Truncation & Error Redistribution},
  year = {2026},
  url = {https://github.com/<your-username>/MAPA-LOBSTER}
}
```

---

# Disclaimer

MAPA-LOBSTER is a **research prototype** developed as part of an undergraduate research project in FPGA-based approximate computing.

The software models, hardware implementations, benchmark results, and synthesis reports are intended for **educational, experimental, and research purposes**. Reported results may vary depending on FPGA device, synthesis settings, datasets, quantization schemes, and compiler optimizations.

This repository is **not intended for production deployment**. The proposed architecture should be independently validated before being used in safety-critical or commercial systems.

---

# License

This project is licensed under the **MIT License**.

See the **LICENSE** file for details.

---

## Project Team

**Project:** MAPA-LOBSTER

**Developed by**
- Himanish Rana
- Teammate 1
- Teammate 2

**Project Supervisor**
- Dr. Bharat Garg

**Institution**
Department of Electrical and Computer Engineering  
Thapar Institute of Engineering & Technology, Patiala, India

## Acknowledgements

This work was carried out as an undergraduate capstone project at **Thapar Institute of Engineering & Technology**.

The authors sincerely thank **Dr. Bharat Garg** for his valuable guidance, technical discussions, and continuous support throughout the development of the MAPA-LOBSTER framework.

We also acknowledge the Department of Electrical and Computer Engineering and the open-source FPGA, Python, and machine learning communities whose tools and resources greatly contributed to this work.

---

<div align="center">

### ⭐ If you find this project useful, consider giving it a star!

**Towards Energy-Efficient Adaptive AI Hardware Acceleration**

</div>
