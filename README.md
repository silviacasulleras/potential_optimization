# potential_optimization

 Potential Optimization for Levitated Nanoparticles: Maximizing quantum features via static potentials.

## Overview

This repository provides a numerical framework for the **optimization of static potentials** to generate large delocalization and non-Gaussian quantum states of levitated nanoparticles under decoherence.  
The optimization maximizes genuinely quantum properties of the system, quantified via the logarithmic negativity of the Wigner function.  

Once the optimal static potential is determined, the time evolution of the quantum state is analyzed to study the generation and robustness of non-classical features.

For technical details of the algorithm, please refer to the official paper:  
S. Casulleras, P. T. Grochowski, and O. Romero-Isart,  
*"Optimization of Static Potentials for Large Delocalization and Non-Gaussian Quantum Dynamics of Levitated Nanoparticles Under Decoherence,"* Phys. Rev. A 110, 033511 (2024).

---

## Authors and Acknowledgments

This code was developed by Katja Kustura (early stages) and Silvia Casulleras at the Institute for Quantum Optics and Quantum Information (IQOQI), Austrian Academy of Sciences and University of Innsbruck, Austria.  
We thank our collaborators Piotr Grochowski and Oriol Romero-Isart for their involvement in the project.

---

## Methodology

- Parametrization of static potentials 
- Time-dependent numerical propagation of the quantum state  
- Phase-space analysis** via Wigner function and logarithmic negativity  
- Gradient-based and constrained optimization under physical constraints  
- Stability and robustness analysis

---

## Usage

This program requires the following Python packages:  
`numpy`, `scipy`, `numba`, and `numbalsoda`.  

To run the program:

```bash
python main.py
