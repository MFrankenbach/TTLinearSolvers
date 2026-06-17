# Objective

This code provides experimental implementations of:
- A generic GMRES function allowing for approximate matrix-vector multiplications and vector truncations/compressions. This can be used, e.g., when vectors are represented as tensor networks.
- A basic DMRG-style / MALS solver for linear systems. No controlled bond expansion / randomized projection implemented.
- An implementation of the 1-site AMEn linear solver. It is intended for positive definite operators. Daring scientists can still try it out for non-positive operators.

# Getting started
All you need to do is `using Pkg; Pkg.instantiate()` in the Julia REPL.

# Entry points for key features
- GMRES: `function gmres`
- AMEn: `function amen[!]`
- DMRG/MALS: `function halfsweep!` without AMEn update (and `nsite>1`)

# Remarks
- In the AMEn bond expansion, we move the orthogonality center and only then expand the bond using the residual. Otherwise, the QR decomposition would just undo the expansion. This currently leads to above-worst-case bond dimensions, which could be mended by an additional QR sweep after each local update.
- We recompute the full residual after each local update. This is more reliable, but in the AMEn paper, it is pointed out that the residual can be updated on the fly via simultaneous sweeps along the solution and residual tensor trains (cheaper, but less reliable).
- Finding an ideal truncation scheme in TT-GMRES is non-trivial: too conservative truncations are obviously inefficient, but high singular value cutoffs can lead to exploding bond dimensions in Krylov vectors! The choice of cutoffs is not fully fine-tuned in the current version.
