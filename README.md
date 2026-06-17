# Objective

This code provides experimental implementations of:
- A generic GMRES function allowing for approximate matrix-vector multiplications and vector truncations/compressions. This can be used, e.g., when vectors are represented as tensor networks.
- A basic DMRG-style / MALS solver for linear systems. No controlled bond expansion / randomized projection implemented.
- An implementation of the 1-site AMEn linear solver. It is intended for positive definite operators. Daring scientists can still try it out for non-positive operators.

# Getting started
All you need to do is `using Pkg; Pkg.instantiate()` in the Julia REPL.
