### Description

This repository contains an implementation of the methods and algorithms as described in the paper ['On the Motivic Higman Conjecture'](https://arxiv.org/abs/2301.02439). The programs in this repository can be used to compute the virtual classes and the E-polynomials of the G-representation variety of closed orientable surfaces, where G is either the group of upper triangular matrices, or the group of unipotent upper triangular matrices. See the paper for the definitions and details.

This implementation is written in [Python](https://www.python.org) and makes use of the [SymPy](https://www.sympy.org) package for symbolic computations.

### Usage

To run the arithmetic method, go to the folder `Arithmetic method` and run one of the following two commands:
```bash
python3 arithmetic_method_Un.py <n = 1, ..., 10> # for the groups of unipotent upper triangular matrices
python3 arithmetic_method_Tn.py <n = 1, ..., 10> # for the groups of upper triangular matrices
```

To run the TQFT method, go to the folder `TQFT method` and run one of the following two commands:
```bash
python3 tqft_method_Un.py <n = 2, 3, 4, 5> # for the groups of unipotent upper triangular matrices
python3 tqft_method_Tn.py <n = 2, 3, 4, 5> # for the groups of upper triangular matrices
```

Note that the TQFT method computes a matrix Z, which is stored in `TQFT_method/data/Un_Z.txt` or `TQFT_method/data/Tn_Z`, respectively. The virtual class of the G-representation variety of a closed orientable surface of genus g is given by the first entry of the g-th power of Z. One has to diagonalize Z in order to obtain a formula in g (e.g. by hand or using MATLAB, as SymPy is not optimized for diagonalizing symbolic matrices). The notebook `TQFT_method/Compute virtual classes.ipynb` shows how these formulas are obtained.

### Files

An explanation of the various files in this repository.

- `Arithmetic method/`
  - `arithmetic_method_Tn.py`: script which applies the arithmetic method to the groups of n x n upper triangular matrices for n = 1, ..., 10
  - `arithmetic_method_Un.py`: script which applies the arithmetic method to the groups of n x n unipotent upper triangular matrices for n = 1, ..., 10
  - `counting_representations.py`: contains the core functions of the arithmetic method
  - `Counting representations.ipynb`: Python notebook containing the same code as the file above, including some examples of how to compute the representation zeta function of various algebraic groups
  - `zeta.py`: contains code to simplify the representation zeta functions
  - `grothendieck_solver.py`: contains code to count points of varieties over finite fields as polynomials in q
- `TQFT method/`
	- `tqft_method_Tn.py`: script which applies the TQFT method to the groups of n x n upper triangular matrices for n = 2, ..., 5
	- `tqft_method_Un.py`: script which applies the TQFT method to the groups of n x n unipotent upper triangular matrices for n = 2, ..., 5
	- `grothendieck_solver.py`: contains code to compute the virtual class of quasi-affine varieties in the Grothendieck ring of varieties
	- `Compute virtual classes.ipynb`: Python notebook which shows how to obtain the virtual classes of the G-representation varieties from the computed matrices Z
	- `data/`: folder containing following data: representatives for unipotent conjugacy classes, diagonal patterns, coefficients E\_ij  and F\_ijk, and matrices Z for every group of interest
	- `data_clean/`:  folder containing only the data of the unipotent representatives and the diagonal patterns. If the data folder is replaced by this folder, then running the TQFT method will recompute the coefficients E\_ij and F\_ijk.

### Results

The programs for both methods have been run, and the results can be found in folders `TQFT method/Results` and `Arithmetic method/Results`.
