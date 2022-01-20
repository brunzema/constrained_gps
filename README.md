# constrained_gps

Reimplementation using **GPyTorch** of the linear inequality constraints algorithm by [Agrell (2019)](https://arxiv.org/pdf/1901.03134.pdf) especailly useful to combining it with Bayesian optimization and BoTorch. The **original NumPy implementation** by the author can be found [here](https://github.com/cagrell/gp_constr).

The main features of this algorithm are:

- stable numerical compuation using Cholesky factors
- suitable for multiple constraints (not implemented yet)
- fast sampling when combining it with [Botev (2016)](https://arxiv.org/pdf/1603.04166.pdf) as implementent [here](https://github.com/brunzema/truncated-mvn-sampler).

Feel free to use this code, but don't forget to cite [Agrell (2019)](https://arxiv.org/pdf/1901.03134.pdf)!

## Main difference to the orignal Python implementation
The Python implementation provided here uses the efficient pipeline of GPyTorch in contrast to the NumPy implemenation by the author of the algorithm.
This is especially useful, for combining the approach with Bayesian optimization and BoTorch.
So far only convexity constraints are possible, but extentions to monotonicity constraints and bounds will follow.

## Dependancies
- [numpy](https://numpy.org)
- [PyTorch](https://pytorch.org)
- [GPyTorch](https://gpytorch.ai)
- [BoTorch](https://botorch.org) (for the use of constraints + Bayesian optimization)

## Example
There is a jupyter-notebook provided with a 1D example.

# Reference
The implementation is based on the [implemenation by author](https://github.com/cagrell/gp_constr).
Furthermore, the implemenation uses the sampling algorithm by [Botev (2016)](https://arxiv.org/pdf/1603.04166.pdf).

[Agrell, C. (2019), **Gaussian processes with lin-
ear operator inequality constraints**. Journal of Machine
Learning Research, 20(135):1–36](https://arxiv.org/pdf/1901.03134.pdf)

[Botev, Z. I., (2016), **The normal law under linear restrictions: simulation and estimation via minimax tilting**,
Journal of the Royal Statistical Society Series B, 79, issue 1, p. 125-148](https://arxiv.org/pdf/1603.04166.pdf)
