# Simple MCMC for Self-Avoiding Walk

The file `saw.py` contains methods for simulating and handling self-avoiding walks (SAWs) using Markov Chain Monte Carlo (MCMC).
See the Jupyter Notebook (information below) for explanations of these terms.
Written for Python 3.

## Notebook

An introduction to Markov chains and MCMC (including the Metropolis-Hastings algorithm),
the self-avoiding walk (weak and strict), and the pivot algorithm can all be found in the
Jupyter notebook `saw-simulation.ipynb`.

## Demo

To generate an animation of 100 iterations of the pivot algorithm for a 50-step SAW, run the following:
```python
import saw
saw.demo(50, 100)
```
