# saw

This Python module contains tools for simulating and handling self-avoiding walks (SAWs).
Written for Python 3.

## Notebook

An introduction to Markov chains and MCMC (including the Metropolis-Hastings algorithm),
the self-avoiding walk (weak and strict), and the pivot algorithm can all be found in the
Jupyter notebook saw-simulation.ipynb.

## Demo

Start by importing the module with
```python
import saw
```

For a live demonstration of 10 iterations of the pivot algorithm on 100-step SAW, run:
```python
saw.demo(100, 10)
```
