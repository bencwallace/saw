# saw

This Python module contains tools for simulating and handling self-avoiding walks (SAWs).
Written for Python 3.

## Code examples

Start by importing the module with
```python
import saw
```

For a live demonstration of 10 iterations of the pivot algorithm on 100-step SAW, run:
```python
saw.demo(100, 10)
```
and press Enter to advance the iteration.

To initialize a new 100-step SAW as a straight line, run:
```python
s = saw.saw(100)
```

To pivot the walk, a rotation matrix is needed. Define the 2 by 2 matrix corresponding to a 90 degree counterclockwise rotation as follows:
```python
import numpy as np
r = np.array([[0, -1], [1, 0]])
```
Plot the walk with
```python
saw.plotwalk(s)
```
Then pivot the walk at the halfway point with
```python
s.pivot(50, r)
```
and plot the pivotted walk as before:
```python
saw.plotwalk(s)
```

To perform 1000 iterations of the pivot algorithm, run:
```python
s.mix(1000)
```
Then plot the walk again to see the result. This time, let's use a different line style:
```python
saw.plotwalk(s, '-')
```
